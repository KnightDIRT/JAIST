#30fps
"""
Optimized real-time fisheye undistort + Depth-Anything (small) pipeline.
Key changes from original:
 - GPU preprocessing path (bypass slow HF processor when possible)
 - Preallocated input tensor reused each frame
 - Exponential smoothing (no frame buffer)
 - Separate undistort thread to overlap CPU remap with GPU inference
 - Optionally use OpenCL (cv2.UMat) if available
 - Precompute undistort maps for actual camera capture size (avoid per-frame resize)
 - Lightweight profiling and safe fallback to original processor if needed
 - Reduced OpenCV thread usage to avoid contention with PyTorch
"""

import argparse
import json
import threading
import queue
import time
import sys
import os
import math

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForDepthEstimation

# reduce OpenCV thread usage to avoid contention with PyTorch
cv2.setNumThreads(1)


class CameraStream:
    def __init__(self, src=0, backend=None):
        if backend is not None:
            self.cap = cv2.VideoCapture(src, backend)
        else:
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")
        # read one frame to determine resolution
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


class UndistortDepth:
    def __init__(self,
                 calib_file,
                 model_name="LiheYoung/depth-anything-small-hf",
                 device=None,
                 display_size=(640, 360),
                 inference_size=(384, 384),
                 remap_interp=cv2.INTER_LINEAR,
                 use_opencl=False):

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[INFO] Using device: {self.device}")

        # load calibration
        try:
            with open(calib_file, "r") as f:
                calib = json.load(f)
            self.K = np.array(calib["K"], dtype=np.float64)
            self.D = np.array(calib["D"], dtype=np.float64)
            self.calib_image_size = tuple(calib["image_size"])  # (w,h)
        except Exception as e:
            raise RuntimeError(f"Failed to load calibration file: {e}")

        self.remap_interp = remap_interp
        self.display_w, self.display_h = display_size
        self.inference_w, self.inference_h = inference_size

        # model + processor as a fallback
        print(f"[INFO] Loading model: {model_name}")
        self.processor = None
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)

        # move to device & half precision if GPU
        if self.device.type == "cuda":
            try:
                self.model = self.model.half().to(self.device)
            except Exception:
                self.model = self.model.to(self.device)

            # try to torch.compile where available
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"[WARN] torch.compile skipped/failed: {e}")
        else:
            self.model = self.model.to(self.device)

        self.model.eval()

        # preallocate input tensor (GPU) if cuda
        if self.device.type == "cuda":
            self.input_tensor = torch.zeros((1, 3, self.inference_h, self.inference_w),
                                            device=self.device, dtype=torch.half)
        else:
            self.input_tensor = torch.zeros((1, 3, self.inference_h, self.inference_w),
                                            device=self.device, dtype=torch.float32)

        # smoothing (exponential)
        self.depth_smooth = None
        self.smooth_alpha = 0.3

        # internal flags
        self._use_direct_preproc = True  # attempted fast path; will fallback to HF processor if needed

        # warmup
        self._warmup()

        # undistort precomputation placeholders
        self.map1 = None
        self.map2 = None
        self.using_umat = False
        self.use_opencl = use_opencl

    def init_undistort_maps_for_size(self, target_size):
        """Precompute maps for a given target_size (w,h) to avoid per-frame resize.

        This routine *scales* the intrinsic matrix from the calibration image size to the
        requested target size before building the undistort/rectify maps. If OpenCL/UMat
        is requested and available, the numpy maps are wrapped as UMat objects for
        cv2.remap to use the OpenCL path.
        """
        w, h = target_size
        cal_w, cal_h = self.calib_image_size

        # compute scale factors from calibration size -> target size
        sx = float(w) / float(cal_w)
        sy = float(h) / float(cal_h)

        # scale the intrinsics appropriately (fx, fy, cx, cy)
        K_scaled = self.K.copy()
        K_scaled[0, 0] *= sx   # fx
        K_scaled[0, 2] *= sx   # cx
        K_scaled[1, 1] *= sy   # fy
        K_scaled[1, 2] *= sy   # cy

        # use the scaled intrinsics as the new camera matrix (you can tune this later
        # to use a different "balance" or FOV if desired)
        new_K = K_scaled.copy()

        # compute undistort/rectify maps on the numpy arrays
        map1_np, map2_np = cv2.fisheye.initUndistortRectifyMap(
            K_scaled, self.D, np.eye(3), new_K, (w, h), cv2.CV_32FC1
        )

        # optionally wrap into UMat for OpenCL accelerated remap (if requested)
        if self.use_opencl and hasattr(cv2, 'UMat'):
            try:
                self.map1 = cv2.UMat(map1_np)
                self.map2 = cv2.UMat(map2_np)
                self.using_umat = True
            except Exception as e:
                print(f"[WARN] UMat conversion failed, falling back to numpy maps: {e}")
                self.map1 = map1_np
                self.map2 = map2_np
                self.using_umat = False
        else:
            self.map1 = map1_np
            self.map2 = map2_np
            self.using_umat = False

        print(f"[INFO] init_undistort_maps_for_size: target={(w,h)}, using_umat={self.using_umat}")

    def _warmup(self, n=2):
        if self.device.type != "cuda":
            return
        print("[INFO] Warmup model (2 iterations)...")
        dummy = torch.zeros((1, 3, self.inference_h, self.inference_w),
                            device=self.device, dtype=self.input_tensor.dtype)
        with torch.no_grad():
            for _ in range(n):
                with torch.amp.autocast("cuda"):
                    try:
                        _ = self.model(pixel_values=dummy)
                    except Exception:
                        _ = self.model(dummy)
        torch.cuda.synchronize()
        print("[INFO] Warmup complete.")

    def undistort(self, frame):
        """CPU undistort path using precomputed maps (expects frame in native capture size).
        Returns undistorted BGR uint8 image of same size as map output."""
        if self.map1 is None or self.map2 is None:
            # fallback: resize to calib size and compute on the fly (rare)
            frame_rs = cv2.resize(frame, (self.calib_image_size[0], self.calib_image_size[1]),
                                  interpolation=cv2.INTER_LINEAR)
            und = cv2.remap(frame_rs, *cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K, (self.calib_image_size[0], self.calib_image_size[1]), cv2.CV_32FC1
            ), interpolation=self.remap_interp)
            return und

        if self.using_umat and hasattr(cv2, 'UMat'):
            uframe = cv2.UMat(frame)
            und_um = cv2.remap(uframe, self.map1, self.map2, interpolation=self.remap_interp)
            und = und_um.get()
        else:
            und = cv2.remap(frame, self.map1, self.map2, interpolation=self.remap_interp)
        return und

    def estimate_depth(self, undistorted_bgr, profile=False):
        """Estimate depth for undistorted_bgr. Try fast GPU preprocessing path; fall back to processor if needed."""
        t0 = time.time()
        H, W = undistorted_bgr.shape[:2]

        # resize to inference size once (CPU) - keep as uint8 to avoid unnecessary conversions
        rgb_small = cv2.resize(cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB),
                               (self.inference_w, self.inference_h), interpolation=cv2.INTER_LINEAR)

        # fast path: convert numpy -> tensor and copy into preallocated input
        try:
            if self.device.type == "cuda":
                # create a CPU tensor (pinned) view to avoid extra copies when transferring
                cpu_tensor = torch.from_numpy(rgb_small).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
                # convert and copy to half GPU tensor (in-place)
                self.input_tensor.copy_(cpu_tensor.to(self.device, non_blocking=True).to(dtype=self.input_tensor.dtype))
            else:
                self.input_tensor.copy_(torch.from_numpy(rgb_small).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.input_tensor.dtype).div_(255.0))
        except Exception as e:
            # fallback to HF processor path (may be slower)
            if self.processor is None:
                print(f"[WARN] Direct preprocessing failed ({e}). Falling back to HuggingFace processor.")
                self.processor = AutoProcessor.from_pretrained('LiheYoung/depth-anything-small-hf')
            inputs = self.processor(images=rgb_small, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                with torch.amp.autocast(self.device.type) if self.device.type == 'cuda' else torch.cpu.amp.autocast(enabled=False):
                    out = self.model(**inputs)
            if hasattr(out, 'predicted_depth'):
                depth_t = out.predicted_depth
            else:
                depth_t = list(out.values())[0] if isinstance(out, dict) else out
            depth = depth_t.squeeze().detach().cpu().float().numpy()
            depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

            # smoothing
            if self.depth_smooth is None:
                self.depth_smooth = depth_resized
            else:
                self.depth_smooth = (1 - self.smooth_alpha) * self.depth_smooth + self.smooth_alpha * depth_resized

            # colorize
            depth_norm = cv2.normalize(self.depth_smooth, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = np.clip(depth_norm, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)

            if profile:
                t1 = time.time()
                print(f"[PROFILE] Processor fallback total {(t1-t0)*1000:.1f}ms")
            return depth_color

        # inference fast path using preallocated input tensor
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = self.model(pixel_values=self.input_tensor)
            else:
                out = self.model(pixel_values=self.input_tensor)

        # extract depth
        if hasattr(out, "predicted_depth"):
            depth_t = out.predicted_depth
        else:
            depth_t = list(out.values())[0] if isinstance(out, dict) else out

        depth = depth_t.squeeze().detach().cpu().float().numpy()
        depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # exponential smoothing
        if self.depth_smooth is None:
            self.depth_smooth = depth_resized
        else:
            self.depth_smooth = (1 - self.smooth_alpha) * self.depth_smooth + self.smooth_alpha * depth_resized

        # normalize and colorize
        depth_norm = cv2.normalize(self.depth_smooth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = np.clip(depth_norm, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)

        if profile:
            t1 = time.time()
            print(f"[PROFILE] Fast path total {(t1-t0)*1000:.1f}ms")

        return depth_color


# Undistort worker: reads latest frame from CameraStream and writes latest undistorted frame to a single-item queue
class UndistortWorker:
    def __init__(self, cam: CameraStream, ud: UndistortDepth, out_queue: queue.Queue):
        self.cam = cam
        self.ud = ud
        self.out_q = out_queue
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped:
            ret, frame = self.cam.read()
            if not ret:
                time.sleep(0.005)
                continue
            und = self.ud.undistort(frame)
            # keep only latest frame in queue
            try:
                # empty queue if full
                while not self.out_q.empty():
                    try:
                        self.out_q.get_nowait()
                    except Exception:
                        break
                self.out_q.put_nowait(und)
            except queue.Full:
                pass

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-file", default="Undistort-and-Depth/fisheye_calibration_data.json")
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=640)
    ap.add_argument("--display-height", type=int, default=360)
    ap.add_argument("--inference-width", type=int, default=192 * 3)
    ap.add_argument("--inference-height", type=int, default=192 * 3)
    ap.add_argument("--backend", type=str, default=None, help='cv2 backend e.g. cv2.CAP_DSHOW or cv2.CAP_V4L2')
    ap.add_argument("--use-opencl", action='store_true')
    ap.add_argument("--profile", action='store_true')
    args = ap.parse_args()

    # create camera thread
    try:
        backend = getattr(cv2, args.backend) if args.backend else None
        cam = CameraStream(src=args.camera_id, backend=backend)
    except Exception as e:
        print(f"[ERROR] Camera init failed: {e}")
        sys.exit(1)

    # prepare undistort+depth
    ud = UndistortDepth(
        calib_file=args.calib_file,
        display_size=(args.display_width, args.display_height),
        inference_size=(args.inference_width, args.inference_height),
        use_opencl=args.use_opencl,
    )

    # compute undistort maps at actual capture size to avoid per-frame resize
    try:
        cap_w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if cap_w > 0 and cap_h > 0:
            ud.init_undistort_maps_for_size((cap_w, cap_h))
        else:
            # fallback to calibration size
            ud.init_undistort_maps_for_size((ud.calib_image_size[0], ud.calib_image_size[1]))
    except Exception as e:
        print(f"[WARN] Failed to init undistort maps for capture size: {e}")
        ud.init_undistort_maps_for_size((ud.calib_image_size[0], ud.calib_image_size[1]))

    # single-item queue for undistorted frames (latest only)
    und_q = queue.Queue(maxsize=1)
    und_worker = UndistortWorker(cam, ud, und_q)

    fps_ema = 0.0
    alpha = 0.15
    last_time = time.time()

    print("[INFO] Starting main loop (press 'q' to quit)")
    try:
        while True:
            try:
                undistorted_full = und_q.get(timeout=1.0)
            except queue.Empty:
                time.sleep(0.005)
                continue

            # Step: create display-sized raw image
            display_raw = cv2.resize(undistorted_full, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            # depth estimation
            try:
                depth_colored = ud.estimate_depth(undistorted_full, profile=args.profile)
                depth_disp = cv2.resize(depth_colored, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"[WARN] Depth estimation error: {e}")
                depth_disp = np.full_like(display_raw, 128)

            combined = np.hstack((display_raw, depth_disp))

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_ema = alpha * inst_fps + (1 - alpha) * fps_ema
            fps_text = f"FPS: {fps_ema:.1f}"

            cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            cv2.imshow("Undistorted (L)  |  Depth (R)", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        und_worker.stop()
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Exiting.")


if __name__ == '__main__':
    main()
