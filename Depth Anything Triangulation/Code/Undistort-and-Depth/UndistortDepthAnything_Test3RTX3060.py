#16fps
"""
Real-time fisheye undistort + Depth-Anything (small) optimized for RTX3060 (6GB).
Outputs: left = undistorted raw, right = depth view (undistorted before inference).
Key points:
 - CPU undistort via OpenCV cv2.remap (uses calibration JSON)
 - FP16 inference with torch.amp.autocast('cuda')
 - Threaded camera capture
 - Small inference size (default 320x320) for RTX3060 VRAM safety
"""

import argparse
import json
import threading
import queue
import time
import sys

import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForDepthEstimation
import torch

# -------------------------
# Camera capture thread
# -------------------------
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")
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

# -------------------------
# Depth + Undistort class
# -------------------------
class UndistortDepth:
    def __init__(self,
                 calib_file,
                 model_name="LiheYoung/depth-anything-small-hf",
                 device=None,
                 display_size=(640, 360),
                 inference_size=(320, 320),
                 remap_interp=cv2.INTER_LINEAR):
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
            # expected keys: "K", "D", "image_size" (image_size = [w, h])
            self.K = np.array(calib["K"], dtype=np.float64)
            self.D = np.array(calib["D"], dtype=np.float64)
            self.calib_image_size = tuple(calib["image_size"])  # (w, h)
        except Exception as e:
            raise RuntimeError(f"Failed to load calibration file: {e}")

        # precompute undistort maps (use float32 maps)
        new_K = self.K.copy()
        w_calib, h_calib = self.calib_image_size
        # CV_32F maps are convenient and accurate
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, (w_calib, h_calib), cv2.CV_32FC1
        )
        self.remap_interp = remap_interp

        # display & inference sizes
        self.display_w, self.display_h = display_size
        self.inference_w, self.inference_h = inference_size

        # load model + processor
        print(f"[INFO] Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)

        # move to device & half precision if GPU
        if self.device.type == "cuda":
            self.model = self.model.half().to(self.device)
            # try compile for lower overhead (best-effort)
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"[WARN] torch.compile skipped/failed: {e}")
        else:
            self.model = self.model.to(self.device)

        self.model.eval()

        # warmup few iterations to avoid first-frame stalls
        self._warmup()

    def _warmup(self, n=2):
        if self.device.type != "cuda":
            return
        print("[INFO] Warmup model (2 iterations)...")
        dummy = torch.zeros((1, 3, self.inference_h, self.inference_w),
                            device=self.device, dtype=torch.half)
        with torch.no_grad():
            for _ in range(n):
                with torch.amp.autocast("cuda"):
                    try:
                        out = self.model(pixel_values=dummy)
                    except Exception:
                        # some models accept alternative args; try positional
                        out = self.model(dummy)
        torch.cuda.synchronize()
        print("[INFO] Warmup complete.")

    def undistort(self, frame):
        """
        frame: BGR numpy array from camera (native capture size)
        Steps:
          - resize to calib image size
          - remap with cv2.remap
        returns undistorted BGR image (uint8)
        """
        # resize to calib size (cv2.resize expects (w,h) tuple)
        frame_rs = cv2.resize(frame, (self.calib_image_size[0], self.calib_image_size[1]),
                              interpolation=cv2.INTER_LINEAR)
        und = cv2.remap(frame_rs, self.map1, self.map2, interpolation=self.remap_interp)
        return und

    def estimate_depth(self, undistorted_bgr):
        """
        undistorted_bgr: undistorted BGR image (numpy uint8)
        returns depth_colored (BGR uint8) aligned with undistorted_bgr
        """
        # prepare small image for model inference
        # convert to RGB (processor expects RGB)
        rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
        # resize to inference size (model input)
        rgb_small = cv2.resize(rgb, (self.inference_w, self.inference_h), interpolation=cv2.INTER_LINEAR)

        # process -> returns CPU tensors
        inputs = self.processor(images=rgb_small, return_tensors="pt").to("cpu")

        # if GPU, pin memory then async transfer to half precision
        if self.device.type == "cuda":
            inputs = {k: v.pin_memory() for k, v in inputs.items()}
            inputs_gpu = {k: v.to(self.device, non_blocking=True, dtype=torch.half) for k, v in inputs.items()}
        else:
            inputs_gpu = {k: v.to(self.device) for k, v in inputs.items()}

        # inference
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = self.model(**inputs_gpu)
            else:
                out = self.model(**inputs_gpu)

        # extract predicted_depth tensor and convert to float32 numpy (compatible with OpenCV)
        if hasattr(out, "predicted_depth"):
            depth_t = out.predicted_depth
        else:
            # fallback: try first value
            depth_t = list(out.values())[0] if isinstance(out, dict) else out

        depth = depth_t.squeeze().detach().cpu().float().numpy()  # ensure float32

        # resize depth back to undistorted size (width,height)
        H, W = undistorted_bgr.shape[:2]
        # cv2.resize expects dsize=(w,h)
        depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # normalize and colorize for visualization
        depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = np.clip(depth_norm, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)  # BGR

        return depth_color

# -------------------------
# Main: run loop
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-file", default="Undistort-and-Depth/fisheye_calibration_data.json", help="Calibration JSON with K, D, image_size (w,h)")
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=640)
    ap.add_argument("--display-height", type=int, default=360)
    ap.add_argument("--inference-width", type=int, default=320)
    ap.add_argument("--inference-height", type=int, default=320)
    args = ap.parse_args()

    # create camera thread
    try:
        cam = CameraStream(src=args.camera_id)
    except Exception as e:
        print(f"[ERROR] Camera init failed: {e}")
        sys.exit(1)

    # create undistort+depth estimator
    ud = UndistortDepth(
        calib_file=args.calib_file,
        display_size=(args.display_width, args.display_height),
        inference_size=(args.inference_width, args.inference_height),
    )

    # smoothing FPS variables
    fps_ema = 0.0
    alpha = 0.15  # ema smoothing factor (small -> smoother)
    last_time = time.time()

    print("[INFO] Starting main loop (press 'q' to quit)")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.005)
                continue

            # Step 1: undistort (applied to both raw display and depth input)
            undistorted_full = ud.undistort(frame)  # shape (h_calib, w_calib, 3), uint8 BGR

            # Step 2: create display-sized raw image
            display_raw = cv2.resize(undistorted_full, (args.display_width, args.display_height),
                                     interpolation=cv2.INTER_AREA)

            # Step 3: depth estimation (using undistorted image)
            try:
                depth_colored = ud.estimate_depth(undistorted_full)  # same size as undistorted_full
                # resize depth_colored to display size for side-by-side
                depth_disp = cv2.resize(depth_colored, (args.display_width, args.display_height),
                                        interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"[WARN] Depth estimation error: {e}")
                # fallback: solid gray
                depth_disp = np.full_like(display_raw, 128)

            # Step 4: compose side-by-side (left: raw, right: depth)
            combined = np.hstack((display_raw, depth_disp))

            # FPS measurement (instant + EMA smoothing)
            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_ema = alpha * inst_fps + (1 - alpha) * fps_ema
            fps_text = f"FPS: {fps_ema:.1f}"

            cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            cv2.imshow("Undistorted (L)  |  Depth (R)", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Exiting.")

if __name__ == "__main__":
    main()
