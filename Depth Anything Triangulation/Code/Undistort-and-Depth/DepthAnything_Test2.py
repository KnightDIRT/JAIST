#40fps
"""
Optimized real-time Depth-Anything (small) pipeline.
Removed all undistortion logic — directly runs depth inference on live camera frames.

Key features:
 - GPU preprocessing with preallocated tensor
 - Exponential smoothing for temporal stability
 - Lightweight threading for real-time camera capture
 - Optional profiling
 - Reduced OpenCV thread contention
"""

import argparse
import threading
import queue
import time
import sys
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForDepthEstimation

cv2.setNumThreads(1)


class CameraStream:
    def __init__(self, src=0, backend=None):
        if backend is not None:
            self.cap = cv2.VideoCapture(src, backend)
        else:
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


class DepthEstimator:
    def __init__(self,
                 model_name="LiheYoung/depth-anything-small-hf",
                 device=None,
                 display_size=(640, 360),
                 inference_size=(384, 384)):

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[INFO] Using device: {self.device}")

        # load model
        print(f"[INFO] Loading model: {model_name}")
        self.processor = None
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)

        if self.device.type == "cuda":
            try:
                self.model = self.model.half().to(self.device)
            except Exception:
                self.model = self.model.to(self.device)
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"[WARN] torch.compile skipped: {e}")
        else:
            self.model = self.model.to(self.device)
        self.model.eval()

        # preallocations
        self.display_w, self.display_h = display_size
        self.inference_w, self.inference_h = inference_size

        if self.device.type == "cuda":
            self.input_tensor = torch.zeros((1, 3, self.inference_h, self.inference_w),
                                            device=self.device, dtype=torch.half)
        else:
            self.input_tensor = torch.zeros((1, 3, self.inference_h, self.inference_w),
                                            device=self.device, dtype=torch.float32)

        # temporal smoothing
        self.depth_smooth = None
        self.smooth_alpha = 0.5 #0 is max smoothing, 1 is no smoothing

        self._warmup()

    def _warmup(self, n=2):
        if self.device.type != "cuda":
            return
        print("[INFO] Warming up model...")
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
        print("[INFO] Warmup done.")

    def estimate_depth(self, frame, profile=False):
        t0 = time.time()
        H, W = frame.shape[:2]
        rgb_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               (self.inference_w, self.inference_h), interpolation=cv2.INTER_LINEAR)

        try:
            if self.device.type == "cuda":
                cpu_tensor = torch.from_numpy(rgb_small).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
                self.input_tensor.copy_(cpu_tensor.to(self.device, non_blocking=True).to(dtype=self.input_tensor.dtype))
            else:
                self.input_tensor.copy_(torch.from_numpy(rgb_small).permute(2, 0, 1).unsqueeze(0)
                                        .to(device=self.device, dtype=self.input_tensor.dtype).div_(255.0))
        except Exception as e:
            if self.processor is None:
                print(f"[WARN] Fast preprocessing failed ({e}), using HF processor.")
                self.processor = AutoProcessor.from_pretrained('LiheYoung/depth-anything-small-hf')
            inputs = self.processor(images=rgb_small, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                with torch.amp.autocast(self.device.type) if self.device.type == 'cuda' else torch.cpu.amp.autocast(enabled=False):
                    out = self.model(**inputs)
            depth_t = getattr(out, "predicted_depth", list(out.values())[0])
            depth = depth_t.squeeze().detach().cpu().float().numpy()
            return self._postprocess_depth(depth, W, H, profile, t0)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = self.model(pixel_values=self.input_tensor)
            else:
                out = self.model(pixel_values=self.input_tensor)

        depth_t = getattr(out, "predicted_depth", list(out.values())[0])
        depth = depth_t.squeeze().detach().cpu().float().numpy()
        return self._postprocess_depth(depth, W, H, profile, t0)

    def _postprocess_depth(self, depth, W, H, profile, t0):
        depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        if self.depth_smooth is None:
            self.depth_smooth = depth_resized
        else:
            self.depth_smooth = (1 - self.smooth_alpha) * self.depth_smooth + self.smooth_alpha * depth_resized

        depth_norm = cv2.normalize(self.depth_smooth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = np.clip(depth_norm, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)

        if profile:
            print(f"[PROFILE] Frame {(time.time() - t0)*1000:.1f}ms")

        return depth_color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=640)
    ap.add_argument("--display-height", type=int, default=360)
    ap.add_argument("--inference-width", type=int, default=384)
    ap.add_argument("--inference-height", type=int, default=384)
    ap.add_argument("--backend", type=str, default=None)
    ap.add_argument("--profile", action='store_true')
    args = ap.parse_args()

    backend = getattr(cv2, args.backend) if args.backend else None
    cam = CameraStream(src=args.camera_id, backend=backend)
    depth_estimator = DepthEstimator(
        display_size=(args.display_width, args.display_height),
        inference_size=(args.inference_width, args.inference_height)
    )

    fps_ema = 0.0
    alpha = 0.15
    last_time = time.time()

    print("[INFO] Starting main loop (press 'q' to quit)")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.01)
                continue

            depth_colored = depth_estimator.estimate_depth(frame, profile=args.profile)

            display_raw = cv2.resize(frame, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)
            depth_disp = cv2.resize(depth_colored, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            combined = np.hstack((display_raw, depth_disp))

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_ema = alpha * inst_fps + (1 - alpha) * fps_ema
            cv2.putText(combined, f"FPS: {fps_ema:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            cv2.imshow("Raw (L) | Depth (R)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Exiting.")


if __name__ == "__main__":
    main()
