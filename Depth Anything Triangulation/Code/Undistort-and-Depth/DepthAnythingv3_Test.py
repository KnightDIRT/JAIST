import argparse
import threading
import time
import cv2
import numpy as np
import torch
import sys

repo_path = r"C:/Users/Torenia/Depth-Anything-3/src"
sys.path.insert(0, repo_path)

from depth_anything_3.api import DepthAnything3

cv2.setNumThreads(1)


# ============================
#   CAMERA THREAD
# ============================
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
                time.sleep(0.005)
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


# ============================
#   DEPTH ANYTHING 3 WRAPPER
# ============================
class DepthAnything3RT:
    def __init__(self,
                 model_name="depth-anything/DA3-SMALL",
                 device=None,
                 inference_size=(504, 378),
                 display_size=(640, 360)):

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] Using device: {self.device}")

        # load model
        print(f"[INFO] Loading DA3 model: {model_name}")
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(self.device)

        self.model = self.model.to(self.device)
        torch.set_float32_matmul_precision("high")

        self.model.eval()

        # dimensions
        self.inf_w, self.inf_h = inference_size
        self.disp_w, self.disp_h = display_size

        # temporal smoothing
        self.depth_smooth = None
        self.smooth_alpha = 0.35

        # warmup
        self._warmup()

    # =======================
    #   Warmup
    # =======================
    def _warmup(self):
        if self.device.type != "cuda":
            return

        print("[INFO] Warming up...")
        dummy = np.zeros((self.inf_h, self.inf_w, 3), dtype=np.uint8)
        _ = self.model.inference([dummy])
        torch.cuda.synchronize()
        print("[INFO] Warmup complete.")

    # =======================
    #   Depth inference
    # =======================
    def estimate(self, frame):
        # resize for inference
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (self.inf_w, self.inf_h), interpolation=cv2.INTER_LINEAR)

        # model expects list of images (numpy HWC uint8)
        prediction = self.model.inference([rgb_small])

        depth = prediction.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()

        depth = depth[0]  # [H, W]

        return self._postprocess(depth, frame.shape)

    # =======================
    #   Postprocess (color)
    # =======================
    def _postprocess(self, depth, original_shape):
        H, W = original_shape[:2]

        # resize back to frame size
        depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # smoothing
        if self.depth_smooth is None:
            self.depth_smooth = depth_resized
        else:
            self.depth_smooth = self.depth_smooth * (1 - self.smooth_alpha) + depth_resized * self.smooth_alpha

        # normalize
        depth_norm = cv2.normalize(self.depth_smooth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)

        # COLOR: MATCH YOUR DepthAnything_Test2.py (PLASMA)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)

        return depth_color


# ============================
#   MAIN LOOP
# ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=640)
    ap.add_argument("--display-height", type=int, default=360)
    ap.add_argument("--inference-width", type=int, default=504)
    ap.add_argument("--inference-height", type=int, default=378)
    ap.add_argument("--backend", type=str, default=None)
    args = ap.parse_args()

    backend = getattr(cv2, args.backend) if args.backend else None

    cam = CameraStream(src=args.camera_id, backend=backend)

    da3 = DepthAnything3RT(
        inference_size=(args.inference_width, args.inference_height),
        display_size=(args.display_width, args.display_height)
    )

    fps_ema = 0.0
    alpha = 0.12
    last = time.time()

    print("[INFO] Starting realtime loop (press 'q' to quit)")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.01)
                continue

            depth_color = da3.estimate(frame)

            # Resize for display
            raw_disp = cv2.resize(frame, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)
            depth_disp = cv2.resize(depth_color, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            # side-by-side
            combined = np.hstack((raw_disp, depth_disp))

            # FPS
            now = time.time()
            fps = 1.0 / max(1e-6, now - last)
            last = now
            fps_ema = fps_ema * (1 - alpha) + fps * alpha

            cv2.putText(combined, f"FPS: {fps_ema:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Raw (L) | Depth (R)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Exiting.")


if __name__ == "__main__":
    main()
