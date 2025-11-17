"""
DepthAnything v3 — Realtime metric depth viewer (updated)
Each display pane is 640 x 360 (RGB | Depth | Colorbar).
Combined window size: 1920 x 360.

Model: depth-anything/DA3METRIC-LARGE
"""

import sys
import time
import threading
import argparse
import cv2
import numpy as np
import torch

# === EDIT if your local repo path differs ===
repo_path = r"C:/Users/Torenia/Depth-Anything-3/src"
sys.path.insert(0, repo_path)

from depth_anything_3.api import DepthAnything3

cv2.setNumThreads(1)

# ---------------------------
# Camera thread (non-blocking)
# ---------------------------
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

# ---------------------------
# Utility: colorbar image (vertical)
# ---------------------------
def create_colorbar_image(min_val, max_val, height=360, width=640, cmap=cv2.COLORMAP_MAGMA):
    # Create vertical gradient top->bottom
    gradient = np.linspace(1.0, 0.0, height)[:, None]
    indices = (gradient * 255).astype(np.uint8)
    strip = np.repeat(indices, width, axis=1)
    colorbar = cv2.applyColorMap(strip, cmap)  # BGR

    # Add tick labels (6 ticks)
    for i, label in enumerate(np.linspace(min_val, max_val, 6)):
        y = int((i / 5.0) * (height - 1))
        # flip vertical location
        y = height - 1 - y
        txt = f"{label:.1f}m" if (abs(max_val) > 1e-6 or abs(min_val) > 1e-6) else f"{label:.3f}"
        # choose font scale relative to height
        font_scale = max(0.45, height / 360.0 * 0.45)
        thickness = 1
        cv2.putText(colorbar, txt, (8, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return colorbar

# ---------------------------
# DepthAnything3 RT wrapper
# ---------------------------
class DA3MetricRT:
    def __init__(self,
                 model_name="depth-anything/DA3METRIC-LARGE",
                 device=None,
                 inference_size=(504, 378),
                 smooth_alpha=0.35):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[INFO] Device: {self.device}")

        print(f"[INFO] Loading model {model_name} ...")
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        self.inf_w, self.inf_h = inference_size
        self.smooth_alpha = smooth_alpha
        self.depth_smooth = None

        self._warmup()

    def _warmup(self):
        if self.device.type != "cuda":
            return
        print("[INFO] Warmup ...")
        dummy = np.zeros((self.inf_h, self.inf_w, 3), dtype=np.uint8)
        _ = self.model.inference([dummy])
        torch.cuda.synchronize()
        print("[INFO] Warmup done.")

    def estimate(self, frame_bgr):
        """Return (depth_color_bgr, raw_depth_meters_array, (cmin,cmax,is_metric))
        raw_depth_meters_array is HxW floats in meters if model provides metric depth,
        otherwise it's a relative depth array.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (self.inf_w, self.inf_h), interpolation=cv2.INTER_LINEAR)

        pred = self.model.inference([rgb_small])

        depth_obj = None
        if hasattr(pred, "metric_depth"):
            depth_obj = pred.metric_depth
        elif hasattr(pred, "depth_metric"):
            depth_obj = pred.depth_metric
        elif hasattr(pred, "depth"):
            depth_obj = pred.depth
        else:
            try:
                depth_obj = pred["metric_depth"]
            except Exception:
                depth_obj = None

        if isinstance(depth_obj, torch.Tensor):
            depth = depth_obj.detach().cpu().numpy()
        elif isinstance(depth_obj, np.ndarray):
            depth = depth_obj
        else:
            try:
                depth = np.array(depth_obj)
            except Exception:
                raise RuntimeError("Unable to extract depth array from model prediction.")

        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]

        H, W = frame_bgr.shape[:2]
        depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        if self.depth_smooth is None:
            self.depth_smooth = depth_resized
        else:
            self.depth_smooth = self.depth_smooth * (1 - self.smooth_alpha) + depth_resized * self.smooth_alpha

        mean_val = float(np.nanmean(self.depth_smooth))
        is_likely_metric = (0.02 < mean_val < 200.0)

        if is_likely_metric:
            lo = float(np.nanpercentile(self.depth_smooth, 2.0))
            hi = float(np.nanpercentile(self.depth_smooth, 98.0))
            if hi <= lo + 1e-3:
                hi = lo + 1.0
            if hi - lo < 0.5:
                hi = lo + 0.5
            depth_norm = np.clip((self.depth_smooth - lo) / (hi - lo), 0.0, 1.0)
            colorbar_min, colorbar_max = lo, hi
        else:
            lo, hi = float(np.nanmin(self.depth_smooth)), float(np.nanmax(self.depth_smooth))
            if hi <= lo + 1e-6:
                hi = lo + 1.0
            depth_norm = (self.depth_smooth - lo) / (hi - lo)
            colorbar_min, colorbar_max = lo, hi

        depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

        return depth_color, self.depth_smooth.copy(), (colorbar_min, colorbar_max, is_likely_metric)


# ---------------------------
# Mouse/cursor handling
# ---------------------------
cursor_pos = (0, 0)
def on_mouse(event, x, y, flags, param):
    global cursor_pos
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_pos = (x, y)


# ---------------------------
# Main app
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=0)
    # default display pane size: 640 x 360 each
    ap.add_argument("--display-width", type=int, default=640)
    ap.add_argument("--display-height", type=int, default=360)
    ap.add_argument("--inference-width", type=int, default=504)
    ap.add_argument("--inference-height", type=int, default=378)
    ap.add_argument("--model", type=str, default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    backend = None

    cam = CameraStream(src=args.camera_id, backend=backend)
    da3 = DA3MetricRT(model_name=args.model,
                      device=args.device,
                      inference_size=(args.inference_width, args.inference_height),
                      smooth_alpha=0.35)

    window_name = "DA3 Metric — RGB | Depth | Scale (each 640x360)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.display_width * 3, args.display_height)
    cv2.setMouseCallback(window_name, on_mouse)

    fps_ema = 0.0
    alpha = 0.12
    last_time = time.time()

    print("[INFO] Press 'q' to quit.")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.01)
                continue

            # make sure original frame is reasonably sized; we will resize panes to display size
            frame_resized_for_display = cv2.resize(frame, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            # estimate depth on full original frame, but visualization will be resized
            depth_color_full, depth_raw, (cmin, cmax, is_metric) = da3.estimate(frame)

            depth_disp = cv2.resize(depth_color_full, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)
            rgb_disp = frame_resized_for_display

            # Create colorbar pane of exact pane size
            colorbar = create_colorbar_image(cmin, cmax, height=args.display_height, width=args.display_width, cmap=cv2.COLORMAP_MAGMA)

            # Compose: RGB | Depth | Colorbar (each 640x360)
            combined = np.hstack((rgb_disp, depth_disp, colorbar))

            # compute FPS
            now = time.time()
            fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_ema = fps_ema * (1 - alpha) + fps * alpha

            # Put FPS text
            cv2.putText(combined, f"FPS: {fps_ema:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

            # Cursor handling
            x, y = cursor_pos
            pane_w = args.display_width
            pane_h = args.display_height

            depth_value_text = None
            cross_x, cross_y = None, None

            if 0 <= y < pane_h:
                # RGB pane
                if 0 <= x < pane_w:
                    # Map to original frame coords (depth_raw is same size as original frame)
                    H0, W0 = depth_raw.shape[:2]
                    rx = int((x / pane_w) * W0)
                    ry = int((y / pane_h) * H0)
                    rx = np.clip(rx, 0, W0 - 1)
                    ry = np.clip(ry, 0, H0 - 1)
                    val = float(depth_raw[ry, rx])
                    depth_value_text = f"{val:.2f} m" if is_metric else f"{val:.3f} (rel)"
                    cross_x = int((x / pane_w) * pane_w) + pane_w  # crosshair placed on depth pane (second pane)
                    cross_y = y
                # Depth pane
                elif pane_w <= x < 2 * pane_w:
                    rel_x = x - pane_w
                    H0, W0 = depth_raw.shape[:2]
                    rx = int((rel_x / pane_w) * W0)
                    ry = int((y / pane_h) * H0)
                    rx = np.clip(rx, 0, W0 - 1)
                    ry = np.clip(ry, 0, H0 - 1)
                    val = float(depth_raw[ry, rx])
                    depth_value_text = f"{val:.2f} m" if is_metric else f"{val:.3f} (rel)"
                    cross_x = x
                    cross_y = y

            # Draw depth text and crosshair
            if depth_value_text is not None:
                cv2.putText(combined, f"Depth (cursor): {depth_value_text}", (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2, cv2.LINE_AA)

            if cross_x is not None and cross_y is not None:
                cv2.circle(combined, (cross_x, cross_y), 8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.line(combined, (cross_x-12, cross_y), (cross_x+12, cross_y), (0,0,255), 1, cv2.LINE_AA)
                cv2.line(combined, (cross_x, cross_y-12), (cross_x, cross_y+12), (0,0,255), 1, cv2.LINE_AA)

            metric_label = "Metric depth (meters)" if is_metric else "Relative depth"
            cv2.putText(combined, metric_label, (12, args.display_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)

            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Exiting.")

if __name__ == "__main__":
    main()
