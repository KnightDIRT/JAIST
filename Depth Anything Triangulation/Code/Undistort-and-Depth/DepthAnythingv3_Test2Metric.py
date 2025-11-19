"""
DepthAnything ShipMetric (v3 integration)

This is a drop-in rewrite of DepthAnything_ShipMetric.py with the Depth-Anything v3
inference wrapper replacing the original v1 model code. The rest of the pipeline
(LGNet fill, metric calibration, Open3D visualization, etc.) was preserved with
minimal changes so it behaves the same as before.

Key changes:
- Replaced DepthEstimator (HF v1) with DepthAnything3Estimator (v3 wrapper)
  that uses depth_anything_3.api.DepthAnything3.from_pretrained(...).
- Kept same public API: estimate_depth(frame_bgr, profile=False) ->
  (depth_color_bgr uint8, depth_rel float32)
- Added warmup and device handling similar to your v3 test script.
- Fixed a small bug in CameraStream.stop (typo: Trueq -> True).

Note: Update `repo_path_v3` to point to your local depth-anything-3 repo if needed.
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

import open3d as o3d

# LGNet generative fill integration (unchanged)
import torchvision.transforms.functional as F
from PIL import Image

# add your Depth-Anything v3 repo to path (adjust if necessary)
repo_path_v3 = r"C:/Users/Torenia/Depth-Anything-3/src"
if repo_path_v3 not in sys.path:
    sys.path.insert(0, repo_path_v3)

from depth_anything_3.api import DepthAnything3

# LGNet code (unchanged from original file)
repo_path_lg = r"C:/Users/Torenia/LGNet"
if repo_path_lg not in sys.path:
    sys.path.insert(0, repo_path_lg)

from options.test_options import TestOptions
from models import create_model


def lggnet_postprocess(img_tensor):
    img = (img_tensor + 1) / 2 * 255
    img = img.permute(0, 2, 3, 1).int().cpu().numpy().astype(np.uint8)
    return img[0]


class LGNetFiller:
    def __init__(self):
        sys_argv_backup = sys.argv.copy()
        sys.argv += [
            '--dataroot', './dummy',
            '--name', 'celebahq_LGNet',
            '--model', 'pix2pixglg',
            '--netG1', 'unet_256',
            '--netG2', 'resnet_4blocks',
            '--netG3', 'unet256',
            '--input_nc', '4',
            '--direction', 'AtoB',
            '--no_dropout',
            '--gpu_ids', '0'
        ]
        opt = TestOptions().parse()
        opt.num_threads = 0
        opt.batch_size = 1
        opt.no_flip = True
        opt.serial_batches = True
        opt.display_id = -1

        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        sys.argv = sys_argv_backup

    def fill(self, frame_bgr, mask_bool):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb).resize((256, 256))
        mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255)).resize((256, 256))

        frame_tensor = (F.to_tensor(frame_pil) * 2 - 1).unsqueeze(0)
        mask_tensor = F.to_tensor(mask_img.convert("L")).unsqueeze(0)

        data = {'A': frame_tensor, 'B': mask_tensor, 'A_paths': ''}
        self.model.set_input(data)
        with torch.no_grad():
            self.model.forward()

        comp_img = lggnet_postprocess(self.model.merged_images3)
        comp_bgr = cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR)
        comp_bgr = cv2.resize(comp_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return comp_bgr


# =====================
# Metric settings (unchanged)
# =====================
BAR_START_PX = (320, 0)
BAR_END_PX = (323, 200)
BAR_START_DISTANCE_M = 0.0
BAR_END_DISTANCE_M = 0.222
BAR_PIXEL_HALF_WIDTH = 15
CAL_ALPHA = 0.08
DEPTH_MIN_M = 0.054
DEPTH_MAX_M = 1
SCALE_BAR_WIDTH = 40
SCALE_BAR_RANGE_M = (0.0, 1.0)


# Camera / helper classes (kept / small fix)
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


# =====================
# Depth-Anything v3 wrapper
# =====================
class DepthAnything3Estimator:
    def __init__(self,
                 model_name="depth-anything/DA3-LARGE",
                 device=None,
                 display_size=(640, 360),
                 inference_size=(504, 378)):

        # device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[INFO] Using device: {self.device}")

        print(f"[INFO] Loading DA3 model: {model_name}")
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        torch.set_float32_matmul_precision("high")
        self.model.eval()

        # sizes
        self.inf_w, self.inf_h = inference_size
        self.disp_w, self.disp_h = display_size

        # smoothing
        self.depth_smooth = None
        self.smooth_alpha = 0.35

        # warmup
        self._warmup()

    def _warmup(self):
        if self.device.type != "cuda":
            return
        print("[INFO] Warming up DA3...")
        dummy = np.zeros((self.inf_h, self.inf_w, 3), dtype=np.uint8)
        try:
            _ = self.model.inference([dummy])
            torch.cuda.synchronize()
            print("[INFO] Warmup complete.")
        except Exception as e:
            print(f"[WARN] Warmup failed: {e}")

    def estimate_depth(self, frame_bgr, profile=False):
        """Return (depth_color_bgr uint8, depth_rel float32)
        depth_rel is the relative depth map (smoothed but not metric).
        """
        t0 = time.time()
        H, W = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (self.inf_w, self.inf_h), interpolation=cv2.INTER_LINEAR)

        # DA3 expects a list of numpy HWC uint8 images
        try:
            prediction = self.model.inference([rgb_small])
        except Exception as e:
            raise RuntimeError(f"DA3 inference failed: {e}")

        depth = prediction.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        depth = depth[0]

        # resize back
        depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # smoothing
        if self.depth_smooth is None:
            self.depth_smooth = depth_resized
        else:
            self.depth_smooth = (1 - self.smooth_alpha) * self.depth_smooth + self.smooth_alpha * depth_resized

        depth_norm = cv2.normalize(self.depth_smooth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)

        if profile:
            t1 = time.time()
            print(f"[PROFILE] DA3 total {(t1-t0)*1000:.1f}ms")

        return depth_color, self.depth_smooth.astype(np.float32)


# The rest of the file is the main application logic from your original ShipMetric file
# with the new DepthAnything3Estimator used in place of the HF-based DepthEstimator.

# (Below: copied and adapted main loop but using DepthAnything3Estimator)

# Metric helpers (same as original)

def make_bar_mask(shape, p0, p1, half_width):
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    p0i = (int(round(p0[0])), int(round(p0[1])))
    p1i = (int(round(p1[0])), int(round(p1[1])))
    cv2.line(mask, p0i, p1i, color=255, thickness=max(1, int(half_width * 2)))
    return mask.astype(bool)


def compute_median_in_mask(depth_rel, mask):
    vals = depth_rel[mask]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def solve_nonlinear_bar_mapping(v1, D1, v2, D2, offset_m):
    if v1 is None or v2 is None or abs(v1 - v2) < 1e-6:
        return None, None
    z1 = math.sqrt(max(D1**2 - offset_m**2, 0.0))
    z2 = math.sqrt(max(D2**2 - offset_m**2, 0.0))
    A = np.array([[v1, 1.0], [v2, 1.0]], dtype=np.float64)
    b = np.array([z1, z2], dtype=np.float64)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(sol[0]), float(sol[1])


def draw_metric_scale_bar(img, current_distance_m, range_m=(0.0, 5.0), width=SCALE_BAR_WIDTH):
    h, w = img.shape[:2]
    bar_h = int(h * 0.85)
    pad = 8
    x0 = w - width - pad
    y0 = int((h - bar_h) / 2)
    x1 = w - pad
    y1 = y0 + bar_h
    cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 40), -1)
    min_m, max_m = range_m
    num_ticks = 6
    for i in range(num_ticks + 1):
        fy = i / num_ticks
        yy = int(y1 - fy * bar_h)
        cv2.line(img, (x0, yy), (x0 + 8, yy), (200, 200, 200), 1)
        val = min_m + fy * (max_m - min_m)
        txt = f"{val:.1f}m"
        cv2.putText(img, txt, (x0 - 45, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
    cur = np.clip(current_distance_m, min_m, max_m)
    frac = (cur - min_m) / (max_m - min_m) if (max_m - min_m) > 1e-6 else 0.0
    yy_cur = int(y1 - frac * bar_h)
    cv2.circle(img, (x0 + 12, yy_cur), 6, (0, 255, 255), -1)
    cv2.putText(img, f"{current_distance_m:.2f}m", (x0 - 80, yy_cur + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img


# Frame worker
class FrameWorker:
    def __init__(self, cam: CameraStream, out_queue: queue.Queue):
        self.cam = cam
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
            try:
                while not self.out_q.empty():
                    try:
                        self.out_q.get_nowait()
                    except Exception:
                        break
                self.out_q.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)


# Main function (adapted to use DepthAnything3Estimator)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=640)
    ap.add_argument("--display-height", type=int, default=360)
    ap.add_argument("--inference-width", type=int, default=504)
    ap.add_argument("--inference-height", type=int, default=378)
    ap.add_argument("--backend", type=str, default=None, help='cv2 backend e.g. cv2.CAP_DSHOW or cv2.CAP_V4L2')
    ap.add_argument("--profile", action='store_true')
    ap.add_argument("--mask-before-image", type=str, default="./Undistort-and-Depth/Mask3.png",
                help="Path to an image mask file. Colored (non-black) areas are ignored for depth estimation.")
    ap.add_argument("--mask-image", type=str, default="./Undistort-and-Depth/MaskA2.png",
                help="Path to an image mask file. Colored (non-black) areas are ignored for proximity distance.")
    ap.add_argument("--debug-mask", action='store_true',
                help="Toggle display of the mask overlay (press 'm' to toggle).")
    ap.add_argument("--point-skip", type=int, default=4, help="pixel stride for point cloud subsampling (bigger -> fewer points, faster)")
    ap.add_argument("--pc-window-width", type=int, default=960)
    ap.add_argument("--pc-window-height", type=int, default=720)
    args = ap.parse_args()

    try:
        backend = getattr(cv2, args.backend) if args.backend else None
        cam = CameraStream(src=args.camera_id, backend=backend)
    except Exception as e:
        print(f"[ERROR] Camera init failed: {e}")
        sys.exit(1)

    # instantiate v3 depth estimator
    depth_est = DepthAnything3Estimator(
        model_name="depth-anything/DA3-LARGE",
        display_size=(args.display_width, args.display_height),
        inference_size=(args.inference_width, args.inference_height),
    )

    cap_w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask_before_img = mask_img = None
    mask_before_bool = mask_bool = None
    mask_overlay = True

    def _make_bool_mask_from_img(img_bgr, target_w, target_h, invert=False):
        if img_bgr is None:
            return None
        nonblack = np.any(img_bgr > 20, axis=2)
        if invert:
            bool_mask = ~nonblack
        else:
            bool_mask = nonblack
        if bool_mask.shape[1] != target_w or bool_mask.shape[0] != target_h:
            bool_mask = cv2.resize(bool_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return bool_mask

    if args.mask_before_image and os.path.exists(args.mask_before_image):
        tmp = cv2.imread(args.mask_before_image, cv2.IMREAD_COLOR)
        if tmp is not None:
            mask_before_img = tmp
            mask_before_bool = _make_bool_mask_from_img(mask_before_img, cap_w, cap_h, invert=True)
            print(f"[INFO] Loaded mask-before image: colored = valid, black = masked")
        else:
            print(f"[WARN] Failed to read mask-before image at {args.mask_before_image}")

    if args.mask_image and os.path.exists(args.mask_image):
        tmp = cv2.imread(args.mask_image, cv2.IMREAD_COLOR)
        if tmp is not None:
            mask_img = tmp
            mask_bool = _make_bool_mask_from_img(mask_img, cap_w, cap_h, invert=True)
            print(f"[INFO] Loaded mask-after image: colored = masked/ignored")
        else:
            print(f"[WARN] Failed to read mask image at {args.mask_image}")

    frame_q = queue.Queue(maxsize=1)
    frame_worker = FrameWorker(cam, frame_q)

    fps_ema = 0.0
    alpha = 0.15
    last_time = time.time()

    a_ema = None
    b_ema = None

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Real-Time Metric Point Cloud", width=args.pc_window_width, height=args.pc_window_height)
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic, copy=True)
    R = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]], dtype=float)
    extrinsic[:3, :3] = extrinsic[:3, :3] @ R
    cam_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    cylinder_height = 0.24
    cylinder_heightUsable = 0.08
    cylinder_radius = 0.045
    cylinder_center = [0.0, 0.0, -0.12]
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=cylinder_height, resolution=40, split=1)
    cylinder.translate(cylinder_center)
    cylinder_wire = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder)
    cylinder_wire.paint_uniform_color([0.0, 1.0, 0.0])
    vis.add_geometry(cylinder_wire)

    pcd = o3d.geometry.PointCloud()
    init_points = np.zeros((1, 3), dtype=np.float32)
    init_colors = np.zeros((1, 3), dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(init_points)
    pcd.colors = o3d.utility.Vector3dVector(init_colors)
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([0.0, 0.0, 0.0])
    ctr = vis.get_view_control()

    uv_grid_ready = False
    u_full = v_full = None

    print("[INFO] Starting main loop (press 'q' in the OpenCV window to quit)")

    try:
        mouse_x, mouse_y = args.display_width // 2, args.display_height // 2
        def mouse_callback(event, x, y, flags, param):
            nonlocal mouse_x, mouse_y
            if event == cv2.EVENT_MOUSEMOVE:
                if x < args.display_width:
                    mouse_x, mouse_y = x, y
                else:
                    mouse_x, mouse_y = x - args.display_width, y

        cv2.namedWindow("LGNet Filled (L)  |  Metric Depth (R)")
        cv2.setMouseCallback("LGNet Filled (L)  |  Metric Depth (R)", mouse_callback)

        while True:
            try:
                frame_full = frame_q.get(timeout=1.0)
            except queue.Empty:
                time.sleep(0.005)
                continue

            try:
                if mask_before_bool is not None:
                    if not hasattr(depth_est, "lggnet"):
                        print("[INFO] Initializing LGNet generative fill...")
                        depth_est.lggnet = LGNetFiller()
                    mask_before_full = cv2.resize(mask_before_bool.astype(np.uint8), (frame_full.shape[1], frame_full.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    masked_frame = depth_est.lggnet.fill(frame_full, mask_before_full)
                else:
                    masked_frame = frame_full
            except Exception as e:
                print(f"[WARN] LGNet fill failed: {e} — falling back to raw frame for masked_frame.")
                masked_frame = frame_full

            display_masked = cv2.resize(masked_frame, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            try:
                depth_colored_full, depth_rel_full = depth_est.estimate_depth(masked_frame, profile=args.profile)
                depth_disp = cv2.resize(depth_colored_full, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"[WARN] Depth estimation error: {e}")
                depth_disp = np.full_like(display_masked, 128)
                depth_rel_full = np.full((frame_full.shape[0], frame_full.shape[1]), 0.5, dtype=np.float32)

            H_full, W_full = depth_rel_full.shape[:2]

            if not uv_grid_ready or (u_full.shape[1] != W_full or v_full.shape[0] != H_full):
                us = np.arange(W_full)
                vs = np.arange(H_full)
                u_full, v_full = np.meshgrid(us, vs)
                uv_grid_ready = True

            bar_mask = make_bar_mask((H_full, W_full), BAR_START_PX, BAR_END_PX, BAR_PIXEL_HALF_WIDTH)

            def sample_median_at_point(depth_map, pt, box_half=8):
                x, y = int(round(pt[0])), int(round(pt[1]))
                x0 = max(0, x - box_half)
                x1 = min(depth_map.shape[1], x + box_half)
                y0 = max(0, y - box_half)
                y1 = min(depth_map.shape[0], y + box_half)
                vals = depth_map[y0:y1, x0:x1].ravel()
                if vals.size == 0:
                    return None
                return float(np.median(vals))

            d_start_rel = sample_median_at_point(depth_rel_full, BAR_START_PX, box_half=max(4, BAR_PIXEL_HALF_WIDTH))
            d_end_rel   = sample_median_at_point(depth_rel_full, BAR_END_PX, box_half=max(4, BAR_PIXEL_HALF_WIDTH))

            if d_start_rel is None:
                d_start_rel = compute_median_in_mask(depth_rel_full, bar_mask)
            if d_end_rel is None:
                d_end_rel = compute_median_in_mask(depth_rel_full, bar_mask)

            CAM_OFFSET_M = 0.06
            a_new, b_new = solve_nonlinear_bar_mapping(d_start_rel, BAR_START_DISTANCE_M, d_end_rel, BAR_END_DISTANCE_M, CAM_OFFSET_M)

            if a_new is None or b_new is None:
                if a_ema is None:
                    a_new, b_new = 1.0, 0.0
                else:
                    a_new, b_new = a_ema, b_ema

            a_ref, b_ref = a_new, b_new

            if a_ema is None:
                a_ema, b_ema = a_ref, b_ref
            else:
                a_ema = (1 - CAL_ALPHA) * a_ema + CAL_ALPHA * a_ref
                b_ema = (1 - CAL_ALPHA) * b_ema + CAL_ALPHA * b_ref

            z_est = a_ema * depth_rel_full + b_ema
            depth_m_full = np.sqrt(np.maximum(z_est**2 + CAM_OFFSET_M**2, 0.0))
            depth_m_full = np.clip(depth_m_full, DEPTH_MIN_M, DEPTH_MAX_M)

            depth_m_disp = cv2.resize(depth_m_full, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            vis_img = display_masked.copy()

            sx = args.display_width / float(W_full)
            sy = args.display_height / float(H_full)
            def scale_pt(pt):
                return (int(round(pt[0] * sx)), int(round(pt[1] * sy)))

            bar_start_disp = scale_pt(BAR_START_PX)
            bar_end_disp   = scale_pt(BAR_END_PX)
            bar_half_width_disp = max(1, int(round(BAR_PIXEL_HALF_WIDTH * (sx + sy) * 0.5)))

            cv2.line(vis_img, bar_start_disp, bar_end_disp, (255, 200, 0), 2)
            cv2.line(vis_img, bar_start_disp, bar_end_disp, (255, 200, 0), thickness=max(3, bar_half_width_disp*2))

            cx_disp, cy_disp = int(np.clip(mouse_x, 0, args.display_width - 1)), int(np.clip(mouse_y, 0, args.display_height - 1))
            cursor_depth_m = float(depth_m_disp[cy_disp, cx_disp])

            cv2.circle(vis_img, (cx_disp, cy_disp), 6, (0,255,255), -1)
            cv2.putText(vis_img, f"{cursor_depth_m * 100:.2f} cm", (cx_disp + 10, cy_disp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.putText(vis_img, f"a={a_ema:.4f} b={b_ema:.3f}", (10, args.display_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            depth_vis = cv2.normalize(depth_m_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_vis_with_scale = draw_metric_scale_bar(depth_vis, cursor_depth_m, range_m=SCALE_BAR_RANGE_M, width=SCALE_BAR_WIDTH)

            if mask_overlay:
                color_mask = np.zeros_like(vis_img)
                if mask_before_bool is not None:
                    mask_before_disp = cv2.resize(mask_before_bool.astype(np.uint8), (args.display_width, args.display_height), interpolation=cv2.INTER_NEAREST).astype(bool)
                    color_mask[mask_before_disp] = (0, 255, 0)
                if mask_bool is not None:
                    mask_after_disp = cv2.resize(mask_bool.astype(np.uint8), (args.display_width, args.display_height), interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlap_disp = mask_after_disp & (np.any(color_mask > 0, axis=2))
                    color_mask[mask_after_disp] = (0, 0, 255)
                    color_mask[overlap_disp] = (0, 255, 255)
                vis_img = cv2.addWeighted(vis_img, 1.0, color_mask, 0.45, 0)

            combined = np.hstack((vis_img, depth_vis_with_scale))

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_ema = alpha * inst_fps + (1 - alpha) * fps_ema
            fps_text = f"FPS: {fps_ema:.1f}"

            cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            cv2.imshow("LGNet Filled (L)  |  Metric Depth (R)", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mask_overlay = not mask_overlay
                print(f"[DEBUG] Mask overlay {'ON' if mask_overlay else 'OFF'}")

            # Open3D update (same as original)
            try:
                W_cam = float(W_full)
                H_cam = float(H_full)
                fov_x_deg = 95.0
                fov_y_deg = 2 * math.degrees(math.atan((H_cam / W_cam) * math.tan(math.radians(fov_x_deg / 2.0))))
                fx = (W_cam / 2.0) / math.tan(math.radians(fov_x_deg / 2.0))
                fy = (H_cam / 2.0) / math.tan(math.radians(fov_y_deg / 2.0))
                cx = W_cam / 2.0
                cy = H_cam / 2.0

                depth_o3d = o3d.geometry.Image((np.clip(depth_m_full, 0.0, DEPTH_MAX_M) * 1000.0).astype(np.uint16))
                color_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
                color_o3d = o3d.geometry.Image(color_rgb)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=float(DEPTH_MAX_M), convert_rgb_to_intensity=False)

                intrinsic = o3d.camera.PinholeCameraIntrinsic(width=int(W_cam), height=int(H_cam), fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))

                new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

                new_pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

                pts_np = np.asarray(new_pcd.points)
                col_np = np.asarray(new_pcd.colors)

                if mask_bool is not None:
                    mask_full = cv2.resize(mask_bool.astype(np.uint8), (W_full, H_full), interpolation=cv2.INTER_NEAREST).astype(bool)
                    mask_flat = mask_full.flatten()
                    valid_idx = np.where(mask_flat == 0)[0]
                    if len(valid_idx) == 0:
                        print("[WARN] Mask filtered out all points from point cloud.")
                        pts_np = np.zeros((0,3), dtype=np.float32)
                        col_np = np.zeros((0,3), dtype=np.float32)
                    elif len(valid_idx) < pts_np.shape[0]:
                        pts_np = pts_np[valid_idx]
                        col_np = col_np[valid_idx]

                pcd.points = o3d.utility.Vector3dVector(pts_np)
                pcd.colors = o3d.utility.Vector3dVector(col_np)

                points = pts_np
                cylinder_axis_start = np.array(cylinder_center) - np.array([0, 0, cylinder_height / 2])
                cylinder_axis_end   = np.array(cylinder_center) + np.array([0, 0, cylinder_height / 2])
                axis_dir = cylinder_axis_end - cylinder_axis_start
                axis_dir /= np.linalg.norm(axis_dir)

                v = points - cylinder_axis_start
                proj_len = np.dot(v, axis_dir)
                inside_mask = proj_len >= cylinder_heightUsable
                if not np.any(inside_mask):
                    print("[WARN] No points within cylinder length region.")
                    continue

                points_inside = points[inside_mask]
                v_inside = v[inside_mask]
                proj_len_inside = proj_len[inside_mask]
                proj_point_inside = cylinder_axis_start + np.outer(proj_len_inside, axis_dir)
                dist_to_axis = np.linalg.norm(points_inside - proj_point_inside, axis=1)
                dist_to_surface = dist_to_axis - cylinder_radius

                closest_distance = np.min(np.abs(dist_to_surface))
                min_idx = np.argmin(np.abs(dist_to_surface))
                closest_pt = points_inside[min_idx]

                print(f"[DEBUG] Closest perpendicular distance (mask {'ON' if mask_overlay else 'OFF'}): {(closest_distance * 100):.4f} cm")

                highlight_radius = 0.005
                dists = np.linalg.norm(points - closest_pt, axis=1)
                highlight_idx = np.where(dists < highlight_radius)[0]

                col_np_highlight = np.asarray(pcd.colors)
                col_np_highlight[highlight_idx] = [1.0, 0.0, 0.0]
                pcd.colors = o3d.utility.Vector3dVector(col_np_highlight)

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

            except Exception as e:
                print(f"[WARN] Open3D update failed: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        frame_worker.stop()
        cam.stop()
        cv2.destroyAllWindows()
        try:
            vis.destroy_window()
        except Exception:
            pass
        print("[INFO] Exiting.")


if __name__ == '__main__':
    main()
