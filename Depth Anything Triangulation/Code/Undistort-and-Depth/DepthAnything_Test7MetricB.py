"""
Optimized real-time Depth-Anything (small) pipeline
+ integrated pseudo-metric calibration using a fixed bar (start/end) and center plate.
Undistortion step removed - works directly with raw camera frames.

ADDED: real-time Open3D point cloud rendering of the metric depth map (subsampled for speed).
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

# --- ADDED/CHANGED: import Open3D for realtime point cloud ---
import open3d as o3d


# ===================================================
# --- METRIC CALIBRATION SETTINGS (adjust here) ---
# ===================================================
# bar endpoints in pixels (image coordinates of raw capture)
# These are pixel coordinates in the raw capture resolution (not display size).
BAR_START_PX = (310, 0)   # near end (x0,y0) near top edge
BAR_END_PX   = (310, 180) # far end (x1,y1) closer to center
BAR_START_DISTANCE_M = 0.0   # meters to bar start (near end) length is 0.03 but account for cylinder radius #0.054
BAR_END_DISTANCE_M   = 0.222   # meters to bar end (far end) length is 0.24 but account for cylinder radius #0.244
BAR_PIXEL_HALF_WIDTH = 10     # half-width in pixels to average across the bar width

# plate: center location and radius in px (in raw capture coordinates)
PLATE_CENTER_PX = (310, 220)    # if None, will use image center
PLATE_RADIUS_PX = 20      # radius in px for averaging
PLATE_DISTANCE_M = 0.222    # known distance to plate (meters) #0.24

# calibration smoothing
CAL_ALPHA = 0.08   # EMA smoothing for a,b

# limit for pseudo-metric depth (clamp)
DEPTH_MIN_M = 0.054
DEPTH_MAX_M = 1

# visualization: metric scale bar width (pixels) and range
SCALE_BAR_WIDTH = 40
SCALE_BAR_RANGE_M = (0.0, 1.0)  # min, max shown on vertical scale bar

# ===================================================
# --- Camera / Model / Pipeline classes ---
# ===================================================
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
            import platform
            # --- Skip torch.compile on Windows since Triton isn't available ---
            if platform.system().lower() != "windows":
                try:
                    import triton
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("[INFO] Model compiled with torch.compile")
                except Exception as e:
                    print(f"[WARN] torch.compile skipped/failed: {e}")
            else:
                print("[INFO] Skipping torch.compile on Windows (Triton not supported)")
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

        # smoothing (exponential) for model depth output
        self.depth_smooth = None
        self.smooth_alpha = 0.3

        # internal flags
        self._use_direct_preproc = True  # attempted fast path; will fallback to HF processor if needed

        # warmup
        self._warmup()

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

    def estimate_depth(self, frame_bgr, profile=False):
        """
        Estimate depth for raw BGR frame.
        RETURNS: (depth_color_bgr (H,W,3) uint8, depth_map_float (H,W) float32)
        depth_map_float is the model's (smoothed) relative depth (not yet metric).
        """
        t0 = time.time()
        H, W = frame_bgr.shape[:2]

        # resize to inference size once (CPU) - keep as uint8 to avoid unnecessary conversions
        rgb_small = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
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
            return depth_color, self.depth_smooth.astype(np.float32)

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

        return depth_color, self.depth_smooth.astype(np.float32)


# Frame worker: reads latest frame from CameraStream and writes to a single-item queue
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
            # keep only latest frame in queue
            try:
                # empty queue if full
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


# ===================================================
# --- Metric calibration helpers & visual overlays ---
# ===================================================
def make_bar_mask(shape, p0, p1, half_width):
    """Return boolean mask for a rectangular region centered on line p0->p1 with given half_width."""
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    p0i = (int(round(p0[0])), int(round(p0[1])))
    p1i = (int(round(p1[0])), int(round(p1[1])))
    # draw thick line onto mask
    cv2.line(mask, p0i, p1i, color=255, thickness=max(1, int(half_width * 2)))
    return mask.astype(bool)


def make_plate_mask(shape, center_px, radius):
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cx, cy = int(round(center_px[0])), int(round(center_px[1]))
    cv2.circle(mask, (cx, cy), int(round(radius)), 255, -1)
    return (mask > 0)


def compute_median_in_mask(depth_rel, mask):
    vals = depth_rel[mask]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def solve_nonlinear_bar_mapping(v1, D1, v2, D2, offset_m):
    """Nonlinear bar mapping using Pythagorean correction for lateral offset."""
    if v1 is None or v2 is None or abs(v1 - v2) < 1e-6:
        return None, None
    # Convert known distances to optical-axis distances (z)
    z1 = math.sqrt(max(D1**2 - offset_m**2, 0.0))
    z2 = math.sqrt(max(D2**2 - offset_m**2, 0.0))
    A = np.array([[v1, 1.0], [v2, 1.0]], dtype=np.float64)
    b = np.array([z1, z2], dtype=np.float64)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(sol[0]), float(sol[1])


def refine_with_plate(a, b, depth_rel, plate_mask, plate_distance_m, plate_correction_gain=0.3):
    """Refine offset b by comparing predicted plate distance and known plate distance."""
    if plate_mask is None:
        return a, b
    d_plate_rel = compute_median_in_mask(depth_rel, plate_mask)
    if d_plate_rel is None:
        return a, b
    z_pred = a * d_plate_rel + b
    err = plate_distance_m - z_pred
    b_refined = b + plate_correction_gain * err
    return a, b_refined


def draw_metric_scale_bar(img, current_distance_m, range_m=(0.0, 5.0), width=SCALE_BAR_WIDTH):
    """Draw a vertical metric scale bar on the right of an image and return image with bar."""
    h, w = img.shape[:2]
    bar_h = int(h * 0.85)
    pad = 8
    x0 = w - width - pad
    y0 = int((h - bar_h) / 2)
    x1 = w - pad
    y1 = y0 + bar_h

    cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 40), -1)  # background
    # ticks and labels
    min_m, max_m = range_m
    num_ticks = 6
    for i in range(num_ticks + 1):
        fy = i / num_ticks
        yy = int(y1 - fy * bar_h)
        cv2.line(img, (x0, yy), (x0 + 8, yy), (200, 200, 200), 1)
        val = min_m + fy * (max_m - min_m)
        txt = f"{val:.1f}m"
        cv2.putText(img, txt, (x0 - 45, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
    # current marker
    # clamp current into range
    cur = np.clip(current_distance_m, min_m, max_m)
    frac = (cur - min_m) / (max_m - min_m) if (max_m - min_m) > 1e-6 else 0.0
    yy_cur = int(y1 - frac * bar_h)
    cv2.circle(img, (x0 + 12, yy_cur), 6, (0, 255, 255), -1)
    cv2.putText(img, f"{current_distance_m:.2f}m", (x0 - 80, yy_cur + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img


# ===================================================
# --- Main
# ===================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=640)
    ap.add_argument("--display-height", type=int, default=360)
    ap.add_argument("--inference-width", type=int, default=192 * 3)
    ap.add_argument("--inference-height", type=int, default=192 * 3)
    ap.add_argument("--backend", type=str, default=None, help='cv2 backend e.g. cv2.CAP_DSHOW or cv2.CAP_V4L2')
    ap.add_argument("--profile", action='store_true')
    ap.add_argument("--mask-image", type=str, default="./Undistort-and-Depth/Mask.png",
                help="Path to an image mask file. Colored (non-black) areas are ignored for proximity distance.")
    ap.add_argument("--debug-mask", action='store_true',
                help="Toggle display of the mask overlay (press 'm' to toggle).")
    # --- ADDED/CHANGED: options for point cloud rendering ---
    ap.add_argument("--point-skip", type=int, default=4, help="pixel stride for point cloud subsampling (bigger -> fewer points, faster)")
    ap.add_argument("--pc-window-width", type=int, default=960)
    ap.add_argument("--pc-window-height", type=int, default=720)
    args = ap.parse_args()

    # create camera thread
    try:
        backend = getattr(cv2, args.backend) if args.backend else None
        cam = CameraStream(src=args.camera_id, backend=backend)
    except Exception as e:
        print(f"[ERROR] Camera init failed: {e}")
        sys.exit(1)

    # prepare depth estimator
    depth_est = DepthEstimator(
        display_size=(args.display_width, args.display_height),
        inference_size=(args.inference_width, args.inference_height),
    )

    # get capture dimensions
    cap_w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Load mask image (optional)
    mask_img = None
    mask_bool = None
    mask_overlay = True  # runtime toggle state

    if args.mask_image and os.path.exists(args.mask_image):
        mask_img = cv2.imread(args.mask_image, cv2.IMREAD_COLOR)
        if mask_img is not None:
            mask_img = cv2.resize(mask_img, (cap_w, cap_h))
            # anything non-black is masked (ignored)
            mask_bool = np.any(mask_img > 20, axis=2)
            print(f"[INFO] Loaded mask image: {args.mask_image}")
        else:
            print(f"[WARN] Failed to read mask image at {args.mask_image}")

    # if plate center not provided, default to capture image center
    global PLATE_CENTER_PX
    if PLATE_CENTER_PX is None:
        PLATE_CENTER_PX = (cap_w // 2, cap_h // 2)

    # single-item queue for frames (latest only)
    frame_q = queue.Queue(maxsize=1)
    frame_worker = FrameWorker(cam, frame_q)

    fps_ema = 0.0
    alpha = 0.15
    last_time = time.time()

    # calibration EMA state
    a_ema = None
    b_ema = None

    # --- ADDED/CHANGED: Open3D visualizer + point cloud setup ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Real-Time Metric Point Cloud", width=args.pc_window_width, height=args.pc_window_height)
    
    # --- Rotate Open3D camera view 180° around Y-axis (face opposite direction) ---
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()

    # Make the extrinsic writable
    extrinsic = np.array(cam_params.extrinsic, copy=True)

    # 180° rotation around Y-axis matrix
    R = np.array([[-1, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, -1]], dtype=float)

    # Apply rotation (multiply extrinsic rotation part)
    extrinsic[:3, :3] = extrinsic[:3, :3] @ R

    # Reassign to camera parameters
    cam_params.extrinsic = extrinsic

    # Apply back to the view control
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    # --- DEBUG: add a visible dummy geometry before the main loop ---
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # vis.add_geometry(frame)

    # --- Add cylindrical wireframe representing structure in front of camera ---
    cylinder_height = 0.24 # meters
    cylinder_heightUsable = 0.08 # meters
    cylinder_radius = 0.045  # meters
    cylinder_center = [0.0, 0.0, -0.12]  # position in front of camera (+Z)

    # Create the cylinder
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=cylinder_radius,
        height=cylinder_height,
        resolution=40,
        split=1
    )

    # Move cylinder to its center position
    cylinder.translate(cylinder_center)

    # Convert to wireframe (show only edges)
    cylinder_wire = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder)
    cylinder_wire.paint_uniform_color([0.0, 1.0, 0.0])  # green wireframe

    # Add to visualizer
    vis.add_geometry(cylinder_wire)


    pcd = o3d.geometry.PointCloud()
    # initialize with placeholder points
    init_points = np.zeros((1, 3), dtype=np.float32)
    init_colors = np.zeros((1, 3), dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(init_points)
    pcd.colors = o3d.utility.Vector3dVector(init_colors)
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([0.0, 0.0, 0.0])
    # optional: choose a nicer camera view (user can change interactively)
    ctr = vis.get_view_control()

    # Precompute pixel coordinates grid for backprojection (full resolution)
    # We'll use these to create XYZ from u,v,z quickly and then subsample with a stride.
    uv_grid_ready = False
    u_full = v_full = None

    print("[INFO] Starting main loop (press 'q' in the OpenCV window to quit)")

    try:
        # mouse position tracker (for test dot on both sides)
        mouse_x, mouse_y = args.display_width // 2, args.display_height // 2
        def mouse_callback(event, x, y, flags, param):
            nonlocal mouse_x, mouse_y
            if event == cv2.EVENT_MOUSEMOVE:
                # Map mouse position to display coordinates (accounting for combined view)
                # Combined view is side-by-side, so x ranges from 0 to 2*display_width
                if x < args.display_width:
                    # Left side (raw frame)
                    mouse_x, mouse_y = x, y
                else:
                    # Right side (depth), map back to single frame coordinates
                    mouse_x, mouse_y = x - args.display_width, y

        cv2.namedWindow("Raw Frame (L)  |  Metric Depth (R)")
        cv2.setMouseCallback("Raw Frame (L)  |  Metric Depth (R)", mouse_callback)
        
        while True:
            try:
                frame_full = frame_q.get(timeout=1.0)
            except queue.Empty:
                time.sleep(0.005)
                continue

            # Step: create display-sized raw image (for visualization only)
            display_raw = cv2.resize(frame_full, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            # depth estimation: get both colorized and float relative depth
            try:
                depth_colored_full, depth_rel_full = depth_est.estimate_depth(frame_full, profile=args.profile)
                # depth_colored_full is same dims as frame_full
                depth_disp = cv2.resize(depth_colored_full, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"[WARN] Depth estimation error: {e}")
                depth_disp = np.full_like(display_raw, 128)
                depth_rel_full = np.full((frame_full.shape[0], frame_full.shape[1]), 0.5, dtype=np.float32)

            H_full, W_full = depth_rel_full.shape[:2]

            # prepare pixel grid if not ready
            if not uv_grid_ready or (u_full.shape[1] != W_full or v_full.shape[0] != H_full):
                # create u (x) and v (y) coordinate grids
                us = np.arange(W_full)
                vs = np.arange(H_full)
                u_full, v_full = np.meshgrid(us, vs)
                uv_grid_ready = True

            # -----------------------
            # Metric calibration logic
            # -----------------------
            # build masks in full capture coordinates
            bar_mask = make_bar_mask((H_full, W_full), BAR_START_PX, BAR_END_PX, BAR_PIXEL_HALF_WIDTH)
            plate_mask = make_plate_mask((H_full, W_full), PLATE_CENTER_PX, PLATE_RADIUS_PX)

            # medians on bar endpoints: sample small windows near start/end using perpendicular projection
            # we'll approximate by sampling small boxes around the exact start/end pixel
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

            # fallback: compute median across mask halves if point sampling fails
            if d_start_rel is None:
                # take quarter near BAR_START side of mask
                d_start_rel = compute_median_in_mask(depth_rel_full, bar_mask)
            if d_end_rel is None:
                d_end_rel = compute_median_in_mask(depth_rel_full, bar_mask)

            # Solve mapping from bar endpoints
            CAM_OFFSET_M = 0.06
            a_new, b_new = solve_nonlinear_bar_mapping(
                d_start_rel, BAR_START_DISTANCE_M,
                d_end_rel, BAR_END_DISTANCE_M,
                CAM_OFFSET_M
            )

            # If solve failed, fallback to previous or default
            if a_new is None or b_new is None:
                if a_ema is None:
                    a_new, b_new = 1.0, 0.0
                else:
                    a_new, b_new = a_ema, b_ema

            # refine with plate
            a_ref, b_ref = refine_with_plate(a_new, b_new, depth_rel_full, plate_mask, PLATE_DISTANCE_M, plate_correction_gain=0.3)

            # EMA smoothing for a,b
            if a_ema is None:
                a_ema, b_ema = a_ref, b_ref
            else:
                a_ema = (1 - CAL_ALPHA) * a_ema + CAL_ALPHA * a_ref
                b_ema = (1 - CAL_ALPHA) * b_ema + CAL_ALPHA * b_ref

            # compute pseudo-metric depth map (full resolution)
            z_est = a_ema * depth_rel_full + b_ema
            depth_m_full = np.sqrt(np.maximum(z_est**2 + CAM_OFFSET_M**2, 0.0))
            depth_m_full = np.clip(depth_m_full, DEPTH_MIN_M, DEPTH_MAX_M)

            # Downsample depth_m to display resolution for overlay / visualization
            depth_m_disp = cv2.resize(depth_m_full, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            # --- Visualization overlays on display_raw ---
            vis_img = display_raw.copy()

            # map coordinates of bar & plate from full -> display space
            sx = args.display_width / float(W_full)
            sy = args.display_height / float(H_full)
            def scale_pt(pt):
                return (int(round(pt[0] * sx)), int(round(pt[1] * sy)))

            bar_start_disp = scale_pt(BAR_START_PX)
            bar_end_disp   = scale_pt(BAR_END_PX)
            plate_center_disp = scale_pt(PLATE_CENTER_PX)
            bar_half_width_disp = max(1, int(round(BAR_PIXEL_HALF_WIDTH * (sx + sy) * 0.5)))

            # draw bar line + wide debug band
            cv2.line(vis_img, bar_start_disp, bar_end_disp, (255, 200, 0), 2)
            # approximate rectangle around bar for debugging
            # draw thick line as band
            cv2.line(vis_img, bar_start_disp, bar_end_disp, (255, 200, 0), thickness=max(3, bar_half_width_disp*2))
            #cv2.putText(vis_img, f"bar: {BAR_START_DISTANCE_M:.2f}m -> {BAR_END_DISTANCE_M:.2f}m",
            #            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)

            # draw plate circle + box
            cv2.circle(vis_img, plate_center_disp, max(5, int(round(PLATE_RADIUS_PX * (sx + sy) * 0.5))), (0,255,0), 2)
            xpl = plate_center_disp[0] - int(round(PLATE_RADIUS_PX * sx))
            ypl = plate_center_disp[1] - int(round(PLATE_RADIUS_PX * sy))
            cv2.rectangle(vis_img, (xpl, ypl), (xpl + int(round(2*PLATE_RADIUS_PX * sx)), ypl + int(round(2*PLATE_RADIUS_PX * sy))), (0,255,0), 1)
            #cv2.putText(vis_img, f"plate: {PLATE_DISTANCE_M:.2f}m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # compute depth at mouse cursor position
            cx_disp, cy_disp = int(np.clip(mouse_x, 0, args.display_width - 1)), int(np.clip(mouse_y, 0, args.display_height - 1))
            cursor_depth_m = float(depth_m_disp[cy_disp, cx_disp])

            # draw moving yellow depth test dot
            cv2.circle(vis_img, (cx_disp, cy_disp), 6, (0,255,255), -1)
            cv2.putText(vis_img, f"{cursor_depth_m * 100:.2f} cm", (cx_disp + 10, cy_disp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # show a,b debug
            cv2.putText(vis_img, f"a={a_ema:.4f} b={b_ema:.3f}", (10, args.display_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            # prepare colored depth visualization (right side)
            depth_vis = cv2.normalize(depth_m_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # draw metric scale bar on the right of depth_vis
            depth_vis_with_scale = draw_metric_scale_bar(depth_vis, cursor_depth_m, range_m=SCALE_BAR_RANGE_M, width=SCALE_BAR_WIDTH)

            if mask_overlay and mask_bool is not None:
                mask_disp = cv2.resize(mask_bool.astype(np.uint8) * 255,
                                    (args.display_width, args.display_height))
                color_mask = np.zeros_like(vis_img)
                color_mask[mask_disp > 0] = (0, 0, 255)
                vis_img = cv2.addWeighted(vis_img, 1.0, color_mask, 0.4, 0)
                
            combined = np.hstack((vis_img, depth_vis_with_scale))

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_ema = alpha * inst_fps + (1 - alpha) * fps_ema
            fps_text = f"FPS: {fps_ema:.1f}"

            cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            cv2.imshow("Raw Frame (L)  |  Metric Depth (R)", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mask_overlay = not mask_overlay
                print(f"[DEBUG] Mask overlay {'ON' if mask_overlay else 'OFF'}")

            # --- Build and update Open3D point cloud using RGBD projection ---
            try:
                # Intrinsics
                W_cam = float(W_full)
                H_cam = float(H_full)
                fov_x_deg = 95.0
                fov_y_deg = 2 * math.degrees(math.atan((H_cam / W_cam) * math.tan(math.radians(fov_x_deg / 2.0))))
                fx = (W_cam / 2.0) / math.tan(math.radians(fov_x_deg / 2.0))
                fy = (H_cam / 2.0) / math.tan(math.radians(fov_y_deg / 2.0))
                cx = W_cam / 2.0
                cy = H_cam / 2.0

                depth_o3d = o3d.geometry.Image((np.clip(depth_m_full, 0.0, DEPTH_MAX_M) * 1000.0).astype(np.uint16))
                color_rgb = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
                color_o3d = o3d.geometry.Image(color_rgb)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d,
                    depth_scale=1000.0,
                    depth_trunc=float(DEPTH_MAX_M),
                    convert_rgb_to_intensity=False
                )

                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=int(W_cam), height=int(H_cam),
                    fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy)
                )

                new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

                # Fix coordinate system (OpenCV -> Open3D)
                new_pcd.transform([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ])

                # Convert to numpy arrays for filtering
                pts_np = np.asarray(new_pcd.points)
                col_np = np.asarray(new_pcd.colors)

                # ---------------------------------------------------------
                # MASK FILTERING — only apply when overlay is enabled
                # ---------------------------------------------------------
                if mask_overlay and mask_bool is not None:
                    mask_small = cv2.resize(mask_bool.astype(np.uint8), (W_full, H_full), interpolation=cv2.INTER_NEAREST)
                    mask_flat = mask_small.flatten()
                    valid_idx = np.where(mask_flat == 0)[0]
                    if len(valid_idx) > 0 and len(valid_idx) < len(pts_np):
                        pts_np = pts_np[valid_idx]
                        col_np = col_np[valid_idx]
                    else:
                        print("[WARN] Mask removed all or no points from point cloud.")

                # Update point cloud geometry
                pcd.points = o3d.utility.Vector3dVector(pts_np)
                pcd.colors = o3d.utility.Vector3dVector(col_np)
                
                # ---------------------------------------------
                # Compute closest perpendicular distance to cylinder axis
                # ---------------------------------------------
                points = pts_np  # already masked if overlay on

                cylinder_axis_start = np.array(cylinder_center) - np.array([0, 0, cylinder_height / 2])
                cylinder_axis_end   = np.array(cylinder_center) + np.array([0, 0, cylinder_height / 2])
                axis_dir = cylinder_axis_end - cylinder_axis_start
                axis_dir /= np.linalg.norm(axis_dir)

                v = points - cylinder_axis_start
                proj_len = np.dot(v, axis_dir)

                # --- NEW: ignore points outside cylinder length ---
                #inside_mask = (proj_len >= 0) & (proj_len >= cylinder_heightUsable)
                inside_mask = proj_len >= cylinder_heightUsable
                if not np.any(inside_mask):
                    print("[WARN] No points within cylinder length region.")
                    continue  # skip this frame

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

                # =====================================================
                # 3D Highlight in Point Cloud
                # =====================================================
                # Highlight this point and neighbors (if available)
                highlight_radius = 0.005  # 5mm neighborhood
                dists = np.linalg.norm(points - closest_pt, axis=1)
                highlight_idx = np.where(dists < highlight_radius)[0]

                col_np_highlight = np.asarray(pcd.colors)
                col_np_highlight[highlight_idx] = [1.0, 0.0, 0.0]  # bright red
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