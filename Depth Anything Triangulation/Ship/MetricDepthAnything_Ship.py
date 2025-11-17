"""
Real-time Metric Depth-Anything pipeline
- LGNet generative fill for masked regions.
- ZoeDepth metric model-based depth estimation.
- Open3D point cloud rendering of the metric depth map.
"""

import argparse
import json
import threading
import queue
import time
import sys
import os
import math
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# reduce OpenCV thread usage to avoid contention with PyTorch
cv2.setNumThreads(1)

# Open3D for realtime point cloud
import open3d as o3d

# LGNet generative fill integration
import torchvision.transforms.functional as TF
from PIL import Image

# ZoeDepth metric model imports
# Make sure repo_path points to your zoedepth repo if necessary.
repo_path = r"C:/Users/Torenia/Depth-Anything/metric_depth"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

try:
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config
except Exception as e:
    print("[WARN] Could not import zoedepth modules. Ensure repo path is correct.", e)
    # We'll still allow the script to start, but will fail on model init if missing.

# LGNet filler
repo_path_lg = r"C:/Users/Torenia/LGNet"
if repo_path_lg not in sys.path:
    sys.path.insert(0, repo_path_lg)

try:
    from options.test_options import TestOptions
    from models import create_model
except Exception as e:
    print("[WARN] LGNet imports failed. LGNet filler will be unavailable unless LGNet repo is present.", e)

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

        frame_tensor = (TF.to_tensor(frame_pil) * 2 - 1).unsqueeze(0)
        mask_tensor = TF.to_tensor(mask_img.convert("L")).unsqueeze(0)

        data = {'A': frame_tensor, 'B': mask_tensor, 'A_paths': ''}
        self.model.set_input(data)
        with torch.no_grad():
            self.model.forward()

        comp_img = lggnet_postprocess(self.model.merged_images3)
        comp_bgr = cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR)
        comp_bgr = cv2.resize(comp_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return comp_bgr

# Camera stream helper
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

# Utility masks + overlays
def _make_bool_mask_from_img(img_bgr, target_w, target_h, invert=False):
    if img_bgr is None:
        return None
    nonblack = np.any(img_bgr > 20, axis=2)
    if invert:
        bool_mask = ~nonblack
    else:
        bool_mask = nonblack
    if bool_mask.shape[1] != target_w or bool_mask.shape[0] != target_h:
        bool_mask = cv2.resize(bool_mask.astype(np.uint8),
                               (target_w, target_h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    return bool_mask

def draw_metric_scale_bar(img, current_distance_m, range_m=(0.0, 5.0), width=40):
    h, w = img.shape[:2]
    bar_h = int(h * 0.85)
    pad = 8
    x0 = w - width - pad
    y0 = int((h - bar_h) / 2)
    x1 = w - pad
    y1 = y0 + bar_h

    cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 40), -1)  # background
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

# ZoeDepth metric model wrapper
class ZoeMetric:
    def __init__(self, pretrained_resource, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[INFO] ZoeDepth device: {self.device}")

        # config + model build
        config = get_config("zoedepth", "eval", "nyu", pretrained_resource=pretrained_resource)
        self.model = build_model(config).to(self.device).eval()
        # use half if possible for speed
        try:
            if self.device.type == "cuda":
                self.model = self.model.half()
        except Exception:
            pass

        torch.backends.cudnn.benchmark = True

        # GPU color map LUT (magma) for quick colormap via indexing
        cmap_np = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_MAGMA)
        self.cmap_torch = torch.from_numpy(cmap_np[:, 0:3]).to(self.device).float() / 255.0  # (256,3)

    def predict_metric(self, frame_bgr):
        """
        Input: BGR numpy HxW x3 (uint8)
        Return: depth_m (H,W) float32 (meters), depth_vis_bgr (H,W,3) uint8 for display
        """
        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # to tensor on device
        frame_gpu = torch.from_numpy(frame_rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # preserve dtype: if model is half, convert
        if next(self.model.parameters()).dtype == torch.half:
            frame_gpu = frame_gpu.half()
        frame_gpu = frame_gpu / 255.0

        # resize to model expected
        img_t = F.interpolate(frame_gpu, (480, 640), mode='bilinear', align_corners=False)

        with torch.no_grad():
            pred = self.model(img_t)
            # model returns dict with 'metric_depth' key
            if isinstance(pred, dict):
                metric = pred.get("metric_depth", pred.get("out"))
            else:
                metric = pred

        # resize metric depth to original resolution
        depth = F.interpolate(metric, size=(H, W), mode='bilinear', align_corners=False)
        # depth tensor shape (1,1,H,W) or (1,H,W)
        if depth.ndim == 4:
            depth_map = depth[0, 0].detach()
        elif depth.ndim == 3:
            depth_map = depth[0].detach()
        else:
            depth_map = depth.detach().squeeze()

        # convert to cpu float32 numpy (meters)
        depth_m = depth_map.cpu().float().numpy()

        # visualization: clip to [0, max_vis] and map to 0-255
        return depth_m, self._colormap_from_depth(depth_m)

    def _colormap_from_depth(self, depth_m, max_vis=5.0):
        """
        Use GPU LUT for fast mapping if possible; otherwise fallback to CPU cv2.applyColorMap.
        Maps [0, max_vis] to [0,255].
        """
        d = np.clip(depth_m, 0.0, max_vis)
        depth_u8 = (d / max_vis * 255.0).round().astype(np.uint8)
        try:
            # Use CPU mapping (fast enough) to get BGR uint8
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
            return depth_color
        except Exception:
            # fallback: manual mapping using stored torch LUT
            idx = torch.from_numpy(depth_u8).to(self.cmap_torch.device).long()
            # idx shape HxW, need to index
            rgb = self.cmap_torch[idx]  # HxW x 3
            rgb_cpu = (rgb * 255.0).byte().cpu().numpy()
            return rgb_cpu

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=640)
    ap.add_argument("--display-height", type=int, default=360)
    ap.add_argument("--backend", type=str, default=None, help='cv2 backend e.g. cv2.CAP_DSHOW or cv2.CAP_V4L2')
    ap.add_argument("--profile", action='store_true')
    ap.add_argument("--mask-before-image", type=str, default="./Undistort-and-Depth/MaskAll.png",
                help="Path to an image mask file. Colored (non-black) areas are kept for LGNet input.")
    ap.add_argument("--mask-image", type=str, default="./Undistort-and-Depth/MaskA2.png",
                help="Path to an image mask file. Colored (non-black) areas are ignored for point cloud.")
    ap.add_argument("--debug-mask", action='store_true')
    ap.add_argument("--metric-model-path", type=str,
                    default="local::C:/Users/Torenia/OneDrive/Documents/GitHub/JAIST/Depth Anything Triangulation/Code/models/depth_anything_metric_depth_indoor.pt",
                    help="ZoeDepth metric pretrained resource path.")
    ap.add_argument("--max-depth-m", type=float, default=1.0, help="Max depth (m) for visualization/clipping")
    ap.add_argument("--point-skip", type=int, default=4, help="pixel stride for point cloud subsampling (bigger -> fewer points, faster)")
    ap.add_argument("--pc-window-width", type=int, default=960)
    ap.add_argument("--pc-window-height", type=int, default=720)
    args = ap.parse_args()

    # camera thread
    try:
        backend = getattr(cv2, args.backend) if args.backend else None
        cam = CameraStream(src=args.camera_id, backend=backend)
    except Exception as e:
        print(f"[ERROR] Camera init failed: {e}")
        sys.exit(1)

    # load masks if provided
    cap_w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask_before_img = mask_img = None
    mask_before_bool = mask_bool = None
    mask_overlay = True

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

    # initialize Zoe metric model
    try:
        print("[INFO] Loading ZoeDepth metric model...")
        zoe = ZoeMetric(pretrained_resource=args.metric_model_path, device=None)
    except Exception as e:
        print("[ERROR] Failed to init ZoeDepth metric model:", e)
        traceback.print_exc()
        cam.stop()
        sys.exit(1)

    # LGNet filler
    lgf = None

    # single-item queue for frames (latest only)
    frame_q = queue.Queue(maxsize=1)
    # small worker to keep grabbing frames into queue
    def frame_grabber(cam, q):
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.01)
                continue
            # keep only latest
            try:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Exception:
                        break
                q.put_nowait(frame)
            except queue.Full:
                pass
    grab_thread = threading.Thread(target=frame_grabber, args=(cam, frame_q), daemon=True)
    grab_thread.start()

    # Open3D visualizer setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Real-Time Metric Point Cloud", width=args.pc_window_width, height=args.pc_window_height)
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic, copy=True)
    R = np.array([[-1, 0, 0],
                  [ 0, 1, 0],
                  [ 0, 0, -1]], dtype=float)
    extrinsic[:3, :3] = extrinsic[:3, :3] @ R
    cam_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    # Add cylinder wireframe as reference
    cylinder_height = 0.24 # meters
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

    uv_grid_ready = False
    u_full = v_full = None

    fps_ema = 0.0
    alpha = 0.15
    last_time = time.time()

    # Mouse tracker for display cursor
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

    print("[INFO] Starting main loop (press 'q' on OpenCV window to quit)")

    try:
        while True:
            try:
                frame_full = frame_q.get(timeout=1.0)
            except queue.Empty:
                time.sleep(0.005)
                continue

            # LGNet fill
            try:
                if mask_before_bool is not None:
                    if lgf is None:
                        print("[INFO] Initializing LGNet generative fill...")
                        lgf = LGNetFiller()
                    mask_before_full = cv2.resize(mask_before_bool.astype(np.uint8),
                                                  (frame_full.shape[1], frame_full.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                    masked_frame = lgf.fill(frame_full, mask_before_full)
                else:
                    masked_frame = frame_full
            except Exception as e:
                print(f"[WARN] LGNet fill failed: {e} — falling back to raw frame.")
                masked_frame = frame_full

            # display left pane - masked frame
            display_masked = cv2.resize(masked_frame, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            # ZoeDepth metric prediction
            try:
                depth_m_full, depth_color_full = zoe.predict_metric(masked_frame)
                # depth_m_full shape HxW (meters)
            except Exception as e:
                print(f"[WARN] ZoeDepth metric prediction failed: {e}")
                depth_m_full = np.full((frame_full.shape[0], frame_full.shape[1]), args.max_depth_m / 2.0, dtype=np.float32)
                depth_color_full = np.full_like(display_masked, 128)

            H_full, W_full = depth_m_full.shape[:2]

            if not uv_grid_ready or (u_full is None) or (u_full.shape[1] != W_full or v_full.shape[0] != H_full):
                us = np.arange(W_full)
                vs = np.arange(H_full)
                u_full, v_full = np.meshgrid(us, vs)
                uv_grid_ready = True

            # prepare display depth (right pane)
            depth_disp = cv2.resize(depth_color_full, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            # overlay visuals on left
            vis_img = display_masked.copy()

            # unified mask overlay
            if mask_overlay:
                color_mask = np.zeros_like(vis_img)
                if mask_before_bool is not None:
                    mask_before_disp = cv2.resize(mask_before_bool.astype(np.uint8),
                                                  (args.display_width, args.display_height),
                                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                    color_mask[mask_before_disp] = (0, 255, 0)
                if mask_bool is not None:
                    mask_after_disp = cv2.resize(mask_bool.astype(np.uint8),
                                                 (args.display_width, args.display_height),
                                                 interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlap_disp = mask_after_disp & (np.any(color_mask > 0, axis=2))
                    color_mask[mask_after_disp] = (0, 0, 255)
                    color_mask[overlap_disp] = (0, 255, 255)
                vis_img = cv2.addWeighted(vis_img, 1.0, color_mask, 0.45, 0)

            # compute depth at mouse cursor on display resolution
            cx_disp, cy_disp = int(np.clip(mouse_x, 0, args.display_width - 1)), int(np.clip(mouse_y, 0, args.display_height - 1))
            # find corresponding index in full res:
            sx = W_full / float(args.display_width)
            sy = H_full / float(args.display_height)
            cx_full = int(np.clip(round(cx_disp * sx), 0, W_full - 1))
            cy_full = int(np.clip(round(cy_disp * sy), 0, H_full - 1))
            cursor_depth_m = float(depth_m_full[cy_full, cx_full])

            # draw test dot and depth label on left view
            cv2.circle(vis_img, (cx_disp, cy_disp), 6, (0,255,255), -1)
            cv2.putText(vis_img, f"{cursor_depth_m * 100:.2f} cm", (cx_disp + 10, cy_disp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # draw a simple a,b debug placeholder (no calibration anymore)
            cv2.putText(vis_img, f"Metric: ZoeDepth", (10, args.display_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            # depth visualization right pane with metric scale bar
            depth_vis_with_scale = draw_metric_scale_bar(depth_disp.copy(), cursor_depth_m, range_m=(0.0, args.max_depth_m), width=40)

            # combined image: left (masked_rgb) | right (depth)
            combined = np.hstack((vis_img, depth_vis_with_scale))

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_ema = alpha * inst_fps + (1 - alpha) * fps_ema
            fps_text = f"FPS: {fps_ema:.1f}"

            cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            cv2.imshow("LGNet Filled (L)  |  Metric Depth (R)", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mask_overlay = not mask_overlay
                print(f"[DEBUG] Mask overlay {'ON' if mask_overlay else 'OFF'}")


            # Build and update Open3D point cloud
            try:
                W_cam = float(W_full)
                H_cam = float(H_full)
                fov_x_deg = 95.0
                fov_y_deg = 2 * math.degrees(math.atan((H_cam / W_cam) * math.tan(math.radians(fov_x_deg / 2.0))))
                fx = (W_cam / 2.0) / math.tan(math.radians(fov_x_deg / 2.0))
                fy = (H_cam / 2.0) / math.tan(math.radians(fov_y_deg / 2.0))
                cx = W_cam / 2.0
                cy = H_cam / 2.0

                depth_o3d = o3d.geometry.Image((np.clip(depth_m_full, 0.0, args.max_depth_m) * 1000.0).astype(np.uint16))
                color_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
                color_o3d = o3d.geometry.Image(color_rgb)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d,
                    depth_scale=1000.0,
                    depth_trunc=float(args.max_depth_m),
                    convert_rgb_to_intensity=False
                )

                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=int(W_cam), height=int(H_cam),
                    fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy)
                )

                new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

                new_pcd.transform([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ])

                pts_np = np.asarray(new_pcd.points)
                col_np = np.asarray(new_pcd.colors)

                # mask filtering: if mask_bool present (True == kept/valid earlier), invert to be excluded set
                if mask_bool is not None and pts_np.shape[0] > 0:
                    mask_full = cv2.resize(mask_bool.astype(np.uint8), (W_full, H_full), interpolation=cv2.INTER_NEAREST).astype(bool)
                    mask_flat = mask_full.flatten()
                    # Keep where mask == 0 (unmasked/valid)
                    valid_idx = np.where(mask_flat == 0)[0]
                    if len(valid_idx) == 0:
                        pts_np = np.zeros((0,3), dtype=np.float32)
                        col_np = np.zeros((0,3), dtype=np.float32)
                    elif len(valid_idx) < pts_np.shape[0]:
                        pts_np = pts_np[valid_idx]
                        col_np = col_np[valid_idx]
                    else:
                        pass

                pcd.points = o3d.utility.Vector3dVector(pts_np)
                pcd.colors = o3d.utility.Vector3dVector(col_np)

                # compute closest perpendicular distance to cylinder axis (same as before)
                if pts_np.shape[0] > 0:
                    cylinder_axis_start = np.array(cylinder_center) - np.array([0, 0, cylinder_height / 2])
                    cylinder_axis_end   = np.array(cylinder_center) + np.array([0, 0, cylinder_height / 2])
                    axis_dir = cylinder_axis_end - cylinder_axis_start
                    axis_dir /= np.linalg.norm(axis_dir)

                    v = pts_np - cylinder_axis_start
                    proj_len = np.dot(v, axis_dir)

                    inside_mask = proj_len >= cylinder_heightUsable
                    if np.any(inside_mask):
                        points_inside = pts_np[inside_mask]
                        proj_len_inside = proj_len[inside_mask]
                        proj_point_inside = cylinder_axis_start + np.outer(proj_len_inside, axis_dir)
                        dist_to_axis = np.linalg.norm(points_inside - proj_point_inside, axis=1)
                        dist_to_surface = dist_to_axis - cylinder_radius

                        closest_distance = np.min(np.abs(dist_to_surface))
                        min_idx = np.argmin(np.abs(dist_to_surface))
                        closest_pt = points_inside[min_idx]
                        print(f"[DEBUG] Closest perp distance: {(closest_distance * 100):.4f} cm")
                        # highlight nearest point
                        highlight_radius = 0.005
                        dists = np.linalg.norm(pts_np - closest_pt, axis=1)
                        highlight_idx = np.where(dists < highlight_radius)[0]
                        col_np_highlight = np.asarray(pcd.colors)
                        if highlight_idx.size > 0 and col_np_highlight.shape[0] >= highlight_idx.max() + 1:
                            col_np_highlight[highlight_idx] = [1.0, 0.0, 0.0]
                            pcd.colors = o3d.utility.Vector3dVector(col_np_highlight)
                    else:
                        # no inside points - skip highlighting
                        pass

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

            except Exception as e:
                print(f"[WARN] Open3D update failed: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] Cleaning up...")
        try:
            vis.destroy_window()
        except Exception:
            pass
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Exiting.")

if __name__ == "__main__":
    main()