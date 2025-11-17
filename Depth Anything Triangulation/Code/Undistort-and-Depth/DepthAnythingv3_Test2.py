"""
DepthAnything v3 -- Minimal viewer (LGNet fill optional + DA3 relative depth)
Kept minimal: Camera -> (optional LGNet fill) -> DA3 -> Relative depth color map -> Display

Notes:
- Fixes: robust handling of DA3 output shapes/types, guarantees 2D uint8 for applyColorMap.
- Removed: metric calibration, Open3D, pointclouds, scale bars, etc.
"""

import argparse
import threading
import queue
import time
import sys
import os
import math

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F

# -------------------------
# Add Depth-Anything v3 repo path if needed (adjust to your path)
# -------------------------
repo_path_v3 = r"C:/Users/Torenia/Depth-Anything-3/src"
if repo_path_v3 not in sys.path:
    sys.path.insert(0, repo_path_v3)

from depth_anything_3.api import DepthAnything3

# -------------------------
# Add LGNet repo path if needed (adjust to your path)
# -------------------------
repo_path_lg = r"C:/Users/Torenia/LGNet"
if repo_path_lg not in sys.path:
    sys.path.insert(0, repo_path_lg)

# LGNet imports (these modules are from the LGNet repo)
from options.test_options import TestOptions
from models import create_model


# -------------------------
# CameraStream (threaded)
# -------------------------
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


# -------------------------
# LGNet filler (minimal wrapper)
# -------------------------
def lggnet_postprocess(img_tensor):
    img = (img_tensor + 1) / 2 * 255
    img = img.permute(0, 2, 3, 1).int().cpu().numpy().astype(np.uint8)
    return img[0]


class LGNetFiller:
    def __init__(self):
        # keep CLI args expected by LGNet test harness but don't disturb user's args
        _argv_backup = sys.argv.copy()
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

        sys.argv = _argv_backup

    def fill(self, frame_bgr, mask_bool):
        """frame_bgr: HxWx3 BGR, mask_bool: HxW boolean (True = to fill)"""
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


# -------------------------
# DepthAnything v3 wrapper (relative depth only)
# -------------------------
class DepthAnything3Estimator:
    def __init__(self,
                 model_name="depth-anything/DA3-SMALL",
                 device=None,
                 display_size=(640, 360),
                 inference_size=(504, 378)):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[INFO] Using device: {self.device}")

        print(f"[INFO] Loading DA3 model: {model_name}")
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        torch.set_float32_matmul_precision("high")

        self.inf_w, self.inf_h = inference_size
        self.disp_w, self.disp_h = display_size

        self.depth_smooth = None
        self.smooth_alpha = 0.35

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

    def _coerce_depth_to_2d_float(self, depth):
        """
        Accepts depth which may be:
          - torch.Tensor (-> converted outside)
          - numpy array shapes: (1,H,W), (H,W,1), (H,W), (H,W,3)
        Returns HxW float32 array.
        """
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()

        depth = np.asarray(depth)

        # If shape (1, H, W) -> take first
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth2 = depth[0]
        # If shape (H, W, 1) -> squeeze last
        elif depth.ndim == 3 and depth.shape[2] == 1:
            depth2 = depth[:, :, 0]
        # If shape (H, W, 3) -> convert to single channel by taking first channel
        elif depth.ndim == 3 and depth.shape[2] == 3:
            # take luminance-like combination OR first channel; using first channel is minimal change
            depth2 = depth[:, :, 0]
        elif depth.ndim == 2:
            depth2 = depth
        else:
            # fallback: squeeze everything to 2D if possible
            depth2 = np.squeeze(depth)
            if depth2.ndim != 2:
                # last resort: take mean across last axis
                depth2 = np.mean(depth2, axis=-1)

        return depth2.astype(np.float32)

    def estimate_depth(self, frame_bgr, profile=False):
        """
        Returns:
            depth_color_bgr (uint8 HxWx3), depth_rel (float32 HxW)
        """
        t0 = time.time()
        H, W = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (self.inf_w, self.inf_h), interpolation=cv2.INTER_LINEAR)

        # model inference
        try:
            prediction = self.model.inference([rgb_small])
        except Exception as e:
            raise RuntimeError(f"DA3 inference failed: {e}")

        depth = getattr(prediction, "depth", None)
        if depth is None:
            raise RuntimeError("DA3 returned no depth in prediction object.")

        # coerce to 2D float32
        depth_2d = self._coerce_depth_to_2d_float(depth)

        # resize to original frame size (float)
        depth_resized = cv2.resize(depth_2d, (W, H), interpolation=cv2.INTER_LINEAR)

        # smoothing (exponential moving average)
        if self.depth_smooth is None:
            self.depth_smooth = depth_resized.astype(np.float32)
        else:
            self.depth_smooth = (1.0 - self.smooth_alpha) * self.depth_smooth + self.smooth_alpha * depth_resized.astype(np.float32)

        # Ensure final depth array is float32 HxW
        depth_out = self.depth_smooth.astype(np.float32)

        # Normalize to 0..255 uint8 for applyColorMap
        depth_uint8 = cv2.normalize(depth_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Colorize
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)

        if profile:
            t1 = time.time()
            print(f"[PROFILE] DA3 total {(t1-t0)*1000:.1f}ms")

        return depth_color, depth_out


# -------------------------
# Frame worker to prevent blocking camera
# -------------------------
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


# -------------------------
# main()
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=640)
    ap.add_argument("--display-height", type=int, default=360)
    ap.add_argument("--inference-width", type=int, default=504)
    ap.add_argument("--inference-height", type=int, default=378)
    ap.add_argument("--backend", type=str, default=None, help='cv2 backend e.g. cv2.CAP_DSHOW')
    ap.add_argument("--profile", action='store_true')
    ap.add_argument("--mask-before-image", type=str, default="./Undistort-and-Depth/MaskTrain2.png",
                    help="Path to an image mask file. Colored (non-black) areas are considered valid for LGNet fill.")
    
    args = ap.parse_args()

    try:
        backend = getattr(cv2, args.backend) if args.backend else None
        cam = CameraStream(src=args.camera_id, backend=backend)
    except Exception as e:
        print(f"[ERROR] Camera init failed: {e}")
        sys.exit(1)

    # instantiate DA3 estimator
    depth_est = DepthAnything3Estimator(
        model_name="depth-anything/DA3-SMALL",
        display_size=(args.display_width, args.display_height),
        inference_size=(args.inference_width, args.inference_height),
    )

    cap_w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # optional LGNet mask-before image (colored = valid)
    mask_before_bool = None
    if args.mask_before_image and os.path.exists(args.mask_before_image):
        tmp = cv2.imread(args.mask_before_image, cv2.IMREAD_COLOR)
        if tmp is not None:
            nonblack = np.any(tmp > 20, axis=2)
            mask_before_bool = cv2.resize((~nonblack).astype(np.uint8), 
                                        (cap_w, cap_h),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
            print(f"[INFO] Loaded mask-before image: colored areas will be used for LGNet fill.")
        else:
            print(f"[WARN] Failed to read mask-before image at {args.mask_before_image}")

    frame_q = queue.Queue(maxsize=1)
    frame_worker = FrameWorker(cam, frame_q)

    fps_ema = 0.0
    alpha = 0.15
    last_time = time.time()

    cv2.namedWindow("LGNet Filled (Left)  |  Relative Depth (Right)")

    try:
        while True:
            try:
                frame_full = frame_q.get(timeout=1.0)
            except queue.Empty:
                time.sleep(0.005)
                continue

            # LGNet fill if available
            try:
                if mask_before_bool is not None:
                    if not hasattr(depth_est, "lggnet"):
                        print("[INFO] Initializing LGNet generative fill...")
                        depth_est.lggnet = LGNetFiller()
                    masked_frame = depth_est.lggnet.fill(frame_full, mask_before_bool)
                else:
                    masked_frame = frame_full
            except Exception as e:
                print(f"[WARN] LGNet fill failed: {e} — using raw frame.")
                masked_frame = frame_full

            # DepthAnything relative depth
            try:
                depth_colored_full, depth_rel_full = depth_est.estimate_depth(masked_frame, profile=args.profile)
                depth_disp = cv2.resize(depth_colored_full, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"[WARN] Depth estimation error: {e}")
                depth_disp = np.full((args.display_height, args.display_width, 3), 128, dtype=np.uint8)
                depth_rel_full = np.full((frame_full.shape[0], frame_full.shape[1]), 0.5, dtype=np.float32)

            display_masked = cv2.resize(masked_frame, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            combined = np.hstack((display_masked, depth_disp))

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_ema = alpha * inst_fps + (1 - alpha) * fps_ema
            cv2.putText(combined, f"FPS: {fps_ema:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            cv2.imshow("LGNet Filled (Left)  |  Relative Depth (Right)", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        frame_worker.stop()
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Exiting.")

if __name__ == "__main__":
    main()
