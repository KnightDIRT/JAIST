"""
Stripped-down Depth-Anything demo
- Removed metric calibration and Open3D parts
- Shows LGNet inpainted frame (left) and relative depth estimation (right)
- Keeps mask-before (optional) and LGNet generative fill

Usage:
python DepthAnything_Test11_NoMetric.py --camera-id 0 --display-width 640 --display-height 360 

"""

import argparse
import sys
import os
import threading
import queue
import time

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForDepthEstimation

# reduce OpenCV thread usage to avoid contention with PyTorch
cv2.setNumThreads(1)

# --- LGNet generative fill integration ---
import torchvision.transforms.functional as F
from PIL import Image

# If you use a local LGNet repo, adjust repo_path or remove LGNet usage
repo_path = r"C:/Users/Torenia/LGNet"
if os.path.isdir(repo_path) and repo_path not in sys.path:
    sys.path.insert(0, repo_path)

try:
    from options.test_options import TestOptions
    from models import create_model
    _HAS_LGNET = True
except Exception:
    # LGNet optional: script will fall back to raw camera frames if LGNet not available
    _HAS_LGNET = False


def lggnet_postprocess(img_tensor):
    img = (img_tensor + 1) / 2 * 255
    img = img.permute(0, 2, 3, 1).int().cpu().numpy().astype(np.uint8)
    return img[0]


class LGNetFiller:
    def __init__(self):
        if not _HAS_LGNET:
            raise RuntimeError("LGNet modules not found on PYTHONPATH")
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


# ===================================================
# Camera / Depth Estimator / Pipeline classes
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
        self.smooth_alpha = 0.5

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
        depth_map_float is the model's (smoothed) relative depth (not metric).
        """
        t0 = time.time()
        H, W = frame_bgr.shape[:2]

        # resize to inference size once (CPU) - keep as uint8 to avoid unnecessary conversions
        rgb_small = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                               (self.inference_w, self.inference_h), interpolation=cv2.INTER_LINEAR)

        # fast path: convert numpy -> tensor and copy into preallocated input
        try:
            if self.device.type == "cuda":
                cpu_tensor = torch.from_numpy(rgb_small).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
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
# Main: simplified display showing inpainted and relative depth
# ===================================================

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=640)
    ap.add_argument("--display-height", type=int, default=360)
    ap.add_argument("--inference-width", type=int, default=384)
    ap.add_argument("--inference-height", type=int, default=384)
    ap.add_argument("--backend", type=str, default=None, help='cv2 backend e.g. cv2.CAP_DSHOW or cv2.CAP_V4L2')
    ap.add_argument("--profile", action='store_true')
    ap.add_argument("--mask-before-image", type=str, default="./Undistort-and-Depth/MaskTrain2.png",
                    help="Path to an image mask file. Colored areas are KEPT before depth estimation.")
    args = ap.parse_args()

    try:
        backend = getattr(cv2, args.backend) if args.backend else None
        cam = CameraStream(src=args.camera_id, backend=backend)
    except Exception as e:
        print(f"[ERROR] Camera init failed: {e}")
        sys.exit(1)

    depth_est = DepthEstimator(
        display_size=(args.display_width, args.display_height),
        inference_size=(args.inference_width, args.inference_height),
    )

    cap_w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask_before_img = None
    mask_before_bool = None
    if args.mask_before_image and os.path.exists(args.mask_before_image):
        tmp = cv2.imread(args.mask_before_image, cv2.IMREAD_COLOR)
        if tmp is not None:
            mask_before_img = tmp
            mask_before_bool = _make_bool_mask_from_img(mask_before_img, cap_w, cap_h, invert=True)
            print(f"[INFO] Loaded mask-before image: colored = valid, black = masked")
        else:
            print(f"[WARN] Failed to read mask-before image at {args.mask_before_image}")

    frame_q = queue.Queue(maxsize=1)
    frame_worker = FrameWorker(cam, frame_q)

    fps_ema = 0.0
    alpha = 0.15
    last_time = time.time()

    mouse_x, mouse_y = args.display_width // 2, args.display_height // 2
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        if event == cv2.EVENT_MOUSEMOVE:
            if x < args.display_width:
                mouse_x, mouse_y = x, y
            else:
                mouse_x, mouse_y = x - args.display_width, y

    cv2.namedWindow("LGNet Filled (L)  |  Relative Depth (R)")
    cv2.setMouseCallback("LGNet Filled (L)  |  Relative Depth (R)", mouse_callback)

    lgg = None

    try:
        while True:
            try:
                frame_full = frame_q.get(timeout=1.0)
            except queue.Empty:
                time.sleep(0.005)
                continue

            # --- build masked_frame (inpainted) if mask provided and LGNet available ---
            try:
                if mask_before_bool is not None and _HAS_LGNET:
                    if lgg is None:
                        print("[INFO] Initializing LGNet generative fill...")
                        lgg = LGNetFiller()
                    mask_before_full = cv2.resize(mask_before_bool.astype(np.uint8), (frame_full.shape[1], frame_full.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    masked_frame = lgg.fill(frame_full, mask_before_full)
                else:
                    masked_frame = frame_full
            except Exception as e:
                print(f"[WARN] LGNet fill failed: {e} — falling back to raw frame for masked_frame.")
                masked_frame = frame_full

            display_masked = cv2.resize(masked_frame, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)

            # depth estimation on masked_frame (relative depth)
            try:
                depth_colored_full, depth_rel_full = depth_est.estimate_depth(masked_frame, profile=args.profile)
                depth_disp = cv2.resize(depth_colored_full, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"[WARN] Depth estimation error: {e}")
                depth_disp = np.full_like(display_masked, 128)
                depth_rel_full = np.full((frame_full.shape[0], frame_full.shape[1]), 0.5, dtype=np.float32)

            # Combine left (inpainted) and right (relative depth) views
            combined = np.hstack((display_masked, depth_disp))

            # FPS
            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_ema = alpha * inst_fps + (1 - alpha) * fps_ema
            fps_text = f"FPS: {fps_ema:.1f}"
            cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            # show relative depth value at cursor (raw relative units)
            cx_disp, cy_disp = int(np.clip(mouse_x, 0, args.display_width - 1)), int(np.clip(mouse_y, 0, args.display_height - 1))
            # map cursor to full-res depth map
            H_full, W_full = depth_rel_full.shape[:2]
            sx = W_full / float(args.display_width)
            sy = H_full / float(args.display_height)
            cx_full = int(np.clip(cx_disp * sx, 0, W_full - 1))
            cy_full = int(np.clip(cy_disp * sy, 0, H_full - 1))
            cursor_depth_rel = float(depth_rel_full[cy_full, cx_full])
            cv2.putText(combined, f"rel: {cursor_depth_rel:.4f}", (cx_disp + 10, cy_disp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("LGNet Filled (L)  |  Relative Depth (R)", combined)
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


if __name__ == '__main__':
    main()
