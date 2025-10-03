#Problem with Xformers

"""
Realtime camera feed wrapper for running UniDepth (https://github.com/lpiccinelli-eth/UniDepth)

Usage:
    python realtime_unidepth_wrapper.py --name unidepth-v2-vitl14 --device cuda

This script:
- Opens webcam (or video file)
- Captures frames asynchronously
- Runs UniDepth inference (following demo.py logic)
- Displays original + colorized depth map
- Shows FPS overlay

Dependencies:
    pip install torch torchvision opencv-python numpy unidepth

"""

import argparse
import threading
import time
import queue
import sys
import os

import cv2
import numpy as np
import torch

repo_path = r"C:/Users/Torenia/UniDepth"
sys.path.insert(0, repo_path)

from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from unidepth.utils import colorize
from unidepth.utils.camera import Pinhole


# --------------------------- Capture thread ---------------------------
class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480, queue_size=4):
        self.src = src
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))

        self.stopped = False
        self.Q = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            if self.Q.full():
                try:
                    self.Q.get_nowait()
                except queue.Empty:
                    pass
            self.Q.put(frame)

    def read(self):
        try:
            return self.Q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass


# --------------------------- Model loader & inference ---------------------------
def load_unidepth_model(name: str, device: torch.device):
    if name.startswith("unidepth-v1"):
        model = UniDepthV1.from_pretrained(f"lpiccinelli/{name}")
    elif name.startswith("unidepth-v2"):
        model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
    else:
        raise ValueError(f"Unknown model name: {name}")

    # Set default interpolation mode like demo.py
    if isinstance(model, UniDepthV2):
        model.interpolation_mode = "bilinear"
        # model.resolution_level = 9  # optional

    model = model.to(device).eval()
    return model


# --------------------------- Main wrapper ---------------------------
def run_realtime(name: str, camera_id: int, width: int, height: int, device: str):
    dev = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
    print(f"[INFO] Using device: {dev}")

    model = load_unidepth_model(name, dev)

    cap = VideoCaptureAsync(src=camera_id, width=width, height=height, queue_size=3).start()

    fps_smooth = 0.0
    t_last = time.time()

    try:
        while True:
            frame = cap.read()
            if frame is None:
                time.sleep(0.005)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)

            # Dummy intrinsics (identity pinhole)
            H, W = rgb.shape[:2]
            fx = fy = max(H, W)
            cx, cy = W / 2, H / 2
            K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).unsqueeze(0).to(dev)
            camera = Pinhole(K=K)

            if isinstance(model, (UniDepthV2old, UniDepthV1)):
                camera = camera.K.squeeze(0)

            # ensure float32 to avoid fp16/xformers-only kernels
            rgb_torch = rgb_torch.float()

            # run inference without AMP/autocast so xformers' fp16 operators are not selected
            if dev.type == 'cuda':
                with torch.cuda.amp.autocast(enabled=False):
                    with torch.no_grad():
                        predictions = model.infer(rgb_torch, camera)
            else:
                with torch.no_grad():
                    predictions = model.infer(rgb_torch, camera)

            depth_pred = predictions["depth"].squeeze().cpu().numpy()
            depth_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")

            combined = np.hstack((frame, depth_col))

            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

            cv2.putText(combined, f"FPS: {fps_smooth:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(combined, f"Device: {dev}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(combined, f"Model: {name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('UniDepth - input | depth', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                outpath = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(outpath, combined)
                print(f"[INFO] Saved {outpath}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        cv2.destroyAllWindows()

# --------------------------- CLI ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Realtime UniDepth wrapper')
    p.add_argument('--name', type=str, default='unidepth-v2-vits14',
                   help='Model name: unidepth-v1-cnvnxtl | unidepth-v1-vitl14 | unidepth-v2-vits14 | unidepth-v2-vitb14 | unidepth-v2-vitl14')
    p.add_argument('--camera-id', type=int, default=0, help='OpenCV camera id or video file')
    p.add_argument('--width', type=int, default=640, help='Camera capture width')
    p.add_argument('--height', type=int, default=480, help='Camera capture height')
    p.add_argument('--device', type=str, default='cuda', help='torch device: cuda or cpu')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_realtime(args.name, args.camera_id, args.width, args.height, args.device)