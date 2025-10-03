#No longer supported

"""
zoedepth_webcam.py
Real-time webcam demo using ZoeDepth via torch.hub.

Usage:
    python zoedepth_webcam.py
Press 'q' to quit.
"""

import sys
import time
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.cm as cm

repo_path = r"C:/Users/Torenia/MiDaS"
sys.path.insert(0, repo_path)
repo_path = r"C:/Users/Torenia/ZoeDepth"
sys.path.insert(0, repo_path)

def colorize_depth(depth, cmap='viridis', vmin=None, vmax=None):
    """
    depth: 2D numpy array (float) - arbitrary scale
    returns: HxWx3 uint8 BGR image for OpenCV display
    """
    # Optionally set vmin/vmax for better contrast; if None use min/max
    if vmin is None: vmin = float(np.nanmin(depth))
    if vmax is None: vmax = float(np.nanmax(depth))
    # avoid division by zero
    if vmax - vmin == 0:
        vmax = vmin + 1e-6
    nd = (depth - vmin) / (vmax - vmin)
    nd = np.clip(nd, 0.0, 1.0)
    colormap = cm.get_cmap(cmap)
    colored = colormap(nd)[:, :, :3]  # H x W x 3 (RGB floats 0..1)
    colored_uint8 = (colored * 255).astype(np.uint8)
    # convert RGB->BGR for OpenCV
    bgr = cv2.cvtColor(colored_uint8, cv2.COLOR_RGB2BGR)
    return bgr

def main(camera_index=0, model_name="ZoeD_NK", use_cuda=True):
    # Device selection
    device = "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load ZoeDepth model via torch.hub (downloads if needed).
    # Available model names include "ZoeD_N", "ZoeD_K", "ZoeD_NK". See ZoeDepth README.
    repo = "isl-org/ZoeDepth"
    print("[INFO] Loading ZoeDepth model via torch.hub (this may take a moment)...")
    model = torch.hub.load(repo, model_name, pretrained=True).eval()
    model = model.to(device)
    print("[INFO] Model loaded.")

    # OpenCV capture
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}")

    # For FPS measurement
    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[WARN] Empty frame, retrying...")
                time.sleep(0.01)
                continue

            # Convert OpenCV BGR -> PIL RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # ZoeDepth provides infer_pil helper which returns numpy depth map
            # See ZoeDepth README for infer_pil usage.
            depth_np = model.infer_pil(pil_img)  # returns numpy by default

            # depth_np may be single-channel float (metric). Resize to original frame size if needed.
            if depth_np.shape != frame_bgr.shape[:2]:
                # infer_pil may already match input size; if not, resize depth map
                depth_np = cv2.resize(depth_np.astype('float32'),
                                      (frame_bgr.shape[1], frame_bgr.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)

            # Colorize for display
            depth_vis = colorize_depth(depth_np, cmap='magma')

            # Composite display: left=video, right=depth
            h, w = frame_bgr.shape[:2]
            combined = np.hstack((frame_bgr, depth_vis))

            # Draw FPS
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0
            cv2.putText(combined, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            cv2.imshow("ZoeDepth - Press 'q' to quit", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Exited.")

if __name__ == "__main__":
    # Example: main(camera_index=0, model_name="ZoeD_NK", use_cuda=True)
    main()
