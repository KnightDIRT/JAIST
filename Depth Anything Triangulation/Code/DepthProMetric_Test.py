# Running very accurately but very slowly

"""
Realtime webcam wrapper for Apple's DepthPro (ml-depth-pro).

Adapts the provided sample run.py to use live camera input instead of static images.

Usage:
    python realtime_depthpro_wrapper.py --camera 0

Dependencies:
    pip install opencv-python torch torchvision numpy matplotlib pillow tqdm
"""

import sys
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

repo_path = r"C:/Users/Torenia/ml-depth-pro/src"
sys.path.insert(0, repo_path)

from depth_pro.depth_pro import create_model_and_transforms


def get_torch_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def preprocess_frame(frame, transform):
    """Convert OpenCV BGR frame to PIL and apply DepthPro transform."""
    from PIL import Image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    return transform(img_pil), img


def postprocess_depth(prediction, frame_shape):
    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    inverse_depth = 1.0 / depth

    max_inv = min(inverse_depth.max(), 1 / 0.1)
    min_inv = max(1 / 250, inverse_depth.min())
    inv_norm = (inverse_depth - min_inv) / (max_inv - min_inv)

    inv_norm = cv2.resize(inv_norm, (frame_shape[1], frame_shape[0]))
    depth_uint8 = (inv_norm * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    return depth_colored, depth_uint8


def run_camera(args):
    device = get_torch_device()
    print(f"Using device: {device}")

    model, transform = create_model_and_transforms(
        device=device,
        precision=torch.half,
    )
    model.eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera}")

    fps_deque = deque(maxlen=30)

    window_name = "Realtime DepthPro"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            start = time.time()
            input_tensor, _ = preprocess_frame(frame, transform)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                prediction = model.infer(input_tensor)

            depth_colored, depth_uint8 = postprocess_depth(prediction, frame.shape)
            blended = cv2.addWeighted(frame, 0.5, depth_colored, 0.5, 0)

            elapsed = time.time() - start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_deque.append(fps)
            fps_avg = sum(fps_deque) / len(fps_deque)

            cv2.putText(blended, f"FPS: {fps_avg:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, blended)
            cv2.imshow("Depth Map", depth_uint8)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Realtime webcam inference with DepthPro")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()
    run_camera(args)


if __name__ == "__main__":
    main()
