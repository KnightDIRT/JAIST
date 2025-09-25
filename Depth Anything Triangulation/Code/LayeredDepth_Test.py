# No pre-trained given

"""
realtime_layered_depth_with_repo.py

Real-time webcam wrapper that calls a model inference function from a local repo,
and robustly visualizes N returned depth layers in a tiled grid.

Usage example:
    python realtime_layered_depth_with_repo.py \
        --repo_module LayeredDepth.my_inference_module \
        --repo_func predict_layers_from_image \
        --camera 0 \
        --width 640 --height 480

The inference function signature expected:
    def predict_fn(frame_bgr: np.ndarray) -> List[np.ndarray]
- frame_bgr: HxWx3 uint8 (OpenCV BGR)
- returns: list of HxW float32 arrays (depth layers). The arrays may be same size or different; the script will resize them to (width,height).
"""

import argparse
import importlib
import time
from collections import deque
import math
import os
from typing import Callable, List

import cv2
import numpy as np

# --------- Args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
parser.add_argument("--width", type=int, default=640, help="model input width (px)")
parser.add_argument("--height", type=int, default=480, help="model input height (px)")
parser.add_argument("--save_dir", type=str, default="estimations_live", help="where to save 16-bit pngs")
parser.add_argument("--save_every", type=int, default=30, help="save every N frames (set 1 for all)")
parser.add_argument("--repo_module", type=str, default="", help="Python module path for repo inference (e.g. LayeredDepth.infer)")
parser.add_argument("--repo_func", type=str, default="predict_layers", help="function name inside repo module to call")
parser.add_argument("--grid_cols", type=int, default=3, help="how many columns in visualization grid (including original image)")
parser.add_argument("--max_layers", type=int, default=6, help="max layers to display/save (script will cap returned layers to this)")
parser.add_argument("--cpu", action="store_true", help="force CPU if module uses torch")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# --------- Utilities ----------
def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """Convert HxW float32 depth to a BGR uint8 colormap (same HxW)."""
    if depth is None:
        return None
    d = np.nan_to_num(depth.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    # robust min/max using percentiles to avoid outliers dominating color scale
    if d.size == 0:
        return np.zeros((args.height, args.width, 3), dtype=np.uint8)
    minv = float(np.percentile(d, 2))
    maxv = float(np.percentile(d, 98))
    if maxv - minv < 1e-6:
        maxv = minv + 1e-6
    norm = np.clip((d - minv) / (maxv - minv), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
    return colored

def overlay_colormap_on_image(image_bgr: np.ndarray, cmap_bgr: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Blend a colormap over the image. Both must be HxWx3 uint8."""
    return cv2.addWeighted(image_bgr, 1.0 - alpha, cmap_bgr, alpha, 0)

def pad_or_crop_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image to exactly target_h x target_w using interpolation (keeps aspect by scaling)."""
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized

# --------- Repo inference loader ----------
def load_repo_predict_function(module_path: str, func_name: str) -> Callable[[np.ndarray], List[np.ndarray]]:
    """
    Dynamically import module_path and return function func_name.
    The function must accept a single HxWx3 uint8 BGR frame and return a list of HxW float32 layers.
    """
    if not module_path:
        raise ImportError("Empty module_path")
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}") from e
    if not hasattr(module, func_name):
        raise ImportError(f"Module '{module_path}' does not contain function '{func_name}'. Available: {dir(module)}")
    func = getattr(module, func_name)
    if not callable(func):
        raise ImportError(f"Object '{func_name}' in module '{module_path}' is not callable.")
    return func

# --------- Demo fallback inference (keeps script usable if user hasn't provided module) ----------
def predict_layers_demo(frame_bgr: np.ndarray) -> List[np.ndarray]:
    """Simple demo: returns two synthetic layers for visualization/testing."""
    H, W = args.height, args.width
    # vertical gradient
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    layer1 = np.tile(y, (1, W)).astype(np.float32)
    xv = np.linspace(-1, 1, W)[None, :].astype(np.float32)
    yv = np.linspace(-1, 1, H)[:, None].astype(np.float32)
    rr = np.sqrt(xv**2 + yv**2)
    layer2 = (1.0 - np.clip(rr, 0.0, 1.0)).astype(np.float32)
    return [layer1, layer2][: args.max_layers]

# --------- Build final predict function (either repo or demo) ----------
predict_fn = None
if args.repo_module:
    try:
        predict_fn = load_repo_predict_function(args.repo_module, args.repo_func)
        print(f"Using repo predict function: {args.repo_module}.{args.repo_func}")
    except Exception as e:
        print(f"Warning: could not load repo predict function: {e}")
        print("Falling back to demo predict function.")
        predict_fn = predict_layers_demo
else:
    print("No repo module provided; using demo predict function.")
    predict_fn = predict_layers_demo

# --------- Visualization grid builder ----------
def build_tiles_for_display(original_img: np.ndarray, depth_layers: List[np.ndarray], cols: int) -> np.ndarray:
    """
    Create a tiled grid image including the original image + colored layer overlays.
    - original_img: HxWx3 uint8
    - depth_layers: list of HxW float32 arrays (will be resized to original size)
    - cols: number of columns in the grid (>=1)
    Returns a single HxW_gridx3 uint8 image.
    """
    H, W = original_img.shape[:2]
    items = []
    # First item always the original
    items.append(original_img.copy())
    # For each layer, create overlay tile (image blended with colormap), and label it.
    for i, L in enumerate(depth_layers[: args.max_layers]):
        if L is None:
            tile = np.zeros_like(original_img)
        else:
            # ensure L is numpy float and resize to HxW
            L_np = np.array(L, dtype=np.float32)
            if L_np.shape != (H, W):
                L_np = cv2.resize(L_np, (W, H), interpolation=cv2.INTER_LINEAR)
            cmap = depth_to_colormap(L_np)
            overlay = overlay_colormap_on_image(original_img, cmap, alpha=0.6)
            cv2.putText(overlay, f"Layer {i+1}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            tile = overlay
        items.append(tile)
    # If no layers returned, show just the original
    if len(items) == 1:
        return items[0]

    # Normalize all tiles to exactly HxW
    tiles = [pad_or_crop_to(it, H, W) for it in items]

    # Compute grid rows
    n_items = len(tiles)
    cols = max(1, cols)
    rows = math.ceil(n_items / cols)

    # Build row by row using hstack
    row_imgs = []
    for r in range(rows):
        start = r * cols
        end = min(start + cols, n_items)
        row_tiles = tiles[start:end]
        # if last row is incomplete, pad with black tiles
        if len(row_tiles) < cols:
            for _ in range(cols - len(row_tiles)):
                row_tiles.append(np.zeros_like(tiles[0]))
        row_img = np.hstack(row_tiles)
        row_imgs.append(row_img)

    tiled = np.vstack(row_imgs)
    return tiled

# --------- Save function ----------
def save_layers(frame_idx: int, depth_layers: List[np.ndarray], save_dir: str):
    """Save each layer as 16-bit PNG in save_dir with naming frameIdx_layerIdx.png"""
    for li, L in enumerate(depth_layers[: args.max_layers]):
        if L is None:
            continue
        LL = np.array(L, dtype=np.float32)
        # map min->0, max->65535
        LL = np.nan_to_num(LL, nan=0.0, posinf=0.0, neginf=0.0)
        minv = float(np.min(LL))
        maxv = float(np.max(LL))
        if maxv - minv < 1e-9:
            scaled = np.zeros_like(LL, dtype=np.uint16)
        else:
            scaled = ((LL - minv) / (maxv - minv) * 65535.0).astype(np.uint16)
        outname = os.path.join(save_dir, f"{frame_idx}_{li+1}.png")
        cv2.imwrite(outname, scaled)

# --------- Main loop ----------
def main_loop():
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: cannot open camera", args.camera)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame_idx = 0
    fps_deque = deque(maxlen=30)
    last_time = time.time()

    print("Starting webcam. Press 'q' to quit, 's' to save current frames immediately.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: empty frame.")
            time.sleep(0.05)
            continue

        # Resize frame sent to model to args.width x args.height (predict_fn expects that)
        model_input = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

        # Call repo/demonstration predict function
        try:
            layers = predict_fn(model_input)
            if layers is None:
                layers = []
        except Exception as e:
            # If model throws, print once and keep running with empty list
            print(f"Model inference error (frame {frame_idx}): {e}")
            layers = []

        # Cap layers length
        layers = list(layers)[: args.max_layers]

        # Build tiled visualization: original + overlay tiles
        tiled = build_tiles_for_display(model_input, layers, cols=args.grid_cols)

        # FPS computation
        now = time.time()
        fps_deque.append(1.0 / max(now - last_time, 1e-9))
        last_time = now
        fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0.0
        cv2.putText(tiled, f"FPS: {fps:.1f}", (10, tiled.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("LayeredDepth Live (repo-backed)", tiled)

        # Periodic save
        if args.save_every > 0 and (frame_idx % args.save_every == 0):
            save_layers(frame_idx, layers, args.save_dir)

        ch = cv2.waitKey(1) & 0xFF
        if ch == ord("q"):
            break
        if ch == ord("s"):
            save_layers(frame_idx, layers, args.save_dir)
            print("Saved current layers.")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
