#Cant download pre-trained from chinese service

"""
Realtime webcam demo wrapper for CoRRN-Pytorch.
"""

import cv2
import torch
import numpy as np
import time
import argparse
import sys
from pathlib import Path

# ========== USER SETTINGS ==========
# Path to the repo root or ensure CoRRN.py is on PYTHONPATH
REPO_DIR = Path("./CoRRN-Pytorch")    # adjust if you cloned elsewhere
CHECKPOINT_PATH = "./checkpoint.pth"  # <- set this to your trained .pth/.pt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Name of the model class defined in CoRRN.py. Inspect CoRRN.py if uncertain.
MODEL_CLASS_NAME = "CoRRN"  # change to actual class name if different
# Webcam index and frame size to send to model (we resize before inference)
CAM_INDEX = 0
INFER_SIZE = (512, 512)  # (width, height) — tune for speed vs quality
# ===================================

# Add repo to path so we can import CoRRN.py directly
repo_path = str(REPO_DIR.resolve())
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# Try import the model definition
try:
    # CoRRN.py might define different names; import module and read attribute dynamically.
    import CoRRN as corrrn_module   # this is the file CoRRN.py in the repo
except Exception as e:
    print("Failed to import CoRRN module from", REPO_DIR, "\nError:", e)
    print("Make sure you cloned the repository and REPO_DIR points to it.")
    raise

# Get the model class
if hasattr(corrrn_module, MODEL_CLASS_NAME):
    ModelClass = getattr(corrrn_module, MODEL_CLASS_NAME)
else:
    # Try to find any nn.Module class in the module as a fallback
    import inspect
    candidates = []
    for name, obj in inspect.getmembers(corrrn_module):
        if inspect.isclass(obj):
            # naive heuristic: class inherits torch.nn.Module
            try:
                if issubclass(obj, torch.nn.Module):
                    candidates.append((name, obj))
            except Exception:
                pass
    if len(candidates) == 0:
        raise RuntimeError(f"No nn.Module subclass found in CoRRN.py; please inspect {REPO_DIR}/CoRRN.py and set MODEL_CLASS_NAME.")
    # pick the first candidate if user didn't specify
    print("MODEL_CLASS_NAME not found. Auto-selecting model class:", candidates[0][0])
    ModelClass = candidates[0][1]

# Instantiate model
model = ModelClass()  # if the model needs args, edit here
model.to(DEVICE)
model.eval()

# Load weights (state_dict or full checkpoint)
if CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # ckpt could be dict with 'state_dict' or straight state_dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    # Attempt to load, handling 'module.' prefix from DataParallel
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # try stripping "module." prefix
        new_state = {}
        for k, v in state.items():
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_k] = v
        model.load_state_dict(new_state)
    print("Loaded checkpoint from", CHECKPOINT_PATH)
else:
    print("Checkpoint not found at:", CHECKPOINT_PATH)
    print("Continuing with randomly initialized model (results will be meaningless).")

# Pre/post-processing helpers
import torchvision.transforms as T

to_tensor = T.ToTensor()  # scales [0,255] -> [0,1] and HWC->CHW
# If the repo uses a specific normalization, add it here. Commonly: mean/std for ImageNet
# norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
# You can enable normalization if required by the model:
# preprocess = T.Compose([to_tensor, norm])
preprocess = lambda im: to_tensor(im)

def run_inference_on_frame(frame_bgr):
    """
    Input: BGR frame (numpy uint8)
    Returns: output image (numpy uint8, same height/width as INFER_SIZE)
    """
    # Convert BGR->RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize for model (the model likely expects certain sizes)
    w, h = INFER_SIZE
    rgb_resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    # Preprocess to tensor
    inp = preprocess(rgb_resized).unsqueeze(0).to(DEVICE)  # shape 1xCxHxW, float
    with torch.no_grad():
        # The exact forward signature may differ; most nets use model(input)
        out = model(inp)
        # out might be a tensor, or a tuple; handle common cases
        if isinstance(out, (list, tuple)):
            out = out[0]
        # If model outputs a dict (e.g., {'B':..., 'G':...}) pick the first tensor
        if isinstance(out, dict):
            # pick first item
            out = next(iter(out.values()))
    # out: tensor 1xCxHxW
    if isinstance(out, torch.Tensor):
        # Move to CPU and convert to numpy
        out_np = out.squeeze(0).cpu().clamp(0, 1).numpy()  # C x H x W
        # If channels first and 3 channels -> transpose to HWC
        if out_np.shape[0] == 3:
            out_img = (np.transpose(out_np, (1, 2, 0)) * 255.0).astype(np.uint8)
            # Convert RGB back to BGR for OpenCV display
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        else:
            # single channel -> convert to grayscale BGR
            out_img = (out_np[0] * 255.0).astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
    else:
        raise RuntimeError("Model output type not recognized:", type(out))
    # Resize output back to original frame size if needed
    return out_img

def make_side_by_side(left_bgr, right_bgr):
    # ensure same heights
    h = left_bgr.shape[0]
    right_resized = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]))
    combined = np.concatenate([left_bgr, right_resized], axis=1)
    return combined

# Open camera
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")

# Determine original frame size (for resizing results back)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read from camera on startup")
orig_h, orig_w = frame.shape[:2]

print("Starting realtime webcam. Press 'q' to quit.")

# main loop
fps_smooth = None
try:
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, breaking.")
            break

        # Run inference
        try:
            out_img = run_inference_on_frame(frame)
        except Exception as e:
            # If inference fails, print and show original only
            print("Inference error:", e)
            out_img = np.zeros_like(frame)

        # Compose side-by-side display
        display = make_side_by_side(frame, out_img)

        # Draw FPS
        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        if fps_smooth is None:
            fps_smooth = fps
        else:
            fps_smooth = 0.85 * fps_smooth + 0.15 * fps
        cv2.putText(display, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("CoRRN - left:original | right:reflection_removed", display)

        # key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # optional: press 's' to save current output
        if key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"orig_{ts}.png", frame)
            cv2.imwrite(f"out_{ts}.png", out_img)
            print("Saved images.")
finally:
    cap.release()
    cv2.destroyAllWindows()
