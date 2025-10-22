#pre-trained mismatch

import cv2
import torch
import torch.nn.functional as F
import numpy as np

import sys
repo_path = r"C:/Users/Torenia/stdnet"
sys.path.insert(0, repo_path)

import models
from utils import state_from_training
from collections import deque
import time

# -------------------------------
# USER SETTINGS
# -------------------------------
MODEL_PATH = "C:/Users/Torenia/stdnet/checkpoints/stdneta_from_pretrained_DAVSOD.pth"  # your trained model path
NFRAMES = 8          # temporal clip length used by STDNet
FRAME_GAP = 4        # gap between frames
INPUT_SIZE = 256     # model input size
USE_GPU = torch.cuda.is_available()
# -------------------------------

def load_model():
    print("Loading STDNet model...")

    args = type('Args', (), {})()
    args.model = 'stdneta'  # for stdneta_from_pretrained_DAVSOD.pth
    args.train_config = 'sdnet'
    args.inference_config = 'baseline'
    args.nobn = False
    args.tid = ['cv']
    args.scales = [1.0]
    args.bn = True
    args.use_cuda = USE_GPU

    # ---- build models
    args.config = args.train_config
    model_training_time = getattr(models, args.model)(args)
    args.config = args.inference_config
    model_inference_time = getattr(models, args.model)(args)

    # ---- load checkpoint
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]

    # Strip "module." prefixes
    cleaned = {}
    for k, v in ckpt.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[new_k] = v

    # ---- try loading each key manually to avoid shape errors
    model_dict = model_training_time.state_dict()
    loaded_keys, skipped = 0, []
    for k, v in cleaned.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                model_dict[k] = v
                loaded_keys += 1
            else:
                skipped.append(k)
        else:
            skipped.append(k)
    model_training_time.load_state_dict(model_dict, strict=False)

    print(f"Loaded {loaded_keys} keys; skipped {len(skipped)} due to mismatch.")
    if skipped:
        print("Skipped keys (shape mismatch):", skipped[:5], "...")

    # ---- reparameterize if available
    if hasattr(model_training_time, "reparameterize"):
        model_training_time.reparameterize()
    elif hasattr(model_training_time, "module") and hasattr(model_training_time.module, "reparameterize"):
        model_training_time.module.reparameterize()
    else:
        print("[WARN] reparameterize() not found — continuing.")

    # --- Safe transfer from training to inference model
    try:
        state_from_training(model_training_time, model_inference_time)
    except KeyError as e:
        print(f"[WARN] state_from_training() missing key: {e}. Applying safe partial transfer...")

        # Manual safe copy of overlapping parameters
        t_state = model_training_time.state_dict()
        i_state = model_inference_time.state_dict()
        common = {k: v for k, v in t_state.items() if k in i_state and i_state[k].shape == v.shape}
        i_state.update(common)
        model_inference_time.load_state_dict(i_state, strict=False)
    print("Reparameterization complete.")

    model_inference_time.eval()
    if USE_GPU:
        model_inference_time = model_inference_time.cuda()

    print("✅ Model ready.")
    return model_inference_time

def preprocess_frame(frame):
    frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    frame = frame.unsqueeze(0)
    return frame

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    frame_buffer = deque(maxlen=NFRAMES)
    print("✅ Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = preprocess_frame(frame_rgb)
        frame_buffer.append(frame_tensor)

        if len(frame_buffer) == NFRAMES:
            # Stack as temporal sequence [1, Nframes, 3, H, W]
            clip = torch.stack(list(frame_buffer), dim=1)
            clip = clip.cuda() if USE_GPU else clip

            with torch.no_grad():
                sal_map = model(clip)
                sal_map = F.interpolate(
                    sal_map, size=(frame.shape[0], frame.shape[1]),
                    mode='bilinear', align_corners=False
                )
                sal_map = torch.squeeze(sal_map).cpu().numpy()
                sal_map = (sal_map - sal_map.min()) / (sal_map.max() + 1e-6)
                sal_map = (sal_map * 255).astype(np.uint8)
                sal_map = cv2.applyColorMap(sal_map, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(frame, 0.6, sal_map, 0.4, 0)
            cv2.imshow("STDNet Real-Time Saliency", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🟢 Done.")

if __name__ == "__main__":
    main()