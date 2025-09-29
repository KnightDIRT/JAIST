import sys
import cv2
import torch
import torch.nn as nn
import numpy as np

repo_path = r"C:/Users/Torenia/Depth-Anything"
sys.path.insert(0, repo_path)
repo_path = r"C:/Users/Torenia/ZoeDepth"
sys.path.insert(0, repo_path)
repo_path = r"C:/Users/Torenia/PatchRefiner"
sys.path.insert(0, repo_path)

from mmengine.config import Config
from estimator.models.builder import build_model
from estimator.models.patchfusion import PatchFusion

# --- SETTINGS ---
CONFIG_PATH = "C:/Users/Torenia/PatchRefiner/configs/patchrefiner_zoedepth/pr_cs.py"  # adjust to your config
CKPT_PATH   = "C:/Users/Torenia/PatchRefiner/work_dir/zoedepth/cs/ssi_7e-2/checkpoint_02.pth"  # your .pth file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(config_path, ckpt_path):
    cfg = Config.fromfile(config_path)

    if ".pth" in ckpt_path:
        model = build_model(cfg.model)
        state = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=True)
        else:
            model.load_state_dict(state, strict=True)
    else:
        # HuggingFace pretrained
        model = PatchFusion.from_pretrained(ckpt_path)

    model = model.to(DEVICE).eval()
    if cfg.get("convert_syncbn", False):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model

@torch.no_grad()
def infer(model, frame):
    # Convert BGR OpenCV image → RGB tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Forward pass
    pred = model(img)

    # Convert prediction to numpy
    if isinstance(pred, dict) and "pred_depth" in pred:
        depth = pred["pred_depth"]
    else:
        depth = pred
    depth = depth.squeeze().detach().cpu().numpy()

    # Normalize to 0–255 for display
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = (depth * 255).astype(np.uint8)
    return depth

def main():
    model = load_model(CONFIG_PATH, CKPT_PATH)

    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print(f"✅ Running on {DEVICE}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        depth_map = infer(model, frame)

        # Show both original and depth
        cv2.imshow("Webcam", frame)
        cv2.imshow("Depth", depth_map)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
