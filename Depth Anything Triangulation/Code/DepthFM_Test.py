# Library conflict issue

# --- START: tiny pytree compatibility shim (paste at very top) ---
import builtins
try:
    import torch
    if not hasattr(torch.utils, "register_pytree_node") and hasattr(torch.utils, "_register_pytree_node"):
        torch.utils.register_pytree_node = torch.utils._register_pytree_node
except Exception:
    def __ensure_pytree_alias():
        import torch as _torch
        if not hasattr(_torch.utils, "register_pytree_node") and hasattr(_torch.utils, "_register_pytree_node"):
            _torch.utils.register_pytree_node = _torch.utils._register_pytree_node
    builtins.__ensure_pytree_alias = __ensure_pytree_alias
# --- END ---

import sys
import os
import cv2
import numpy as np

# Path to depth-fm repo
repo_path = r"C:/Users/Torenia/depth-fm"
sys.path.insert(0, repo_path)

# Import model utilities from depth-fm
from depthfm.dfm import DepthFM
from omegaconf import OmegaConf

import builtins, torch
if hasattr(builtins, "__ensure_pytree_alias"):
    builtins.__ensure_pytree_alias()


def load_depthfm(ckpt_path, device="cuda"):
    """
    Load Depth-FM model from checkpoint.
    """
    print(f"Loading Depth-FM from {ckpt_path}...")
    # Load default config from repo
    config_path = os.path.join(repo_path, "configs", "inference.yaml")
    cfg = OmegaConf.load(config_path)

    model = DepthFM(cfg.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


def run_webcam(ckpt_path="checkpoints/depthfm-v1.ckpt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_depthfm(ckpt_path, device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("✅ Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for speed (Depth-FM expects square inputs, often 384x384 or 512x512)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (384, 384))

        # To tensor
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            depth = model(img_tensor, num_steps=2, ensemble_size=4)

        # Convert depth map to displayable image
        depth_np = depth.squeeze().cpu().numpy()
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

        # Show side by side
        stacked = np.hstack((frame, cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))))
        cv2.imshow("Depth-FM Webcam (q to quit)", stacked)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam(ckpt_path=os.path.join(repo_path, "checkpoints", "depthfm-v1.ckpt"))
