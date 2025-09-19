"""
webcam_errnet_realtime.py
Run ERRNet on webcam frames in (near) real time.

Usage:
  python webcam_errnet_realtime.py --checkpoint checkpoints/errnet/errnet_060_00463920.pt --camera 0
"""

import argparse
import time
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import sys

# Make sure to run this script from repo root or adjust this path to the repo root
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))  # allow direct imports from the repo

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='./models/errnet_060_00463920.pt')
parser.add_argument('--camera', type=int, default=1)
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

device = torch.device(args.device)
ckpt_path = Path(args.checkpoint)
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

# --------------------------
# Try to import the repo's model builder
# --------------------------
model = None
try:
    # Many repos provide a function to create a model, e.g. `from models import create_model`
    # Try a few plausible import names; adjust if the repo uses different module names.
    from models import errnet as errnet_module  # <-- if the repo has models/errnet.py
    # If errnet_module has a factory function
    if hasattr(errnet_module, 'ERRNet') or hasattr(errnet_module, 'create_model'):
        # Example construction - you may need to check the exact constructor signature in the repo.
        if hasattr(errnet_module, 'create_model'):
            model = errnet_module.create_model()
        else:
            model = errnet_module.ERRNet()
except Exception:
    pass

# Generic fallback: try to infer and load state_dict directly (may require editing)
if model is None:
    print("Couldn't import a ready-made ERRNet builder from the repo. Using a generic wrapper. "
          "You may need to edit this script to construct the correct model class.")
    # TODO: Replace the following generic network with the actual ERRNet architecture
    # from the repository (e.g. import from model files). For now we make a small identity network
    # as placeholder — **you must replace this** with the real architecture for correct results.
    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Placeholder: identity (no change)
        def forward(self, x):
            return x

    model = DummyNet()

# Attempt to load checkpoint
print("Loading checkpoint:", ckpt_path)
ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
# Many checkpoints are saved as state_dicts, or contain 'state_dict' key; try both.
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
elif isinstance(ckpt, dict) and any(k.startswith('net') or k.startswith('model') for k in ckpt.keys()):
    # try direct
    state_dict = {k: v for k, v in ckpt.items()}
else:
    state_dict = ckpt

# Attempt to adapt keys (common pattern: remove 'module.' prefix from DataParallel)
new_state = {}
for k, v in state_dict.items():
    nk = k
    if nk.startswith('module.'):
        nk = nk[len('module.'):]
    new_state[nk] = v

try:
    model.load_state_dict(new_state, strict=False)
    print("Model state loaded (strict=False).")
except Exception as e:
    print("Warning: could not perfectly load state_dict into model:", e)
    print("You may need to construct the exact model class in the script (see TODO).")

model.to(device)
model.eval()

# transforms: convert BGR->RGB, resize, normalize - tune as repo expects
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),      # repo often uses 224x224 in README examples; adjust if needed
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inv_normalize = T.Normalize(
    mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
    std=[1/s for s in [0.229, 0.224, 0.225]]
)

cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

print("Press 'q' to quit.")
fps_meter = []
try:
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = transform(rgb).unsqueeze(0).to(device)  # shape [1,3,H,W]

        with torch.no_grad():
            out = model(inp)  # repo model should return the background image tensor
            # If model returns multiple outputs, adjust accordingly: e.g. out[0] might be reconstructed image
            if isinstance(out, (list, tuple)):
                out = out[0]

        # Denormalize & to numpy
        try:
            out = out.squeeze(0).cpu()
            out = inv_normalize(out)
            out = torch.clamp(out, 0., 1.)
            out_np = (out.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            # fallback if model returns same size as input or returns mask etc.
            out_bgr = frame.copy()
            print("Could not convert model output to image automatically:", e)

        # Show side-by-side
        combined = cv2.hconcat([cv2.resize(frame, (args.width//2, args.height//2)),
                                cv2.resize(out_bgr, (args.width//2, args.height//2))])
        cv2.imshow("ERRNet - input | reflection removed", combined)

        fps = 1.0 / (time.time() - t0 + 1e-6)
        fps_meter.append(fps)
        if len(fps_meter) > 30: fps_meter.pop(0)
        avg_fps = sum(fps_meter)/len(fps_meter)
        # print fps on window title
        cv2.setWindowTitle("ERRNet - input | reflection removed", f"ERRNet - {avg_fps:.1f} FPS")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
