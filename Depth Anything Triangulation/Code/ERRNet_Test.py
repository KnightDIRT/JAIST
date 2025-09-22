#Too much artifact

"""
webcam_errnet_realtime.py
Run ERRNet on webcam frames in (near) real time.
"""

import argparse
import time
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import sys

# Adjust these as needed:
REPO_ROOT = "C:/Users/Torenia/ERRNet"  # change to your path

# Add repo to PYTHONPATH so that main.py / model code can be imported
sys.path.append(REPO_ROOT)

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
# Import the actual ERRNet model
# --------------------------
from models.errnet_model import ERRNetModel

model = ERRNetModel()  # main ERRNet wrapper

print("Loading checkpoint:", ckpt_path)
ckpt = torch.load(str(ckpt_path), map_location=device)

# Figure out which key to load
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
elif isinstance(ckpt, dict) and 'icnn' in ckpt:
    state_dict = ckpt['icnn']
else:
    state_dict = ckpt

# Clean up any 'module.' prefixes
new_state = {}
for k, v in state_dict.items():
    nk = k[len('module.'):] if k.startswith('module.') else k
    new_state[nk] = v

try:
    if hasattr(model, 'net_i'):  # main generator net inside ERRNetModel
        model.net_i.load_state_dict(new_state, strict=False)
    else:
        model.load_state_dict(new_state, strict=False)
    print("Model state loaded (strict=False).")
except Exception as e:
    print("Warning: could not load state_dict into model:", e)

model.to(device)
model._eval()

# transforms: convert BGR->RGB, resize, normalize - tune as repo expects
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),      # adjust if repo uses different training size
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
            model.input = inp  # ERRNetModel expects self.input
            out = model.forward()  # returns transmission layer (reflection removed)

        # Denormalize & to numpy
        try:
            out = out.squeeze(0).cpu()
            out = inv_normalize(out)
            out = torch.clamp(out, 0., 1.)
            out_np = (out.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            out_bgr = frame.copy()
            print("Could not convert model output to image automatically:", e)

        # Show side-by-side
        combined = cv2.hconcat([cv2.resize(frame, (args.width//2, args.height//2)),
                                cv2.resize(out_bgr, (args.width//2, args.height//2))])
        cv2.imshow("ERRNet - input | reflection removed", combined)

        fps = 1.0 / (time.time() - t0 + 1e-6)
        fps_meter.append(fps)
        if len(fps_meter) > 30: 
            fps_meter.pop(0)
        avg_fps = sum(fps_meter)/len(fps_meter)
        cv2.setWindowTitle("ERRNet - input | reflection removed", f"ERRNet - {avg_fps:.1f} FPS")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()