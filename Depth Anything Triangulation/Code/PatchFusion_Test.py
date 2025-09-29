# pretrained weight import dict failed

import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

repo_path = r"C:/Users/Torenia/Depth-Anything"
sys.path.insert(0, repo_path)
repo_path = r"C:/Users/Torenia/ZoeDepth"
sys.path.insert(0, repo_path)
repo_path = r"C:/Users/Torenia/PatchFusion"
sys.path.insert(0, repo_path)

from estimator.models.patchfusion import PatchFusion
from mmengine.config import Config, ConfigDict
from estimator.models.builder import build_model

# -------------------------------
# Load model
# -------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load config
cfg_path = 'C:/Users/Torenia/PatchFusion/configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py'
cfg = Config.fromfile(cfg_path)

# Update only paths (keep structure intact)
cfg.model.config.pretrain_model = [
    './models/PatchFusion/coarse_pretrain.pth',
    './models/PatchFusion/fine_pretrain.pth'
]
cfg.ckp_path = './models/PatchFusion/patchfusion.pth'

pf_config = cfg.model.config  # This is already a ConfigDict
print("PatchFusion init with:", type(pf_config))
print("is ConfigDict?", isinstance(pf_config, ConfigDict))
assert isinstance(pf_config, ConfigDict), type(pf_config)

model = PatchFusion(pf_config).to(DEVICE).eval()

# Load fusion checkpoint (expect some missing/unexpected key warnings)
ckpt = torch.load(cfg.ckp_path, map_location='cpu')
if hasattr(model, 'load_dict'):
    model.load_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt['model_state_dict'], strict=True)

# model_name = 'Zhyever/patchfusion_depth_anything_vits14'
# model = PatchFusion.from_pretrained(model_name).to(DEVICE).eval()

#model = model.to(DEVICE).eval()
# missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
# print("Missing keys:", missing)
# print("Unexpected keys:", unexpected)

# -------------------------------
# Configs from model
# -------------------------------
image_raw_shape = model.tile_cfg['image_raw_shape']
image_resizer = model.resizer

mode = 'r128'   # inference mode (try r256/r512 if GPU allows)
process_num = 4 # batch process size

# -------------------------------
# Camera inference loop
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Running real-time PatchFusion depth estimation. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert BGR → RGB, normalize
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    image = transforms.ToTensor()(np.asarray(image))  # [3, H, W]

    # Preprocess
    image_lr = image_resizer(image.unsqueeze(0)).float().to(DEVICE)
    image_hr = F.interpolate(
        image.unsqueeze(0),
        image_raw_shape,
        mode='bicubic',
        align_corners=True
    ).float().to(DEVICE)

    # Depth inference
    with torch.no_grad():
        depth_prediction, _ = model(
            mode='infer',
            cai_mode=mode,
            process_num=process_num,
            image_lr=image_lr,
            image_hr=image_hr
        )
        depth_prediction = F.interpolate(
            depth_prediction,
            image.shape[-2:]
        )[0, 0].detach().cpu().numpy()

    # Normalize depth → colormap
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_prediction, alpha=255.0 / np.max(depth_prediction)),
        cv2.COLORMAP_MAGMA
    )

    # Resize for display (to fit side by side)
    h, w = frame.shape[:2]
    depth_colormap = cv2.resize(depth_colormap, (w, h))
    stacked = np.hstack((frame, depth_colormap))

    cv2.imshow("PatchFusion Depth (Press 'q' to quit)", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
