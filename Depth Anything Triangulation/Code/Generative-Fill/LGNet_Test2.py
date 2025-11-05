import sys
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import time
import os

# --- Make sure LGNet repo is in path ---
repo_path = r"C:/Users/Torenia/LGNet"
sys.path.insert(0, repo_path)

from options.test_options import TestOptions
from models import create_model

def postprocess(img):
    img = (img + 1) / 2 * 255
    img = img.permute(0, 2, 3, 1)
    img = img.int().cpu().numpy().astype(np.uint8)
    return img[0]

# --- Required arguments for LGNet initialization ---
sys.argv += [
    '--dataroot', './dummy',
    '--name', 'celebahq_LGNet',
    '--model', 'pix2pixglg',
    '--netG1', 'unet_256',
    '--netG2', 'resnet_4blocks',
    '--netG3', 'unet256',
    '--input_nc', '4',
    '--direction', 'AtoB',
    '--no_dropout',
    '--gpu_ids', '0'
]

# --- Load model ---
opt = TestOptions().parse()
opt.num_threads = 0
opt.batch_size = 1
opt.no_flip = True
opt.serial_batches = True
opt.display_id = -1

model = create_model(opt)
model.setup(opt)
model.eval()

# --- Load mask ---
mask_path = "./Undistort-and-Depth/MaskTrain2.png"
mask = Image.open(mask_path).convert("L")
mask = mask.resize((256, 256))
mask = Image.fromarray(255 - np.array(mask))  # invert mask
mask_tensor = F.to_tensor(mask).unsqueeze(0)

# --- Create output directory for saves ---
os.makedirs("saved_frames", exist_ok=True)

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found!")
    exit()

print("Press 'q' to quit, 's' to save images.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb).resize((256, 256))
    frame_tensor = (F.to_tensor(frame_pil) * 2 - 1).unsqueeze(0)

    # Forward pass
    data = {'A': frame_tensor, 'B': mask_tensor, 'A_paths': ''}
    model.set_input(data)
    with torch.no_grad():
        model.forward()

    # Get results
    comp_img = postprocess(model.merged_images3)
    comp_bgr = cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR)

    # Resize for display (640x360 each)
    before_disp = cv2.resize(frame, (640, 360))
    after_disp = cv2.resize(comp_bgr, (640, 360))

    # Combine side-by-side for display
    combined = np.hstack((before_disp, after_disp))
    cv2.imshow("LGNet: Before (Left) | After (Right)", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save both images at 1280x720
        before_save = cv2.resize(frame, (1280, 720))
        after_save = cv2.resize(comp_bgr, (1280, 720))

        before_path = f"saved_frames/before_{timestamp}.png"
        after_path = f"saved_frames/after_{timestamp}.png"
        cv2.imwrite(before_path, before_save)
        cv2.imwrite(after_path, after_save)

        print(f"✅ Saved {before_path} and {after_path}")

cap.release()
cv2.destroyAllWindows()