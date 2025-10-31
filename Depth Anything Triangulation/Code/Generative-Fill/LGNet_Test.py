import sys
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

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
mask_path = "./Undistort-and-Depth/MaskTrain2.png"  # your static mask file
mask = Image.open(mask_path).convert("L")
mask = mask.resize((256, 256))
mask = Image.fromarray(255 - np.array(mask)) #invert maskq
mask_tensor = F.to_tensor(mask).unsqueeze(0)

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found!")
    exit()

print("Press 'q' to quit.")

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

    # Display result
    cv2.imshow("LGNet Real-time Generative Fill", comp_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
