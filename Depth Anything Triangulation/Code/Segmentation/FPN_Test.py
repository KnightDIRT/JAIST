"""
Real-time segmentation demo using FPN from segmentation_models_pytorch.
Press 'q' to quit.
"""

import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from PIL import Image

# ---------------- CONFIGURATION ----------------
MODEL_ARCH = "FPN"
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SIZE = (640, 480)  # (width, height)
USE_COLORMAP = True       # True = heatmap overlay, False = red binary overlay
# ------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess_frame(frame, preprocess_fn, device):
    """Preprocess OpenCV frame for model inference"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, FRAME_SIZE)
    img = img.astype(np.float32)
    img = preprocess_fn(img)
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
    return tensor

def overlay_mask(frame, prob_map, use_colormap=True):
    """Overlay segmentation mask (probabilities) onto frame"""
    prob_map = cv2.resize(prob_map, (frame.shape[1], frame.shape[0]))
    if use_colormap:
        # convert to heatmap (blue→red)
        heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    else:
        # binary red overlay
        mask = (prob_map > 0.5).astype(np.uint8)
        red = np.zeros_like(frame)
        red[:, :, 2] = 255  # red channel
        blended = cv2.addWeighted(frame, 1.0, red, 0.5, 0, mask=mask)
    return blended

def main():
    print(f"[INFO] Loading {MODEL_ARCH} model (encoder={ENCODER}) on {DEVICE}...")
    preprocess_fn = get_preprocessing_fn(ENCODER, pretrained=ENCODER_WEIGHTS)
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
    ).to(DEVICE).eval()

    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    print("[INFO] Press 'q' to quit.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            tensor = preprocess_frame(frame, preprocess_fn, DEVICE)

            # Inference
            pred = model(tensor)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            prob_map = sigmoid(pred.squeeze().cpu().numpy())

            # Overlay
            overlay = overlay_mask(frame, prob_map, USE_COLORMAP)

            # Display
            cv2.imshow("FPN Real-time Segmentation", overlay)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
