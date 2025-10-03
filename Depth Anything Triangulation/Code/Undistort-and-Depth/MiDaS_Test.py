#No longer supported

import sys
import cv2
import torch
import numpy as np

# --- Import MiDaS ---
repo_path = r"C:/Users/Torenia/MiDaS"
sys.path.insert(0, repo_path)

from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

import torchvision.transforms as transforms


def run_midas_webcam():
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Load MiDaS model ---
    model_type = "DPT_Large"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"

    if model_type == "DPT_Large":
        model_path = repo_path + "/weights/dpt_large-midas-2f21e586.pt"
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384

    elif model_type == "DPT_Hybrid":
        model_path = repo_path + "/weights/dpt_hybrid-midas-501f0c75.pt"
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384

    else:  # MiDaS small
        model_path = repo_path + "/weights/midas_v21_small-70d6b9c8.pt"
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 256, 256

    model.eval()
    model.to(device)

    # --- Transform ---
    transform = transforms.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

    # --- Open webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        img_input = transform({"image": img})["image"]
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(device)

        # --- Run inference ---
        with torch.no_grad():
            prediction = model.forward(img_input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalize depth for visualization
        depth_min, depth_max = depth.min(), depth.max()
        depth_vis = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        # Show side-by-side
        combined = np.hstack((frame, depth_vis))
        cv2.imshow("MiDaS Depth Estimation", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_midas_webcam()
