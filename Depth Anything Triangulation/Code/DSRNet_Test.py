import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = r"C:/Users/Torenia/DSRNet"  # change to your path
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --------------------------- User-editable settings ---------------------------
WEIGHT_PATH = REPO_ROOT + "/weights/dsrnet_l_4000_epoch33.pt"
# -------------------------------------------------------------------------------

from models.dsrnet_model_sirs import DSRNetModel


class OptStub:
    def __init__(self, device):
        # network/training config
        self.inet = "dsrnet_l"     # backbone architecture
        self.loss = "losses"       # must match a module in models/ (adjust if needed)
        self.init_type = "kaiming"
        self.lr = 1e-4
        self.wd = 0.0
        self.resume = False
        self.no_verbose = True
        self.name = "realtime"
        self.weight_path = WEIGHT_PATH
        self.seed = 0

        # runtime config
        self.isTrain = False
        self.gpu_ids = [0] if torch.cuda.is_available() else []
        self.checkpoints_dir = "./checkpoints"


def preprocess_frame(frame, device, long_side=512):
    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    return tensor, frame


def postprocess_output(tensor):
    if isinstance(tensor, (tuple, list)):
        out = tensor[0]
    else:
        out = tensor
    out = out.detach().cpu().squeeze(0)
    out = torch.clamp(out, 0.0, 1.0)
    out = (out * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--long_side", type=int, default=512, help="max long side for resizing frames")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading model architecture...")
    opt = OptStub(device)
    net = DSRNetModel()
    net.isTrain = False
    net.initialize(opt)

    # ensure attributes exist
    net.data_name = None

    weight_path = Path(WEIGHT_PATH)
    if not weight_path.exists():
        print(f"ERROR: weight file not found at {weight_path}.")
        return

    print(f"Loading weights from {weight_path} ...")
    try:
        checkpoint = torch.load(str(weight_path), map_location=device)
        if 'weight' in checkpoint:
            sd = checkpoint['weight']
        elif 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint

        net.network.load_state_dict(sd, strict=False)
    except Exception as e:
        print("ERROR loading weights:", e)
        return

    net.network.to(device)
    net.vgg.to(device)
    net.network.eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {args.camera}")
        return

    print("Press 'q' to quit.")
    fps_time = time.time()
    frame_count = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            tensor, small_frame = preprocess_frame(frame, device, long_side=args.long_side)

            try:
                net.input = tensor.to(device)
                net.data_name = None
                out_t, out_r, out_rr = net.forward()
                out_img = postprocess_output(out_t)
            except Exception as e:
                print("Model forward failed. Error:\n", e)
                return

            if out_img.shape[:2] != small_frame.shape[:2]:
                out_img = cv2.resize(out_img, (small_frame.shape[1], small_frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            vis = np.concatenate([small_frame, out_img], axis=1)

            frame_count += 1
            if frame_count >= 10:
                t = time.time() - fps_time
                fps = frame_count / max(t, 1e-9)
                fps_time = time.time()
                frame_count = 0
            else:
                fps = None

            if fps is not None:
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("DSRNet Real-time (input | output)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
