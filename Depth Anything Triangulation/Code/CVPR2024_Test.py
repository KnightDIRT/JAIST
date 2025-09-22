import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
REPO_ROOT = r"C:/Users/Torenia/Reflection_RemoVal_CVPR2024"
CKPT_PATH = REPO_ROOT + "/ckpt/RD.pth"  # path to your checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE.startswith("cuda")
WEBCAM_IDX = 0
INFER_SIZE = (640, 360)  # width, height
# -------------------------

# add repo to path
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# import the models
from networks.NAFNet_arch import NAFNet_wDetHead
from networks.network_RefDet import RefDet

# build models
def build_models(device=DEVICE, half=USE_HALF):
    net = NAFNet_wDetHead(
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1,1,1,28],   # these match args in testing.py
        dec_blk_nums=[1,1,1,1],
        global_residual=False,
        drop_flag=False,
        drop_rate=0.4,
        concat=True,
        merge_manner=0
    )
    net_Det = RefDet(
        backbone='efficientnet-b3',
        proj_planes=16,
        pred_planes=32,
        use_pretrained=True,
        fix_backbone=False,
        has_se=False,
        num_of_layers=6,
        expansion=4
    )

    # load weights (⚠️ checkpoint may store both nets separately or together)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if "net" in ckpt:
        net.load_state_dict(ckpt["net"], strict=False)
    elif "state_dict" in ckpt:
        net.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        try:
            net.load_state_dict(ckpt, strict=False)
        except:
            print("⚠️ Checkpoint format not recognized, please inspect CKPT_PATH contents.")

    net.to(device).eval()
    net_Det.to(device).eval()
    if half:
        net.half()
        net_Det.half()
    return net, net_Det

def preprocess(frame, size):
    w, h = size
    frame = cv2.resize(frame, (w, h))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    return torch.from_numpy(img).unsqueeze(0)

def postprocess(tensor, size):
    tensor = tensor.clamp(0,1).cpu().float()
    img = tensor[0].numpy().transpose(1,2,0)
    img = (img*255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, size)

def run_realtime():
    net, net_Det = build_models()

    cap = cv2.VideoCapture(WEBCAM_IDX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret: break

        orig_size = (frame.shape[1], frame.shape[0])
        inp = preprocess(frame, INFER_SIZE).to(DEVICE)
        if USE_HALF: inp = inp.half()

        with torch.no_grad():
            sparse = net_Det(inp)
            out = net(inp, sparse)

        out_img = postprocess(out, orig_size)
        vis = np.concatenate([frame, out_img], axis=1)
        cv2.imshow("Reflection Removal (Left=Original, Right=Output)", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()
