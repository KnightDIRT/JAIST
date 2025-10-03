#MMCV failed

"""
Realtime wrapper for Metric3D.
- Loads model via torch.hub from the local repo directory containing hubconf.py
- Reads camera or video, runs per-frame depth inference, displays original + depth colormap,
  and optionally writes an output video.

Requirements:
  pip install torch torchvision opencv-python numpy

Usage examples:
  # webcam (device 0), use GPU if available, use pretrained weights (downloads once)
  python realtime_metric3d.py --repo-dir /path/to/metric3d_repo --model metric3d_vit_small --pretrain --cam 0

  # video file, save output
  python realtime_metric3d.py --repo-dir /path/to/metric3d_repo --model metric3d_vit_small --video input.mp4 --out out.mp4 --pretrain

Notes:
  - The script assumes the model expects ViT input sizing (default input_size=(616,1064)).
    You can change input_size to convnext variant size if you load a convnext model.
  - For metric scaling, you can pass camera intrinsics via --fx --fy --cx --cy (fx used for scaling).
"""

import argparse
import os
import time
import cv2
import numpy as np
import torch

# ------------------------------
# Utilities: preprocessing/post
# ------------------------------
def prepare_frame_for_model(rgb_bgr, input_size=(616, 1064), device='cuda'):
    """
    rgb_bgr: image in BGR (as read by cv2)
    returns: torch tensor on device with shape [1,3,H,W], pad_info, scale, rgb_origin
    """
    # convert BGR->RGB
    rgb_origin = rgb_bgr[:, :, ::-1].copy()
    h_orig, w_orig = rgb_origin.shape[:2]

    # keep ratio resize
    input_h, input_w = input_size
    scale = min(input_h / h_orig, input_w / w_orig)
    new_w = int(w_orig * scale)
    new_h = int(h_orig * scale)
    rgb_resized = cv2.resize(rgb_origin, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # scale intrinsics externally if provided by user (we return scale)
    # padding to input_size (use same padding values as hubconf example)
    padding_val = [123.675, 116.28, 103.53]
    pad_h = input_h - new_h
    pad_w = input_w - new_w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb_padded = cv2.copyMakeBorder(rgb_resized, pad_h_half, pad_h - pad_h_half,
                                    pad_w_half, pad_w - pad_w_half,
                                    cv2.BORDER_CONSTANT, value=padding_val)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    # normalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    rgb_tensor = torch.from_numpy(rgb_padded.transpose(2, 0, 1)).float()
    rgb_tensor = (rgb_tensor - torch.from_numpy(mean)[:, None, None]) / torch.from_numpy(std)[:, None, None]
    rgb_tensor = rgb_tensor.unsqueeze(0).to(device)

    return rgb_tensor, pad_info, scale, rgb_origin

def unpad_and_upsample(pred_depth, pad_info, target_size):
    """
    pred_depth: torch tensor [H, W]
    pad_info: [pad_h_half, pad_h_rem, pad_w_half, pad_w_rem]
    target_size: (h_orig, w_orig)
    returns: numpy depth HxW (float32) resized to target_size
    """
    ph0, ph1, pw0, pw1 = pad_info
    depth_unpad = pred_depth[ph0: pred_depth.shape[0]-ph1, pw0: pred_depth.shape[1]-pw1]
    # upsample to original size (target_size is (h,w))
    depth_up = torch.nn.functional.interpolate(depth_unpad[None, None, :, :], target_size, mode='bilinear', align_corners=False).squeeze()
    return depth_up.cpu().numpy().astype(np.float32)

def depth_to_colormap(depth_map, clip_min=0.1, clip_max=50.0, normalize=True):
    """
    Convert depth (numpy) to BGR colormap image for visualization.
    """
    d = depth_map.copy()
    if normalize:
        d = np.clip(d, clip_min, clip_max)
        d = (d - clip_min) / (clip_max - clip_min)
    d = (255 * (1.0 - d)).astype(np.uint8)  # invert for better visualization (near=bright)
    colored = cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)
    return colored

# ------------------------------
# Realtime loop
# ------------------------------
def main(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")

    # Load model via torch.hub from local repo dir
    repo_dir = args.repo_dir or '.'
    model_name = args.model or 'metric3d_vit_small'
    print(f"Loading model {model_name} from repo {repo_dir} (pretrain={args.pretrain}) ...")
    model = torch.hub.load(repo_dir, model_name, source='local', pretrain=args.pretrain)
    model.to(device).eval()
    print("Model loaded.")

    # choose input_size depending on model (vit default here)
    input_size = tuple(map(int, args.input_size.split(','))) if args.input_size else (616, 1064)
    print("Model input_size:", input_size)

    # open capture
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(int(args.cam))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    # output writer
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w * 2, h))  # side-by-side

    # intrinsics for metric scaling: use fx if provided, else default fx=1000 -> canonical scale = fx/1000
    fx = args.fx if args.fx else args.focal if args.focal else None
    if fx is None:
        print("Warning: no focal length specified. Metric scaling uses fx=1000 assumption (depth will be approximate).")
        fx = 1000.0

    try:
        with torch.no_grad():
            while True:
                st = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or cannot read frame.")
                    break

                # preprocess
                input_tensor, pad_info, scale, rgb_origin = prepare_frame_for_model(frame, input_size=input_size, device=device)

                # inference
                pred_depth_t, confidence, output_dict = model.inference({'input': input_tensor})
                # pred_depth_t has shape [1, H, W] (after model inference in hubconf example)
                pred_depth = pred_depth_t.squeeze().cpu()

                # unpad and upsample to original size
                orig_h, orig_w = rgb_origin.shape[:2]
                depth_map = unpad_and_upsample(pred_depth, pad_info, (orig_h, orig_w))

                # de-canonical transform to metric: canonical focal is 1000.0 in Metric3D example
                canonical_to_real_scale = (fx * scale) / 1000.0  # use scaled fx because we resized image by 'scale'
                depth_map = depth_map * canonical_to_real_scale
                depth_map = np.clip(depth_map, 0.0, args.clip_max)

                # visualize
                depth_vis = depth_to_colormap(depth_map, clip_min=args.clip_min, clip_max=args.clip_max)
                # combine side-by-side (RGB original is BGR)
                combined = np.concatenate([frame, depth_vis], axis=1)

                if writer:
                    writer.write(combined)

                cv2.imshow('Metric3D Realtime — left:rgb right:depth', combined)
                key = cv2.waitKey(1)
                # press q to quit
                if key & 0xFF == ord('q'):
                    break

                # FPS
                et = time.time() - st
                if args.show_fps:
                    print(f"Frame time: {et*1000:.1f} ms FPS: {1.0/et if et>0 else float('inf'):.2f}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-dir', type=str, default='C:/Users/Torenia/Metric3D',
                        help='Path to local Metric3D repo (directory that contains hubconf.py).')
    parser.add_argument('--model', type=str, default='metric3d_vit_small',
                        help='Name of the hubconf model function to load (e.g. metric3d_vit_small).')
    parser.add_argument('--pretrain', action='store_true', help='Load pretrained weights (downloads).')
    parser.add_argument('--cam', type=int, default=0, help='Camera device id (default 0).')
    parser.add_argument('--video', type=str, help='Path to video file instead of camera.')
    parser.add_argument('--out', type=str, help='Path to save output video (side-by-side).')
    parser.add_argument('--cpu', action='store_true', help='Force CPU use (disable CUDA).')
    parser.add_argument('--input-size', type=str, default=None,
                        help='Model input size H,W (e.g. "616,1064"). If not set uses default for ViT.')
    parser.add_argument('--fx', type=float, help='Camera focal length in pixels (fx). If not provided uses --focal or defaults.')
    parser.add_argument('--focal', type=float, help='Alias for fx for convenience.')
    parser.add_argument('--clip-min', type=float, default=0.1, help='Depth clip min for visualization (m).')
    parser.add_argument('--clip-max', type=float, default=50.0, help='Depth clip max for visualization (m).')
    parser.add_argument('--show-fps', action='store_true', help='Print approximate FPS per frame.')
    args = parser.parse_args()
    main(args)
