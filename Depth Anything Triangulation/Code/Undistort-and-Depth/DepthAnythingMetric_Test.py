# Working but not well

import sys
import cv2
import torch
import numpy as np

# Add Depth Anything metric repo to path
repo_path = r"C:/Users/Torenia/Depth-Anything/metric_depth"
sys.path.append(repo_path)

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
qq

@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation (same logic as evaluate.py)."""
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            return pred
        elif isinstance(pred, (list, tuple)):
            return pred[-1]
        elif isinstance(pred, dict):
            if "metric_depth" in pred:
                return pred["metric_depth"]
            # fallback: take first tensor in dict
            for value in pred.values():
                if isinstance(value, torch.Tensor):
                    return value
            raise KeyError(f"Cannot find a tensor in dict keys: {list(pred.keys())}")
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    return 0.5 * (pred1 + pred2)


def load_depth_anything_model(pth_path, pt_path, model_name="zoedepth_nk", dataset="nyu"):
    """
    Load Depth Anything with encoder-decoder weights (.pth) 
    and metric depth head weights (.pt).
    """
    # base config
    config = get_config(model_name, "eval", dataset)

    # stop build_model from auto-loading from pretrained_resource
    config.pretrained_resource = ""  

    model = build_model(config)

    print(f"Loading encoder-decoder weights from {pth_path}")
    state_dict = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    print(f"Loading metric head weights from {pt_path}")
    metric_state = torch.load(pt_path, map_location="cpu")
    if "state_dict" in metric_state:
        metric_state = metric_state["state_dict"]
    model.load_state_dict(metric_state, strict=False)

    return model.cuda().eval()


def main(pth_path, pt_path):
    # --- load model ---
    model = load_depth_anything_model(pth_path, pt_path)

    cap = cv2.VideoCapture(0)  # webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # preprocess (H,W,C -> 1,C,H,W)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0).cuda()

        # inference
        pred = infer(model, img)
        depth = pred.squeeze().cpu().numpy()

        # normalize for visualization
        depth_vis = colorize(depth, vmin=depth.min(), vmax=depth.max())

        # resize to match original frame
        depth_vis_resized = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))

        # drop alpha channel if present
        if depth_vis_resized.shape[2] == 4:
            depth_vis_resized = depth_vis_resized[:, :, :3]

        # show side by side
        combined = np.hstack((frame, depth_vis_resized))
        cv2.imshow("Depth Anything Realtime", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    pth_path = "./models/depth_anything_vitl14.pth"
    pt_path = "./models/depth_anything_metric_depth_indoor.pt"
    main(pth_path, pt_path)
