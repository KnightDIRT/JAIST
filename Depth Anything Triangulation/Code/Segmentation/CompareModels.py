"""
Run multiple segmentation models from segmentation_models.pytorch (SMP)
on a single image, save predictions, overlays, and rank models by a confidence score.

References:
- SMP README / docs used for API and preprocessing function. See: segmentation_models.pytorch README.
"""
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import matplotlib.pyplot as plt

# ----------- Utility functions -----------
def load_image(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    return img

def preprocess_image_pil(pil_img, preprocess_fn, tensor_size=None, device='cpu'):
    # Convert to numpy, apply encoder preprocessing (expects HWC with channels last)
    img_np = np.array(pil_img).astype(np.float32)
    input_arr = preprocess_fn(img_np)  # returns numpy, same HWC
    # To CHW and torch tensor
    input_arr = np.transpose(input_arr, (2, 0, 1))
    tensor = torch.from_numpy(input_arr).unsqueeze(0).float().to(device)
    return tensor

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def save_mask_and_overlay(original_pil, prob_map, out_prefix):
    """
    prob_map: numpy HxW values in [0..1]
    Saves:
     - {out_prefix}_prob.png  (grayscale probabilities)
     - {out_prefix}_mask.png  (binary threshold 0.5)
     - {out_prefix}_overlay.png (overlay on RGB)
    """
    H, W = prob_map.shape
    # Prob image
    prob_img = (prob_map * 255).astype(np.uint8)
    Image.fromarray(prob_img).save(out_prefix + "_prob.png")

    # Binary mask (threshold 0.5)
    bin_mask = (prob_map >= 0.5).astype(np.uint8) * 255
    Image.fromarray(bin_mask).save(out_prefix + "_mask.png")

    # Overlay: color the mask red with alpha
    orig = original_pil.resize((W, H)).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    # red layer
    red = np.zeros((H, W, 4), dtype=np.uint8)
    red[..., 0] = 255  # R
    red[..., 3] = (prob_map * 180).astype(np.uint8)  # alpha by probability
    red_img = Image.fromarray(red, mode="RGBA")
    blended = Image.alpha_composite(orig, red_img)
    blended.convert("RGB").save(out_prefix + "_overlay.png")

# ----------- Main runner -----------
def run_pipeline(image_path, outdir, device, resize_to=(512,512)):
    os.makedirs(outdir, exist_ok=True)
    img = load_image(image_path, target_size=resize_to)

    device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')

    # Choose a list of architectures to try (common and well-supported ones).
    # You can add or remove items here as you like; many SMP architectures exist.
    architectures = [
        "Unet",
        "UnetPlusPlus",
        "Linknet",
        "FPN",
        "PSPNet",
        "PAN",
        "DeepLabV3",
        "DeepLabV3Plus",
        "MAnet",
        "SegFormer",   # modern transformer-based (depends on available encoders)
        "DPT",         # if installed / available in this SMP build
        "UPerNet",
    ]

    # Encoder name to use for all models (common denominator). Change if you want to try many encoders.
    encoder_name = "resnet34"
    encoder_weights = "imagenet"  # use pretrained encoder weights if available

    # Obtain preprocessing func for chosen encoder (normalization used during encoder pretraining).
    preprocess_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)

    results = []

    for arch in architectures:
        try:
            # SMP exposes model classes as attributes, e.g. smp.Unet, smp.DeepLabV3Plus
            ModelClass = getattr(smp, arch)
        except AttributeError:
            print(f"[WARN] SMP does not expose architecture '{arch}' in this installation. Skipping.")
            continue

        print(f"[INFO] Instantiating {arch} (encoder={encoder_name}) ...")
        try:
            # We'll create a binary segmentation model: classes=1 + sigmoid activation later
            model = ModelClass(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            ).to(device)
            model.eval()
        except Exception as e:
            print(f"[WARN] Failed to create model {arch}: {e}. Skipping.")
            continue

        # Preprocess and forward
        try:
            input_tensor = preprocess_image_pil(img, preprocess_fn, tensor_size=resize_to, device=device)
            with torch.no_grad():
                pred = model(input_tensor)  # shape (1,1,H,W)
                # convert to numpy probability map in [0..1]
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                pred_np = pred.squeeze().cpu().numpy()
                # Many SMP models output raw logits (for classes=1). Apply sigmoid.
                prob_map = sigmoid(pred_np)
                # ensure shape HxW
                H, W = prob_map.shape
        except Exception as e:
            print(f"[WARN] Failed inference for {arch}: {e}. Skipping.")
            continue

        # Save outputs
        out_prefix = os.path.join(outdir, f"{arch}")
        save_mask_and_overlay(img, prob_map, out_prefix)

        # Compute simple summary metrics to help ranking:
        mean_prob = float(np.mean(prob_map))
        std_prob = float(np.std(prob_map))
        mean_abs_dev = float(np.mean(np.abs(prob_map - 0.5)))  # how far from 0.5 (higher = more confident)
        mask_area_frac = float(np.mean(prob_map >= 0.5))  # fraction of pixels predicted as foreground

        results.append({
            "arch": arch,
            "mean_prob": mean_prob,
            "std_prob": std_prob,
            "mean_abs_dev": mean_abs_dev,
            "mask_area_frac": mask_area_frac,
            "prob_path": out_prefix + "_prob.png",
            "mask_path": out_prefix + "_mask.png",
            "overlay_path": out_prefix + "_overlay.png",
        })

        print(f"  -> saved: {out_prefix}_prob.png, _mask.png, _overlay.png")
        print(f"  metrics: mean_prob={mean_prob:.4f}, std={std_prob:.4f}, mean_abs_dev={mean_abs_dev:.4f}, area_frac={mask_area_frac:.4f}")

    # Rank by chosen score (mean_abs_dev higher = more confident predictions)
    results_sorted = sorted(results, key=lambda r: r["mean_abs_dev"], reverse=True)

    # Print summary
    print("\n=== Models ranked by confidence score (mean_abs_dev) ===")
    for i, r in enumerate(results_sorted, 1):
        print(f"{i:02d}. {r['arch']:12s} | mean_abs_dev={r['mean_abs_dev']:.4f} | area_frac={r['mask_area_frac']:.4f} | mean_prob={r['mean_prob']:.4f}")

    # -------------------------------------------------------
    # 🖼️ Display overlays of all models in a grid
    # -------------------------------------------------------
    print("\nDisplaying overlay results for all models...")
    n = len(results_sorted)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

    for ax, res in zip(axes.flat, results_sorted):
        try:
            img_overlay = Image.open(res["overlay_path"])
            ax.imshow(img_overlay)
            title = f"{res['arch']}\n(conf={res['mean_abs_dev']:.3f})"
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        except Exception as e:
            ax.set_title(f"{res['arch']} (error)")
            ax.axis('off')
            print(f"[WARN] Could not open overlay for {res['arch']}: {e}")

    # Hide any unused axes
    for ax in axes.flat[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    # -------------------------------------------------------

    return results_sorted


# ----------- CLI -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SMP segmentation architectures on one image")
    parser.add_argument("--image", type=str, default="C:/Users/Torenia/OneDrive/Pictures/Camera Roll/WIN_20251015_15_32_16_Pro.jpg", help="Path to input image")
    parser.add_argument("--outdir", type=str, default="smp_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu")
    parser.add_argument("--size", type=int, nargs=2, default=[512,512], help="Resize image (W H)")
    args = parser.parse_args()

    # Quick check: required package notice
    try:
        import segmentation_models_pytorch as smp  # sanity
    except Exception as e:
        print("ERROR: segmentation_models_pytorch not installed. Install with:\n  pip install -U segmentation-models-pytorch torchvision timm")
        raise

    run_pipeline(args.image, args.outdir, args.device, resize_to=tuple(args.size))
