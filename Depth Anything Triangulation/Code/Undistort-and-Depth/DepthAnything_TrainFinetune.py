import os
import kagglehub
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.io import loadmat

# ======================================================
# Load NYU Depth V2 (.mat) dataset
# ======================================================

def load_nyu_depth_v2_local(download_dir="datasets/nyu_depth_v2", limit=None):
    """
    Loads NYU Depth V2 dataset (CSV-based format) downloaded from KaggleHub.
    Returns a list of (PIL.Image RGB, PIL.Image depth) pairs.
    Expected structure (from soumikrakshit/nyu-depth-v2):
        nyu_data/data/nyu2_train.csv
        nyu_data/data/nyu2_train/
    """
    os.makedirs(download_dir, exist_ok=True)

    print("🔽 Downloading NYU Depth V2 dataset from KaggleHub...")
    dataset_path = kagglehub.dataset_download("soumikrakshit/nyu-depth-v2")
    print(f"✅ Dataset downloaded to: {dataset_path}")

    csv_path = os.path.join(dataset_path, "nyu_data", "data", "nyu2_train.csv")
    img_dir = os.path.join(dataset_path, "nyu_data", "data", "nyu2_train")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Missing image directory: {img_dir}")

    df = pd.read_csv(csv_path)
    print(f"📖 Loaded metadata for {len(df)} samples.")

    # Some CSVs include relative paths; ensure full path resolution
    pairs = []
    for i, row in df.iterrows():
        img_path = os.path.join(img_dir, row[0])
        depth_path = os.path.join(img_dir, row[1])

        if not os.path.exists(img_path) or not os.path.exists(depth_path):
            continue  # skip missing files

        image = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")
        pairs.append((image, depth))

        if limit and len(pairs) >= limit:
            break

    print(f"📸 Loaded {len(pairs)} RGB–Depth pairs successfully.")
    return pairs


# ======================================================
# Dataset class with blindspot overlay
# ======================================================

class BlindspotAugmentedDepthDataset(Dataset):
    def __init__(self, dataset_pairs, processor, blindspot_path):
        self.dataset_pairs = dataset_pairs
        self.processor = processor
        self.blindspot_16_9 = Image.open(blindspot_path).convert("RGBA")

    def __len__(self):
        return len(self.dataset_pairs)

    def __getitem__(self, idx):
        image, depth = self.dataset_pairs[idx]
        W, H = image.size

        # --- Apply blindspot overlay ---
        overlay = self.blindspot_16_9.resize((W, int(W * 9 / 16)), Image.BILINEAR)
        y = (H - overlay.height) // 2
        image.paste(overlay, (0, y), overlay)

        inputs = self.processor(images=image, return_tensors="pt")
        depth_tensor = torch.tensor(np.array(depth), dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return inputs["pixel_values"].squeeze(0), depth_tensor


# ======================================================
# Fine-tuning
# ======================================================

def train_blindspot_scaled():
    model_name = "LiheYoung/depth-anything-base-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)

    # Load NYU Depth dataset (optionally limit for testing)
    dataset_pairs = load_nyu_depth_v2_local(limit=500)
    blindspot_path = "./Undistort-and-Depth/MaskTrain.png"
    if not os.path.exists(blindspot_path):
        raise FileNotFoundError(f"Missing blindspot overlay at {blindspot_path}")

    train_dataset = BlindspotAugmentedDepthDataset(
        dataset_pairs=dataset_pairs,
        processor=processor,
        blindspot_path=blindspot_path
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.L1Loss()

    for epoch in range(3):
        model.train()
        running_loss = 0.0
        for i, (inputs, depth) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            inputs, depth = inputs.to(device), depth.to(device)

            outputs = model(pixel_values=inputs).predicted_depth
            outputs = torch.nn.functional.interpolate(
                outputs.unsqueeze(1),
                size=depth.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze(1)

            loss = loss_fn(outputs, depth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Step {i}: Loss = {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} finished | Avg Loss: {running_loss / len(train_loader):.4f}")

    save_dir = "./depth_anything_blindspot_finetuned"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"🎉 Fine-tuning complete! Model saved to: {save_dir}")


if __name__ == "__main__":
    train_blindspot_scaled()
