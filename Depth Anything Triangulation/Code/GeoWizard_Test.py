# Problem with hugging face

import sys
import cv2
import torch
import logging
import numpy as np
from PIL import Image

repo_path = r"C:/Users/Torenia/GeoWizard/geowizard"
sys.path.insert(0, repo_path)

from models.geowizard_pipeline import DepthNormalEstimationPipeline
from utils.seed_all import seed_all

from diffusers import DDIMScheduler, AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

def main():
    logging.basicConfig(level=logging.INFO)

    # ---- Settings ----
    checkpoint_path = "lemonaddie/geowizard"   # HuggingFace model or local
    denoise_steps = 5                          # reduce for speed
    ensemble_size = 1                          # reduce for speed
    processing_res = 384                       # lower for real-time
    domain = "indoor"                          # "indoor" or "outdoor"
    color_map = "Spectral"
    seed = 42

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ---- Load model ----
    vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained(checkpoint_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(checkpoint_path, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(checkpoint_path, subfolder="feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")

    pipe = DepthNormalEstimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        unet=unet,
        scheduler=scheduler,
    )
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass

    pipe = pipe.to(device)
    seed_all(seed)

    # ---- Start camera ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return

    logging.info("Starting real-time GeoWizard inference. Press 'q' to quit.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert OpenCV BGR to PIL RGB
            input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Run through pipeline
            pipe_out = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=False,
                domain=domain,
                color_map=color_map,
                show_progress_bar=False,
            )

            # Convert results to OpenCV for display
            depth_colored = cv2.cvtColor(np.array(pipe_out.depth_colored), cv2.COLOR_RGB2BGR)
            normal_colored = cv2.cvtColor(np.array(pipe_out.normal_colored), cv2.COLOR_RGB2BGR)

            # Show side by side
            stacked = np.hstack([frame, depth_colored, normal_colored])
            cv2.imshow("GeoWizard - RGB | Depth | Normal", stacked)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
