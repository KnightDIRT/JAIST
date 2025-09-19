"""
webcam_prr_realtime.py

Run “Perceptual Reflection Removal” (ceciliavision) on webcam frames in real time-ish.

Requirements:
  - TensorFlow (version <= 2.10, as per repo) installed, with GPU if possible.
  - cv2 (OpenCV) for webcam and display.
  - The perceptual-reflection-removal repo downloaded.
  - Pretrained model “task” checkpoint from repo’s `pre-trained` folder.
  - VGG-19 model file (imagenet-vgg-verydeep-19.mat) placed in VGG_Model/ folder of repo.
"""

import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

# Adjust these as needed:
REPO_ROOT = "C:/Users/Torenia/perceptual-reflection-removal"  # change to your path
PRETRAINED_TASK = "pre-trained"  # folder under REPO_ROOT where pretrained weights are stored
VGG_MODEL_PATH = os.path.join(REPO_ROOT, "VGG_Model", "imagenet-vgg-verydeep-19.mat")

# Add repo to PYTHONPATH so that main.py / model code can be imported
sys.path.append(REPO_ROOT)

# Import necessary parts from the repo
# Depending on how the repo is organized:
#   main.py defines a network, and code to restore checkpoints, etc.
from main import build_model, get_hypercolumns, preprocess_image, postprocess_transmission
# You will likely need to inspect main.py and see what functions/classes it defines;
# build_model might be the generator network that takes input image+hypercolumns and outputs transmission and reflection.

def load_model(task_dir):
    """Load the trained model from checkpoint for inference."""
    # This depends on how main.py is structured in the repo.
    # For example, main.py might accept arguments like `--task task_name` and `--is_training 0`
    # so you might wrap that logic here.
    model = build_model()  # stub; adjust depending on signature
    # Restore weights from checkpoint
    checkpoint_dir = os.path.join(REPO_ROOT, task_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    checkpoint.restore(latest).expect_partial()
    print("Model restored from", latest)
    return model

def process_frame(frame, model):
    """Process one frame: remove reflection."""
    # Preprocess:
    #   1. Convert BGR to RGB
    #   2. Normalize / resize as needed
    #   3. Compute hypercolumn features via VGG-19
    #   4. Concatenate image + hypercolumns
    #   5. Run model inference

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # The repo test code may resize images to some size (or work with arbitrary size).
    # Let's pick a fixed shorter side, or use original size but may need to pad/resize.
    # For speed, resize to say 512 px on the longer side, maintain aspect ratio.
    h0, w0 = rgb.shape[:2]
    max_dim = 512
    scale = max_dim / max(h0, w0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    input_rgb = cv2.resize(rgb, (new_w, new_h))

    # preprocess for the repo: convert to float [0,1], maybe remove gamma correction
    input_norm = input_rgb.astype(np.float32) / 255.0

    # Compute hypercolumn features
    # Probably use get_hypercolumns from the repo
    hypercols = get_hypercolumns(input_norm, VGG_MODEL_PATH)  # stub, adjust signature

    # Concatenate
    input_aug = np.concatenate([input_norm, hypercols], axis=-1)  # channel last

    # Add batch dimension
    input_tensor = tf.expand_dims(input_aug, axis=0)  # shape [1, H, W, C]

    # Run inference
    # The model returns two outputs: transmission and reflection (or maybe only transmission)
    T_pred, R_pred = model(input_tensor, training=False)

    # Postprocess T_pred: map back to image
    # Probably clip to [0,1], convert to uint8, resize back to original
    T_np = T_pred.numpy()[0]
    T_np = np.clip(T_np, 0.0, 1.0)
    T_np = (T_np * 255.0).astype(np.uint8)
    T_np = cv2.resize(T_np, (w0, h0))
    T_bgr = cv2.cvtColor(T_np, cv2.COLOR_RGB2BGR)

    return T_bgr

def main(camera_index=0):
    model = load_model(PRETRAINED_TASK)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera", camera_index)
        return

    # Optionally set resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit.")
    prev_time = time.time()
    fps_list = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read error")
                break

            t0 = time.time()
            out_frame = process_frame(frame, model)

            fps = 1.0 / (time.time() - t0 + 1e-6)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)

            # Display input and output side by side
            h, w = frame.shape[:2]
            out_small = cv2.resize(out_frame, (w//2, h//2))
            inp_small = cv2.resize(frame, (w//2, h//2))
            concat = cv2.hconcat([inp_small, out_small])
            cv2.imshow("Input | Transmission output", concat)
            cv2.setWindowTitle("Input | Transmission output", f"FPS: {avg_fps:.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(camera_index=0)
