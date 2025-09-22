#old compatibility issue

import os
import sys
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Adjust these as needed:
REPO_ROOT = "C:/Users/Torenia/perceptual-reflection-removal"  # change to your path
PRETRAINED_TASK = "pre-trained"  # folder under REPO_ROOT where pretrained weights are stored
VGG_MODEL_PATH = os.path.join(REPO_ROOT, "VGG_Model", "imagenet-vgg-verydeep-19.mat")

# Add repo to PYTHONPATH so that main.py / model code can be imported
sys.path.append(REPO_ROOT)

from main import build, saver, task, transmission_layer, reflection_layer, input


# Force GPU usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # only use as much memory as needed
config.log_device_placement = False     # set True if you want to see ops on GPU

# Start session
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# Load checkpoint
ckpt = tf.train.get_checkpoint_state(task)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Loaded checkpoint:", ckpt.model_checkpoint_path)
else:
    raise FileNotFoundError("No checkpoint found in {}".format(task))

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_frame = np.expand_dims(frame.astype(np.float32) / 255.0, axis=0)

    # Run through model
    pred_t, pred_r = sess.run(
        [transmission_layer, reflection_layer],
        feed_dict={input: input_frame}
    )

    # Postprocess transmission output
    output_img = np.clip(pred_t[0], 0, 1) * 255.0
    output_img = output_img.astype(np.uint8)

    # Show input vs output
    cv2.imshow("Input", frame)
    cv2.imshow("Reflection Removed", output_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
