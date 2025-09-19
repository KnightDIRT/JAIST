import cv2
import tensorflow as tf
import numpy as np
import sys

# Adjust these as needed:
REPO_ROOT = "C:/Users/Torenia/perceptual-reflection-removal"  # change to your path

# Add repo to PYTHONPATH so that main.py / model code can be imported
sys.path.append(REPO_ROOT)

from main import build_model, preprocess_image, postprocess_transmission

# ---- Setup TF session ----
sess = tf.Session()
input_placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 3])
network = build_model(input_placeholder)  # your model
sess.run(tf.global_variables_initializer())

# ---- Restore checkpoint ----
task = "C:/Users/Torenia/perceptual-reflection-removal/pre-trained"
ckpt = tf.train.get_checkpoint_state(task)
if ckpt:
    saver_restore = tf.train.Saver([var for var in tf.trainable_variables() if 'discriminator' not in var.name])
    saver_restore.restore(sess, ckpt.model_checkpoint_path)
    print("[i] Loaded checkpoint:", ckpt.model_checkpoint_path)
else:
    raise FileNotFoundError("No checkpoint found in " + task)

# ---- OpenCV webcam loop ----
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    # Preprocess
    input_batch = preprocess_image(frame_rgb)
    input_batch = np.expand_dims(input_batch, axis=0)  # add batch dim

    # Run model
    output = sess.run(network, feed_dict={input_placeholder: input_batch})

    # Postprocess
    output_img = postprocess_transmission(output[0])
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    output_img = cv2.resize(output_img, (w, h))

    # Show side by side
    combined = np.hstack((frame, output_img))
    cv2.imshow("Reflection Removal - Original | Output", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sess.close()
