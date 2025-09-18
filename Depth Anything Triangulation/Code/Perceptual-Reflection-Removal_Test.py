"""
Real-time Depth Perceptual Reflection Removal
Based on: https://github.com/ceciliavision/perceptual-reflection-removal
Uses TensorFlow 1.x implementation from the repository
"""

import sys
import os
import cv2
import numpy as np
import time
import argparse
from pathlib import Path

# Add repository to path
repo_path = r"C:/Users/Torenia/perceptual-reflection-removal"
sys.path.append(repo_path)

# Import TensorFlow 1.x (disable v2 behavior)
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Fix for keras compatibility issues
import tensorflow.compat.v1.layers as layers

try:
    # Import from the repository - these are defined in main.py
    from discriminator import build_discriminator
    import scipy.io
    import scipy.stats as st
    import tf_slim as slim
except ImportError as e:
    print(f"Error importing components: {e}")
    print(f"Please ensure the repository is cloned at: {repo_path}")
    print("And install required dependencies: tensorflow==2.x, opencv-python, scipy, tf_slim")
    sys.exit(1)


class ReflectionRemovalProcessor:
    def __init__(self, model_dir="./pre-trained", device="0"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.sess = None
        self.input_placeholder = None
        self.transmission_layer = None
        self.reflection_layer = None
        self.hyper = True  # Use hypercolumn features
        self.channel = 64
        self.setup_model()
        
    def setup_model(self):
        """Setup the TensorFlow model from the repository"""
        # Set GPU device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device)
        
        # Load VGG19 parameters
        vgg_path = scipy.io.loadmat(os.path.join(repo_path, 'VGG_Model/imagenet-vgg-verydeep-19.mat'))
        print("[i] Loaded pre-trained vgg19 parameters")
        
        # Define helper functions from main.py
        def build_net(ntype, nin, nwb=None, name=None):
            if ntype == 'conv':
                return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name)+nwb[1])
            elif ntype == 'pool':
                return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        def get_weight_bias(vgg_layers, i):
            weights = vgg_layers[i][0][0][2][0][0]
            weights = tf.constant(weights)
            bias = vgg_layers[i][0][0][2][0][1]
            bias = tf.constant(np.reshape(bias, (bias.size)))
            return weights, bias

        def lrelu(x):
            return tf.maximum(x*0.2, x)

        def identity_initializer():
            def _initializer(shape, dtype=tf.float32, partition_info=None):
                array = np.zeros(shape, dtype=float)
                cx, cy = shape[0]//2, shape[1]//2
                for i in range(np.minimum(shape[2], shape[3])):
                    array[cx, cy, i, i] = 1
                return tf.constant(array, dtype=dtype)
            return _initializer

        def nm(x):
            """Simplified normalization function to avoid keras compatibility issues"""
            # Just use the weighted combination without batch norm for now
            w0 = tf.Variable(1.0, name='w0')  
            w1 = tf.Variable(0.0, name='w1')
            # Simple instance normalization alternative
            mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
            normalized = (x - mean) / tf.sqrt(variance + 1e-8)
            return w0*x + w1*normalized

        def build_vgg19(input, reuse=False):
            with tf.variable_scope("vgg19"):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                net = {}
                vgg_layers = vgg_path['layers'][0]
                net['input'] = input - np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
                net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
                net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
                net['pool1'] = build_net('pool', net['conv1_2'])
                net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
                net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
                net['pool2'] = build_net('pool', net['conv2_2'])
                net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
                net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
                net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
                net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
                net['pool3'] = build_net('pool', net['conv3_4'])
                net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
                net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
                net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
                net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
                net['pool4'] = build_net('pool', net['conv4_4'])
                net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
                net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
                return net

        def build_model(input):
            """Build the reflection removal model from main.py"""
            if self.hyper:
                print("[i] Hypercolumn ON, building hypercolumn features ... ")
                vgg19_features = build_vgg19(input[:, :, :, 0:3]*255.0)
                for layer_id in range(1, 6):
                    vgg19_f = vgg19_features["conv%d_2" % layer_id]
                    input = tf.concat([tf.image.resize_bilinear(
                        vgg19_f, (tf.shape(input)[1], tf.shape(input)[2]))/255.0, input], axis=3)
            else:
                vgg19_features = build_vgg19(input[:, :, :, 0:3]*255.0)
                for layer_id in range(1, 6):
                    vgg19_f = vgg19_features["conv%d_2" % layer_id]
                    input = tf.concat([tf.image.resize_bilinear(tf.zeros_like(
                        vgg19_f), (tf.shape(input)[1], tf.shape(input)[2]))/255.0, input], axis=3)
            
            # Use variable scope to avoid naming conflicts
            with tf.variable_scope("generator"):
                net = slim.conv2d(input, self.channel, [1, 1], rate=1, activation_fn=lrelu,
                                  normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv0')
                net = slim.conv2d(net, self.channel, [3, 3], rate=1, activation_fn=lrelu,
                                  normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv1')
                net = slim.conv2d(net, self.channel, [3, 3], rate=2, activation_fn=lrelu,
                                  normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv2')
                net = slim.conv2d(net, self.channel, [3, 3], rate=4, activation_fn=lrelu,
                                  normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv3')
                net = slim.conv2d(net, self.channel, [3, 3], rate=8, activation_fn=lrelu,
                                  normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv4')
                net = slim.conv2d(net, self.channel, [3, 3], rate=16, activation_fn=lrelu,
                                  normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv5')
                net = slim.conv2d(net, self.channel, [3, 3], rate=32, activation_fn=lrelu,
                                  normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv6')
                net = slim.conv2d(net, self.channel, [3, 3], rate=64, activation_fn=lrelu,
                                  normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv7')
                net = slim.conv2d(net, self.channel, [3, 3], rate=1, activation_fn=lrelu,
                                  normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv9')
                # output 6 channels --> 3 for transmission layer and 3 for reflection layer
                net = slim.conv2d(net, 3*2, [1, 1], rate=1, activation_fn=None, scope='g_conv_last')
            return net

        # Build the TensorFlow graph
        with tf.variable_scope(tf.get_variable_scope()):
            self.input_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            
            # Build the model
            network = build_model(self.input_placeholder)
            self.transmission_layer, self.reflection_layer = tf.split(
                network, num_or_size_splits=2, axis=3)
        
        # Create session and load model
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        # Load the trained model
        self.load_checkpoint()
        
        print(f"Model loaded successfully on GPU {self.device}")
    
    def load_checkpoint(self):
        """Load the pre-trained checkpoint"""
        try:
            # Construct the correct checkpoint path
            checkpoint_paths = [
                os.path.join("./models", self.model_dir),  # ./models/pre-trained
                os.path.join("models", self.model_dir),    # models/pre-trained
                str(self.model_dir),                       # pre-trained
                os.path.join(repo_path, "models", self.model_dir),  # repo/models/pre-trained
                Path("./models")
            ]
            
            checkpoint_loaded = False
            for ckpt_dir in checkpoint_paths:
                try:
                    print(f"Trying checkpoint directory: {ckpt_dir}")
                    
                    # Check if directory exists
                    if not os.path.exists(ckpt_dir):
                        print(f"  Directory does not exist: {ckpt_dir}")
                        continue
                    
                    # Check if checkpoint file exists
                    checkpoint_file = model_dir / "checkpoint"
                    if not os.path.exists(checkpoint_file):
                        print(f"  Checkpoint file not found: {checkpoint_file}")
                        continue
                    
                    print(f"  Found checkpoint file: {checkpoint_file}")
                    
                    # Use tf.train.get_checkpoint_state with the checkpoint file path
                    ckpt = tf.train.get_checkpoint_state(checkpoint_file)
                    if ckpt and ckpt.model_checkpoint_path:
                        print(f"  Found model checkpoint path: {ckpt.model_checkpoint_path}")
                        
                        # Check if the model files exist
                        if not os.path.exists(ckpt.model_checkpoint_path + ".index"):
                            print(f"  Model index file not found: {ckpt.model_checkpoint_path}.index")
                            continue
                        
                        # Load only generator variables (exclude discriminator for inference)
                        saver_restore = tf.train.Saver(
                            [var for var in tf.trainable_variables() if 'discriminator' not in var.name])
                        
                        print(f'Loading checkpoint: {ckpt.model_checkpoint_path}')
                        saver_restore.restore(self.sess, ckpt.model_checkpoint_path)
                        checkpoint_loaded = True
                        print("Checkpoint loaded successfully!")
                        break
                    else:
                        print(f"  Could not read checkpoint from: {checkpoint_file}")
                        
                except Exception as e:
                    print(f"  Error with path {ckpt_dir}: {e}")
                    continue
            
            if not checkpoint_loaded:
                print("Available checkpoint paths tried:")
                for path in checkpoint_paths:
                    print(f"  - {path}")
                raise FileNotFoundError("No valid checkpoint found in any of the expected locations")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Please ensure you have the pre-trained model in ./models/pre-trained/")
            print("The directory should contain files like: checkpoint, model.ckpt.index, model.ckpt.data-00000-of-00001, etc.")
            sys.exit(1)
    
    def preprocess_frame(self, frame):
        """Preprocess camera frame for the model"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_batch = np.expand_dims(normalized, axis=0)
        
        return input_batch
    
    def postprocess_output(self, output_tensor, original_shape):
        """Convert model output back to displayable format"""
        # Clip values to [0, 1] and convert to [0, 255]
        output = np.minimum(np.maximum(output_tensor, 0.0), 1.0) * 255.0
        
        # Remove batch dimension
        output = np.squeeze(output).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if len(output.shape) == 3 and output.shape[2] == 3:
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        else:
            output_bgr = output
        
        # Resize to original shape if needed
        if output_bgr.shape[:2] != original_shape[:2]:
            output_bgr = cv2.resize(output_bgr, (original_shape[1], original_shape[0]))
        
        return output_bgr
    
    def process_frame(self, frame):
        """Process a single frame through the reflection removal model"""
        # Preprocess
        input_batch = self.preprocess_frame(frame)
        
        # Run inference
        transmission_output, reflection_output = self.sess.run(
            [self.transmission_layer, self.reflection_layer], 
            feed_dict={self.input_placeholder: input_batch})
        
        # Postprocess
        transmission_frame = self.postprocess_output(transmission_output, frame.shape)
        reflection_frame = self.postprocess_output(reflection_output, frame.shape)
        
        return transmission_frame, reflection_frame
    
    def __del__(self):
        """Cleanup TensorFlow session"""
        if self.sess is not None:
            self.sess.close()


def main():
    parser = argparse.ArgumentParser(description="Real-time Reflection Removal using TensorFlow")
    parser.add_argument("--model_dir", type=str, default="pre-trained",
                       help="Model directory name inside ./models/ (default: pre-trained)")
    parser.add_argument("--camera_id", type=int, default=0,
                       help="Camera ID (default: 0)")
    parser.add_argument("--device", type=str, default="0",
                       help="GPU device ID (default: 0)")
    parser.add_argument("--fps_limit", type=int, default=15,
                       help="FPS limit for processing (default: 15)")
    parser.add_argument("--show_fps", action="store_true",
                       help="Display FPS counter")
    parser.add_argument("--show_reflection", action="store_true",
                       help="Also display the separated reflection layer")
    
    args = parser.parse_args()
    
    # Check if repository path exists
    if not os.path.exists(repo_path):
        print(f"Error: Repository path not found: {repo_path}")
        print("Please clone the repository or update the repo_path variable")
        return
    
    # Check if VGG model exists
    vgg_path = os.path.join(repo_path, 'VGG_Model/imagenet-vgg-verydeep-19.mat')
    if not os.path.exists(vgg_path):
        print(f"Error: VGG model not found at: {vgg_path}")
        print("Please download the VGG model as described in the repository README")
        return
    
    # Check if model directory exists
    model_path = os.path.join("./models", args.model_dir)
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found: {model_path}")
        print("Please ensure the pre-trained model is in ./models/pre-trained/")
        return
    
    # Initialize the processor
    print("Loading reflection removal model from repository...")
    print(f"Repository path: {repo_path}")
    processor = ReflectionRemovalProcessor(args.model_dir, args.device)
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, args.fps_limit)
    
    print("Starting real-time reflection removal...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frames") 
    print("  't' - Toggle between original and processed view")
    if args.show_reflection:
        print("  'r' - Toggle reflection layer visibility")
    
    # Performance tracking
    frame_times = []
    show_original = False
    show_reflection_layer = args.show_reflection
    frame_count = 0
    
    try:
        while True:
            start_time = time.time()
            
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Process frame
            if not show_original:
                try:
                    transmission_frame, reflection_frame = processor.process_frame(frame)
                    
                    if show_reflection_layer:
                        # Show side by side: transmission and reflection
                        h, w = transmission_frame.shape[:2]
                        combined = np.hstack([transmission_frame, reflection_frame])
                        display_frame = combined
                        window_title = "Transmission (Left) | Reflection (Right) - Press 't' to toggle"
                    else:
                        display_frame = transmission_frame
                        window_title = "Reflection Removed - Press 't' to toggle"
                        
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    display_frame = frame
                    window_title = "Original (Processing Error)"
            else:
                display_frame = frame
                window_title = "Original - Press 't' to toggle"
            
            # Add FPS counter if requested
            if args.show_fps and len(frame_times) > 0:
                fps = 1.0 / np.mean(frame_times[-30:])  # Average over last 30 frames
                cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(window_title, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                show_original = not show_original
                print(f"Switched to {'original' if show_original else 'processed'} view")
            elif key == ord('r') and args.show_reflection:
                show_reflection_layer = not show_reflection_layer
                print(f"Reflection layer {'visible' if show_reflection_layer else 'hidden'}")
            elif key == ord('s'):
                try:
                    if not show_original:
                        transmission_frame, reflection_frame = processor.process_frame(frame)
                        cv2.imwrite(f"transmission_{frame_count:04d}.jpg", transmission_frame)
                        cv2.imwrite(f"reflection_{frame_count:04d}.jpg", reflection_frame)
                        print(f"Saved transmission_{frame_count:04d}.jpg and reflection_{frame_count:04d}.jpg")
                    else:
                        cv2.imwrite(f"original_{frame_count:04d}.jpg", frame)
                        print(f"Saved original_{frame_count:04d}.jpg")
                except Exception as e:
                    print(f"Error saving frame: {e}")
            
            # Update performance tracking
            end_time = time.time()
            frame_time = end_time - start_time
            frame_times.append(frame_time)
            frame_count += 1
            
            # Limit FPS
            target_frame_time = 1.0 / args.fps_limit
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance statistics
        if frame_times:
            avg_fps = 1.0 / np.mean(frame_times)
            print(f"\nPerformance Statistics:")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()