"""
Real-time Depth Anything V2 with camera feed
Based on: https://github.com/DepthAnything/Depth-Anything-V2
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time
import argparse
from torchvision.transforms import Compose, Normalize
from PIL import Image
import sys
import os

# Add Depth Anything V2 repo to path
repo_path = r"C:/Users/Torenia/Depth-Anything-V2"
sys.path.append(repo_path)

# Import Depth Anything V2 components
# Note: You need to clone the repo and install dependencies first
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("Error: Could not import DepthAnythingV2. Please install the repository:")
    print("git clone https://github.com/DepthAnything/Depth-Anything-V2")
    print("cd Depth-Anything-V2")
    print("pip install -r requirements.txt")
    exit(1)


class DepthAnythingV2RealTime:
    def __init__(self, model_path, encoder='vitl', input_size=518):
        """
        Initialize Depth Anything V2 for real-time inference
        
        Args:
            model_path (str): Path to the model weights
            encoder (str): Model encoder type ('vits', 'vitb', 'vitl')
            input_size (int): Input image size
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model configurations for different encoder sizes
        model_configs = {
            'vits': {'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        
        if encoder not in model_configs:
            raise ValueError(f"Unsupported encoder: {encoder}. Choose from {list(model_configs.keys())}")
        
        config = model_configs[encoder]
        print(f"Initializing {encoder.upper()} model with features={config['features']}")
        
        # Initialize model with correct configuration
        self.model = DepthAnythingV2(
            encoder=encoder, 
            features=config['features'], 
            out_channels=config['out_channels']
        )
        
        # Load weights
        if Path(model_path).exists():
            print(f"Loading model weights from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            print(f"Warning: Model weights not found at {model_path}")
            print("Please download the weights from the official repository")
            print(f"Expected file: {model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.input_size = input_size
        
        # Image preprocessing
        self.transform = Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        h, w = image_rgb.shape[:2]
        image_resized = cv2.resize(image_rgb, (self.input_size, self.input_size))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_resized).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        image_tensor = self.transform(image_tensor)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, (h, w)
        
    def postprocess_depth(self, depth_map, original_size):
        """
        Postprocess depth map to original image size
        
        Args:
            depth_map (torch.Tensor): Raw depth map from model
            original_size (tuple): Original image size (h, w)
            
        Returns:
            np.ndarray: Processed depth map
        """
        h, w = original_size
        
        # Resize depth map to original size
        depth_resized = F.interpolate(
            depth_map.unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Convert to numpy
        depth_np = depth_resized.cpu().numpy()
        
        # Normalize for visualization
        depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        depth_colored = cv2.applyColorMap(
            (depth_normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_INFERNO
        )
        
        return depth_colored, depth_np
        
    def predict_depth(self, image):
        """
        Predict depth for a single image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            tuple: (colored_depth_map, raw_depth_values)
        """
        with torch.no_grad():
            # Preprocess
            input_tensor, original_size = self.preprocess_image(image)
            
            # Inference
            depth_map = self.model(input_tensor)
            
            # Postprocess
            depth_colored, depth_raw = self.postprocess_depth(depth_map, original_size)
            
        return depth_colored, depth_raw


def main():
    parser = argparse.ArgumentParser(description='Real-time Depth Anything V2')
    parser.add_argument('--encoder', choices=['vits', 'vitb', 'vitl'], default='vitb',
                       help='Model encoder type')
    parser.add_argument('--input-size', type=int, default=518,
                       help='Input image size')
    parser.add_argument('--camera-id', type=int, default=1,
                       help='Camera device ID')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Path to save output video')
    parser.add_argument('--fps-limit', type=int, default=30,
                       help='FPS limit for processing')
    
    args = parser.parse_args()
    
    # Model setup
    model_dir = Path("./models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"depth_anything_v2_{args.encoder}.pth"
    
    # Check if model exists
    if not model_path.exists():
        print(f"Model weights not found at {model_path}")
        print("Please download the weights from:")
        print("https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models")
        return
    
    # Initialize depth estimator
    depth_estimator = DepthAnythingV2RealTime(
        model_path=model_path,
        encoder=args.encoder,
        input_size=args.input_size
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return
        
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Video writer setup (optional)
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (1280, 480))
    
    print("Starting real-time depth estimation...")
    print("Press 'q' to quit, 's' to save current frame, 'r' to reset statistics")
    
    # Performance tracking
    frame_count = 0
    total_time = 0
    fps_limit_delay = 1.0 / args.fps_limit
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Predict depth
            inference_start = time.time()
            depth_colored, depth_raw = depth_estimator.predict_depth(frame)
            inference_time = time.time() - inference_start
            
            # Create side-by-side display
            display_frame = np.hstack([frame, depth_colored])
            
            # Add performance info
            fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Device: {depth_estimator.device}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Depth Anything V2 - Real Time', display_frame)
            
            # Save video frame
            if video_writer:
                video_writer.write(display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(f'depth_frame_{timestamp}.jpg', display_frame)
                cv2.imwrite(f'depth_raw_{timestamp}.png', (depth_raw * 65535).astype(np.uint16))
                print(f"Saved frame at timestamp {timestamp}")
            elif key == ord('r'):
                frame_count = 0
                total_time = 0
                print("Statistics reset")
            
            # Update statistics
            frame_count += 1
            total_time += time.time() - start_time
            
            if frame_count % 100 == 0:
                avg_fps = frame_count / total_time
                print(f"Processed {frame_count} frames, Average FPS: {avg_fps:.2f}")
            
            # FPS limiting
            elapsed = time.time() - start_time
            if elapsed < fps_limit_delay:
                time.sleep(fps_limit_delay - elapsed)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            avg_fps = frame_count / total_time
            print(f"\nFinal statistics:")
            print(f"Total frames processed: {frame_count}")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()