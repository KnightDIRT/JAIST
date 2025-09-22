#NO pre-trained given

"""
Real-time Neural Spline Fields (NSF) processing from camera feed
Adapted for Princeton Computational Imaging NSF repository
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
import argparse
from collections import deque
import os
import sys
from pathlib import Path

# Add the NSF repository to Python path (adjust path as needed)
NSF_PATH = "./NSF"  # Path to the cloned NSF repository
if os.path.exists(NSF_PATH):
    sys.path.append(NSF_PATH)

try:
    # Import NSF modules (these imports may need adjustment based on actual repo structure)
    from models.nsf_model import NSFModel  # Adjust import path
    from utils.data_utils import preprocess_burst
    from utils.inference_utils import run_inference
except ImportError as e:
    print(f"Warning: Could not import NSF modules: {e}")
    print("Please ensure the NSF repository is cloned and the import paths are correct")

class RealTimeNSF:
    def __init__(self, model_path=None, device='cuda', burst_size=8, input_size=(512, 512)):
        """
        Initialize real-time NSF processor
        
        Args:
            model_path: Path to pretrained NSF model
            device: Computing device ('cuda' or 'cpu')
            burst_size: Number of frames to accumulate for burst processing
            input_size: Input image size (width, height)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.burst_size = burst_size
        self.input_size = input_size
        
        # Frame buffer for burst processing
        self.frame_buffer = deque(maxlen=burst_size)
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        # Performance metrics
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
    def _load_model(self, model_path):
        """Load the NSF model"""
        try:
            # Initialize NSF model (adjust parameters based on actual model)
            model = NSFModel(
                input_channels=3,
                output_channels=3,
                hidden_dim=256,
                num_layers=8
            )
            
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from {model_path}")
            else:
                print("Warning: No model loaded, using random weights")
                
            model.to(self.device)
            model.eval()
            return model
            
        except NameError:
            print("NSF model not available, using placeholder")
            return None
    
    def preprocess_frame(self, frame):
        """Preprocess camera frame for NSF input"""
        # Resize frame
        frame_resized = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)  # HWC to CHW
        
        return frame_tensor
    
    def process_burst(self, burst_frames):
        """Process a burst of frames through NSF"""
        if self.model is None:
            # Fallback: return the latest frame if no model is loaded
            return burst_frames[-1]
        
        try:
            # Stack burst frames
            burst_tensor = torch.stack(burst_frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
            
            with torch.no_grad():
                # Run NSF inference
                if hasattr(self.model, 'forward'):
                    output = self.model(burst_tensor)
                else:
                    # Fallback processing if model structure is different
                    output = self._fallback_processing(burst_tensor)
                
                # Post-process output
                output = torch.clamp(output, 0, 1)
                output_frame = output[0].cpu()  # Remove batch dimension
                
            return output_frame
            
        except Exception as e:
            print(f"Error in NSF processing: {e}")
            return burst_frames[-1]  # Return latest frame on error
    
    def _fallback_processing(self, burst_tensor):
        """Fallback processing when NSF model is not available"""
        # Simple burst fusion using averaging
        fused = torch.mean(burst_tensor, dim=1)  # Average over time dimension
        
        # Apply some simple enhancement (sharpening)
        kernel = torch.tensor([[[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]], dtype=torch.float32)
        kernel = kernel.repeat(3, 1, 1, 1).to(burst_tensor.device) / 9.0
        
        enhanced = F.conv2d(fused, kernel, padding=1, groups=3)
        enhanced = 0.7 * fused + 0.3 * enhanced
        
        return enhanced
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def run_camera_processing(self, camera_id=0, display=True, save_output=False, output_dir="output"):
        """
        Main camera processing loop
        
        Args:
            camera_id: Camera device ID
            display: Whether to display the output
            save_output: Whether to save processed frames
            output_dir: Directory to save output frames
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera initialized: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {cap.get(cv2.CAP_PROP_FPS)}fps")
        print(f"Processing with NSF on device: {self.device}")
        print("Press 'q' to quit, 's' to save current frame")
        
        # Create output directory
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                
                # Add to buffer
                self.frame_buffer.append(processed_frame)
                
                # Process when buffer is full
                if len(self.frame_buffer) == self.burst_size:
                    # Run NSF processing
                    start_time = time.time()
                    result_tensor = self.process_burst(list(self.frame_buffer))
                    processing_time = (time.time() - start_time) * 1000  # ms
                    
                    # Convert result back to displayable format
                    result_np = result_tensor.permute(1, 2, 0).numpy()  # CHW to HWC
                    result_np = (result_np * 255).astype(np.uint8)
                    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                    
                    # Resize for display
                    display_frame = cv2.resize(result_bgr, (640, 480))
                    original_display = cv2.resize(frame, (640, 480))
                    
                    # Create side-by-side comparison
                    comparison = np.hstack((original_display, display_frame))
                    
                    # Add text overlay
                    self.update_fps()
                    cv2.putText(comparison, f"FPS: {self.current_fps}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(comparison, f"Processing: {processing_time:.1f}ms", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(comparison, "Original", (10, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(comparison, "NSF Processed", (650, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Display result
                    if display:
                        cv2.imshow('NSF Real-time Processing', comparison)
                    
                    # Save frame if requested
                    if save_output:
                        output_path = os.path.join(output_dir, f"nsf_frame_{frame_count:06d}.jpg")
                        cv2.imwrite(output_path, result_bgr)
                        frame_count += 1
                else:
                    # Show original frame while filling buffer
                    display_frame = cv2.resize(frame, (640, 480))
                    cv2.putText(display_frame, f"Filling buffer: {len(self.frame_buffer)}/{self.burst_size}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    if display:
                        cv2.imshow('NSF Real-time Processing', display_frame)
                
                # Handle keyboard input
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and len(self.frame_buffer) == self.burst_size:
                        # Save current processed frame
                        timestamp = int(time.time())
                        save_path = f"nsf_snapshot_{timestamp}.jpg"
                        cv2.imwrite(save_path, result_bgr)
                        print(f"Saved snapshot: {save_path}")
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if display:
                cv2.destroyAllWindows()
            print("Camera processing stopped")

def main():
    parser = argparse.ArgumentParser(description="Real-time NSF camera processing")
    parser.add_argument('--model_path', type=str, default=None, help='Path to NSF model checkpoint')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Processing device')
    parser.add_argument('--burst_size', type=int, default=8, help='Number of frames in burst')
    parser.add_argument('--input_size', type=int, nargs=2, default=[512, 512], help='Input image size (width height)')
    parser.add_argument('--no_display', action='store_true', help='Disable display window')
    parser.add_argument('--save_output', action='store_true', help='Save processed frames')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for saved frames')
    
    args = parser.parse_args()
    
    print("="*50)
    print("NSF Real-time Camera Processing")
    print("="*50)
    print(f"Model path: {args.model_path}")
    print(f"Camera ID: {args.camera_id}")
    print(f"Device: {args.device}")
    print(f"Burst size: {args.burst_size}")
    print(f"Input size: {args.input_size[0]}x{args.input_size[1]}")
    print("="*50)
    
    # Initialize NSF processor
    nsf_processor = RealTimeNSF(
        model_path=args.model_path,
        device=args.device,
        burst_size=args.burst_size,
        input_size=tuple(args.input_size)
    )
    
    # Run camera processing
    nsf_processor.run_camera_processing(
        camera_id=args.camera_id,
        display=not args.no_display,
        save_output=args.save_output,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()