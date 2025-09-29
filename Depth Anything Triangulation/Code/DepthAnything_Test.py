# Very well but relative depth

"""
Real-time Depth-Anything camera feed script using Transformers
Captures webcam feed and generates depth maps in real-time
"""

import cv2
import numpy as np
import torch
from transformers import pipeline
from PIL import Image
import time
import argparse
import sys

class DepthAnythingCamera:
    def __init__(self, model_size="small", device=None):
        """
        Initialize the depth estimation camera
        
        Args:
            model_size: "small", "base", or "large" - larger models are more accurate but slower
            device: torch device to use, auto-detects if None
        """
        self.model_size = model_size
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using GPU acceleration")
            else:
                self.device = torch.device("cpu")
                print("Using CPU (consider GPU for better performance)")
        else:
            self.device = device
        
        # Initialize the depth estimation pipeline
        model_name = f"LiheYoung/depth-anything-{model_size}-hf"
        print(f"Loading model: {model_name}")
        
        try:
            self.depth_estimator = pipeline(
                "depth-estimation",
                model=model_name,
                device=0 if self.device.type == "cuda" else -1
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have internet connection and the transformers library installed")
            sys.exit(1)
        
        # Initialize camera
        self.cap = None
        self.fps_counter = 0
        self.start_time = time.time()
        
    def initialize_camera(self, camera_id):
        """Initialize the camera capture"""
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return False
            
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"Camera initialized successfully (ID: {camera_id})")
        return True
    
    def preprocess_frame(self, frame):
        """Convert OpenCV frame to PIL Image for the model"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        return pil_image
    
    def postprocess_depth(self, depth_result, original_shape):
        """Convert depth result to displayable format"""
        # Extract depth map from pipeline result
        depth = depth_result["depth"]
        
        # Convert to numpy array
        depth_array = np.array(depth)
        
        # Resize to match original frame size
        depth_resized = cv2.resize(depth_array, (original_shape[1], original_shape[0]))
        
        # Normalize to 0-255 range for display
        depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
        
        return depth_colored, depth_uint8
    
    def calculate_fps(self):
        """Calculate and return current FPS"""
        self.fps_counter += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1.0:  # Update FPS every second
            fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.start_time = time.time()
            return fps
        return None
    
    def run(self, display_mode="side_by_side", save_output=False, output_path="output.avi", camera_id=0):
        """
        Run the real-time depth estimation
        
        Args:
            display_mode: "side_by_side", "overlay", or "depth_only"
            save_output: whether to save the output video
            output_path: path for saved video
        """
        if not self.initialize_camera(camera_id):
            return
        
        print("Starting real-time depth estimation...")
        print("Press 'q' to quit, 's' to save current frame, 'm' to change display mode")
        
        # Video writer setup if saving
        fourcc = cv2.VideoWriter_fourcc(*'XVID') if save_output else None
        out = None
        
        current_fps = 0
        display_modes = ["side_by_side", "overlay", "depth_only"]
        mode_index = display_modes.index(display_mode)
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Preprocess frame
                pil_image = self.preprocess_frame(frame)
                
                # Generate depth map
                try:
                    depth_result = self.depth_estimator(pil_image)
                    depth_colored, depth_gray = self.postprocess_depth(depth_result, frame.shape[:2])
                    
                    # Calculate FPS
                    fps = self.calculate_fps()
                    if fps is not None:
                        current_fps = fps
                    
                    # Create display based on mode
                    current_mode = display_modes[mode_index]
                    
                    if current_mode == "side_by_side":
                        # Resize frames to fit side by side
                        h, w = frame.shape[:2]
                        frame_resized = cv2.resize(frame, (w//2, h//2))
                        depth_resized = cv2.resize(depth_colored, (w//2, h//2))
                        
                        # Combine frames
                        top_row = np.hstack([frame_resized, depth_resized])
                        display_frame = top_row
                        
                        # Add labels
                        cv2.putText(display_frame, "Original", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, "Depth", (w//2 + 10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    elif current_mode == "overlay":
                        # Blend original and depth
                        alpha = 0.6
                        display_frame = cv2.addWeighted(frame, alpha, depth_colored, 1-alpha, 0)
                    
                    else:  # depth_only
                        display_frame = depth_colored
                    
                    # Add FPS and info overlay
                    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                              (10, display_frame.shape[0] - 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Mode: {current_mode}", 
                              (10, display_frame.shape[0] - 35),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Model: {self.model_size}", 
                              (10, display_frame.shape[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Initialize video writer if needed
                    if save_output and out is None:
                        height, width = display_frame.shape[:2]
                        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
                    
                    # Save frame if recording
                    if out is not None:
                        out.write(display_frame)
                    
                    # Display frame
                    cv2.imshow('Depth-Anything Real-time', display_frame)
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    cv2.imshow('Depth-Anything Real-time', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    cv2.imwrite(f'depth_frame_{timestamp}.jpg', display_frame)
                    print(f"Frame saved as depth_frame_{timestamp}.jpg")
                elif key == ord('m'):
                    # Change display mode
                    mode_index = (mode_index + 1) % len(display_modes)
                    print(f"Display mode: {display_modes[mode_index]}")
        
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print("Camera feed stopped")

def main():
    parser = argparse.ArgumentParser(description="Real-time Depth-Anything camera feed")
    parser.add_argument("--model-size", choices=["small", "base", "large"], 
                       default="small", help="Model size (default: small)")
    parser.add_argument("--camera-id", type=int, default=0, 
                       help="Camera ID (default: 0)")
    parser.add_argument("--display-mode", choices=["side_by_side", "overlay", "depth_only"],
                       default="side_by_side", help="Display mode (default: side_by_side)")
    parser.add_argument("--save-output", action="store_true", 
                       help="Save output video")
    parser.add_argument("--output-path", default="depth_output.avi",
                       help="Output video path (default: depth_output.avi)")
    parser.add_argument("--device", choices=["cpu", "cuda"], 
                       help="Force device selection (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Convert device argument
    device = torch.device(args.device) if args.device else None
    
    # Create and run the depth camera
    depth_camera = DepthAnythingCamera(
        model_size=args.model_size,
        device=device
    )
    
    depth_camera.run(
        display_mode=args.display_mode,
        save_output=args.save_output,
        output_path=args.output_path,
        camera_id=args.camera_id
    )

if __name__ == "__main__":
    main()