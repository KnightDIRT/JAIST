"""
Detectron2 Real-time Camera Feed Script
This script captures video from camera and runs real-time object detection using Detectron2.
"""

import cv2
import torch
import numpy as np
import time
import argparse
import warnings
from collections import deque

# Suppress the torch.meshgrid warning
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release.*")

def setup_detectron2(model_type="mask_rcnn"):
    """Initialize Detectron2 model and configuration."""
    try:
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        
        print(f"Setting up Detectron2 with {model_type} for segmentation...")
        
        # Configure the model
        cfg = get_cfg()
        
        # Use GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.DEVICE = device
        print(f"Using device: {device}")
        
        # Choose segmentation model based on user preference
        if model_type == "mask_rcnn":
            config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        elif model_type == "mask_rcnn_R101":
            config_file = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        elif model_type == "mask_rcnn_X101":
            config_file = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        else:
            config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        
        # Set confidence threshold for segmentation
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        # Create predictor
        predictor = DefaultPredictor(cfg)
        
        # Get metadata for visualization
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        
        print("✓ Detectron2 setup complete!")
        return predictor, metadata, cfg
        
    except Exception as e:
        print(f"✗ Failed to setup Detectron2: {e}")
        return None, None, None

def setup_camera(camera_id=0, width=640, height=480):
    """Initialize camera capture."""
    print(f"Setting up camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {camera_id}")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"✓ Camera {camera_id} initialized ({width}x{height})")
    return cap

def draw_fps(frame, fps, position=(10, 30)):
    """Draw FPS counter on frame."""
    # Ensure frame is contiguous numpy array
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    cv2.putText(frame, f"FPS: {fps:.1f}", position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def draw_detection_info(frame, predictions):
    """Draw detection count and segmentation info."""
    # Ensure frame is contiguous numpy array
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    instances = predictions["instances"]
    num_detections = len(instances)
    
    # Count objects with masks
    num_with_masks = 0
    if instances.has("pred_masks"):
        num_with_masks = len(instances.pred_masks)
    
    cv2.putText(frame, f"Objects: {num_detections}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Segmented: {num_with_masks}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

class FPSCounter:
    """Simple FPS counter."""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
    
    def update(self):
        self.times.append(time.time())
    
    def get_fps(self):
        if len(self.times) < 2:
            return 0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])

def main():
    parser = argparse.ArgumentParser(description="Detectron2 Real-time Camera Feed")
    parser.add_argument("--camera", type=int, default=1, help="Camera ID (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--model", type=str, default="mask_rcnn", 
                       choices=["mask_rcnn", "mask_rcnn_X101", "mask_rcnn_R101"],
                       help="Segmentation model to use (default: mask_rcnn)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--no-display", action="store_true", 
                       help="Don't display video (for headless systems)")
    
    args = parser.parse_args()
    
    print("Starting Detectron2 Real-time Detection...")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Setup Detectron2
    predictor, metadata, cfg = setup_detectron2(args.model)
    if predictor is None:
        return
    
    # Update threshold if specified
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    
    # Setup camera
    cap = setup_camera(args.camera, args.width, args.height)
    if cap is None:
        return
    
    # Initialize FPS counter and visualizer
    fps_counter = FPSCounter()
    frame_count = 0
    
    try:
        from detectron2.utils.visualizer import Visualizer
        
        print("\n" + "="*50)
        print("Real-time segmentation started!")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'p' - Pause/Resume")
        print("="*50 + "\n")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Update FPS counter
                fps_counter.update()
                frame_count += 1
                
                # Run detection
                start_time = time.time()
                predictions = predictor(frame)
                inference_time = time.time() - start_time
                
                # Visualize results
                v = Visualizer(frame[:, :, ::-1], metadata, scale=1.0)
                out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
                result_frame = out.get_image()[:, :, ::-1]
                
                # Ensure the frame is a proper numpy array and contiguous
                result_frame = np.ascontiguousarray(result_frame, dtype=np.uint8)
                
                # Add FPS and info overlays
                current_fps = fps_counter.get_fps()
                draw_fps(result_frame, current_fps)
                draw_detection_info(result_frame, predictions)
                
                # Add inference time - ensure frame is still contiguous
                if not result_frame.flags['C_CONTIGUOUS']:
                    result_frame = np.ascontiguousarray(result_frame)
                cv2.putText(result_frame, f"Inference: {inference_time*1000:.1f}ms", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame (if not headless)
            if not args.no_display:
                if not paused:
                    cv2.imshow('Detectron2 Real-time Segmentation', result_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s') and not paused:
                    # Save current frame
                    filename = f"segmentation_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"Segmentation frame saved as {filename}")
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
            
            # Print stats every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames, Current FPS: {fps_counter.get_fps():.1f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error during detection: {e}")
    
    finally:
        # Cleanup
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("Camera released and windows closed")

def test_camera():
    """Test if camera is accessible."""
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Camera not accessible")
        print("Try:")
        print("  - Check if camera is connected")
        print("  - Try different camera IDs (1, 2, etc.)")
        print("  - Close other applications using the camera")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("✗ Failed to read from camera")
        return False
    
    print(f"✓ Camera working - Frame shape: {frame.shape}")
    return True

if __name__ == "__main__":
    # Quick camera test
    if not test_camera():
        print("Please fix camera issues before running the main script")
        exit(1)
    
    # Run main detection loop
    main()