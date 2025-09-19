"""
Real-time U2NET background removal for camera feed
Requires: opencv-python, torch, torchvision, numpy, pillow
"""

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import urllib.request
from pathlib import Path
import time

# U2NET Model Definition - Official Implementation
class REBNCONV(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = torch.nn.BatchNorm2d(out_ch)
        self.relu_s1 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

class RSU7(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=(hx5.size(2), hx5.size(3)), mode='bilinear')
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=(hx4.size(2), hx4.size(3)), mode='bilinear')
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size(2), hx3.size(3)), mode='bilinear')
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear')
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear')
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU6(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=(hx4.size(2), hx4.size(3)), mode='bilinear')
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size(2), hx3.size(3)), mode='bilinear')
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear')
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear')
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU5(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size(2), hx3.size(3)), mode='bilinear')
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear')
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear')
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU4(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear')
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear')
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU4F(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin

class U2NET(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        
        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        
        self.side1 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = torch.nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = torch.nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = torch.nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = torch.nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = torch.nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self, x):
        hx = x
        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        # stage 6
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, size=(hx5.size(2), hx5.size(3)), mode='bilinear')
        
        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=(hx4.size(2), hx4.size(3)), mode='bilinear')
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size(2), hx3.size(3)), mode='bilinear')
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear')
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear')
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # side output
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=(d1.size(2), d1.size(3)), mode='bilinear')
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=(d1.size(2), d1.size(3)), mode='bilinear')
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=(d1.size(2), d1.size(3)), mode='bilinear')
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=(d1.size(2), d1.size(3)), mode='bilinear')
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=(d1.size(2), d1.size(3)), mode='bilinear')
        
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

class U2NetCameraProcessor:
    def __init__(self, model_path=None, device=None):
        """
        Initialize U2NET camera processor
        
        Args:
            model_path: Path to U2NET model weights (will download if None)
            device: torch device ('cuda' or 'cpu', auto-detect if None)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = U2NET(3, 1)
        
        if model_path is None:
            model_path = self.download_model()
        
        # Load weights
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                print(f"Model loaded successfully from {model_path}")
            except RuntimeError as e:
                print(f"Error loading model weights: {e}")
                print("Using random weights instead...")
        else:
            print(f"Warning: Model file not found at {model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def download_model(self):
        """Download U2NET model weights if not present"""
        model_dir = Path("./models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "u2net.pth"
        
        if not model_path.exists():
            print("Downloading U2NET model weights...")
            url = "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
            try:
                # Note: This is a simplified download - in practice you might need 
                # to use gdown or handle Google Drive downloads differently
                print("Please download u2net.pth manually from:")
                print("https://github.com/xuebinqin/U-2-Net and place it in ./models/")
                print("Using random weights for demo...")
                # Save random weights for demo purposes
                torch.save(self.model.state_dict(), model_path)
            except Exception as e:
                print(f"Download failed: {e}")
                print("Using random weights for demo...")
                torch.save(self.model.state_dict(), model_path)
                
        return str(model_path)
    
    def preprocess_image(self, image):
        """Preprocess image for U2NET"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Transform and add batch dimension
        tensor_image = self.transform(pil_image).unsqueeze(0)
        return tensor_image.to(self.device)
    
    def postprocess_mask(self, mask, original_size):
        """Postprocess mask to original image size"""
        mask = mask.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (original_size[1], original_size[0]))
        return mask
    
    def process_frame(self, frame):
        """Process single frame and return mask"""
        original_size = frame.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess_image(frame)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            mask = outputs[0]  # Main output
        
        # Postprocess
        mask = self.postprocess_mask(mask, original_size)
        
        return mask
    
    def apply_background_removal(self, frame, mask, background=None):
        """Apply background removal using mask"""
        # Normalize mask to 0-1
        mask_norm = mask.astype(np.float32) / 255.0
        mask_norm = np.stack([mask_norm] * 3, axis=2)
        
        if background is None:
            # Use green screen background
            background = np.zeros_like(frame)
            background[:, :, 1] = 255  # Green
        else:
            # Resize background to match frame
            background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
        
        # Apply mask
        result = frame * mask_norm + background * (1 - mask_norm)
        return result.astype(np.uint8)

def main():
    """Main function to run real-time camera processing"""
    # Initialize processor
    processor = U2NetCameraProcessor()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Starting camera feed. Press 'q' to quit, 's' to save frame, 'b' to toggle background")
    
    # Background options
    backgrounds = [
        None,  # Green screen
        np.full((480, 640, 3), (255, 255, 255), dtype=np.uint8),  # White
        np.full((480, 640, 3), (0, 0, 0), dtype=np.uint8),  # Black
    ]
    bg_index = 0
    
    frame_count = 0
    
    # FPS calculation variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0.0
    fps_update_interval = 30  # Update FPS display every 30 frames
    
    # Processing time tracking
    process_times = []
    max_process_time_samples = 30
    
    while True:
        loop_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Process every few frames for performance
        if frame_count % 2 == 0:  # Process every 2nd frame
            try:
                process_start = time.time()
                
                # Get mask
                mask = processor.process_frame(frame)
                
                # Apply background removal
                current_bg = backgrounds[bg_index] if bg_index < len(backgrounds) else None
                result = processor.apply_background_removal(frame, mask, current_bg)
                
                process_time = time.time() - process_start
                process_times.append(process_time)
                if len(process_times) > max_process_time_samples:
                    process_times.pop(0)
                
                # Create display
                display_frame = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
                
            except Exception as e:
                print(f"Processing error: {e}")
                display_frame = frame
        
        frame_count += 1
        fps_counter += 1
        
        # Calculate and display FPS
        if fps_counter >= fps_update_interval:
            fps_end_time = time.time()
            fps_display = fps_counter / (fps_end_time - fps_start_time)
            fps_counter = 0
            fps_start_time = fps_end_time
        
        # Add FPS and processing time overlay
        if 'display_frame' in locals():
            # FPS text
            fps_text = f"FPS: {fps_display:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Processing time text (average of recent samples)
            if process_times:
                avg_process_time = sum(process_times) / len(process_times)
                process_text = f"Process: {avg_process_time*1000:.1f}ms"
                cv2.putText(display_frame, process_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        window_name = 'U2NET Real-time: Original | Mask | Result'
        cv2.imshow(window_name, display_frame if 'display_frame' in locals() else frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f'u2net_output_{timestamp}.jpg'
            cv2.imwrite(filename, display_frame if 'display_frame' in locals() else frame)
            print(f"Saved frame as {filename}")
        elif key == ord('b'):
            bg_index = (bg_index + 1) % len(backgrounds)
            bg_name = ['Green', 'White', 'Black'][bg_index]
            print(f"Background changed to: {bg_name}")
        
        # Calculate loop time for additional performance info
        loop_time = time.time() - loop_start_time
        
        # Optional: Print detailed performance info every 100 frames
        if frame_count % 100 == 0:
            if process_times:
                avg_process = sum(process_times) / len(process_times)
                print(f"Frame {frame_count}: FPS={fps_display:.1f}, "
                      f"Avg Process Time={avg_process*1000:.1f}ms, "
                      f"Loop Time={loop_time*1000:.1f}ms")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped")
    
    # Final performance summary
    if process_times:
        avg_process_time = sum(process_times) / len(process_times)
        print(f"\nPerformance Summary:")
        print(f"Average Processing Time: {avg_process_time*1000:.1f}ms")
        print(f"Estimated Processing FPS: {1/avg_process_time:.1f}")
        print(f"Total Frames Processed: {frame_count}")


if __name__ == "__main__":
    main()