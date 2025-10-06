#12fps
import cv2
import numpy as np
import torch
import json
import time
import argparse
import sys
import threading
import queue
from transformers import AutoProcessor, AutoModelForDepthEstimation
from PIL import Image
import torch.nn.functional as F


class DepthAnythingCamera:
    def __init__(self, model_size="small", device=None, calib_file="fisheye_calibration_data.json",
                 use_gpu_undistort=False, inference_size=(384, 384)):
        """Initialize real-time depth estimation + fisheye undistortion"""
        self.model_size = model_size
        self.use_gpu_undistort = use_gpu_undistort
        self.inference_size = inference_size

        # Load calibration data
        try:
            with open(calib_file, "r") as f:
                calib_data = json.load(f)
            self.K = np.array(calib_data["K"])
            self.D = np.array(calib_data["D"])
            self.image_size = tuple(calib_data["image_size"])
        except Exception as e:
            print(f"❌ Failed to load calibration file: {e}")
            sys.exit(1)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("🚀 Using GPU")
            else:
                self.device = torch.device("cpu")
                print("⚠️ Using CPU (consider GPU for better performance)")
        else:
            self.device = device

        # Enable PyTorch performance optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Load model directly (no pipeline overhead)
        model_name = f"LiheYoung/depth-anything-{model_size}-hf"
        print(f"Loading model: {model_name}")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
            self.model.half()  # FP16 for faster inference
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

        # Precompute undistortion maps
        new_K = self.K.copy()
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, self.image_size, cv2.CV_32FC1
        )

        # Preconvert undistortion maps to GPU grid if enabled
        if self.use_gpu_undistort and self.device.type == "cuda":
            print("Using GPU-accelerated undistortion")
            self.grid = self._create_gpu_remap_grid(self.map1, self.map2, self.image_size)
        else:
            self.grid = None

        self.cap = None
        self.fps_counter = 0
        self.start_time = time.time()
        self.frame_queue = queue.Queue(maxsize=2)

    def _create_gpu_remap_grid(self, map1, map2, image_size):
        """Convert OpenCV maps to normalized PyTorch grid for F.grid_sample"""
        map1_t = torch.from_numpy(map1)
        map2_t = torch.from_numpy(map2)
        grid = torch.stack((map1_t, map2_t), dim=-1)
        grid[..., 0] = (2.0 * grid[..., 0] / (image_size[0] - 1)) - 1.0
        grid[..., 1] = (2.0 * grid[..., 1] / (image_size[1] - 1)) - 1.0
        return grid.unsqueeze(0).to(self.device)

    def _gpu_undistort(self, frame):
        """Undistort using torch grid_sample"""
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        with torch.no_grad():
            undistorted_t = F.grid_sample(frame_t, self.grid, align_corners=True)
        undistorted = (undistorted_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return undistorted

    def initialize_camera(self, camera_id):
        """Open camera and start capture thread"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {camera_id}")
            return False

        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"✅ Camera initialized (ID: {camera_id})")

        threading.Thread(target=self._capture_loop, daemon=True).start()
        return True

    def _capture_loop(self):
        """Run camera capture in background thread"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def preprocess_frame(self, frame):
        """Undistort + convert to PIL"""
        frame_resized = cv2.resize(frame, self.image_size)
        if self.use_gpu_undistort and self.grid is not None:
            undistorted = self._gpu_undistort(frame_resized)
        else:
            undistorted = cv2.remap(frame_resized, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb).resize(self.inference_size)
        return pil_image, undistorted

    def postprocess_depth(self, depth, original_shape):
        """Convert depth tensor to visualization"""
        depth_resized = cv2.resize(depth, (original_shape[1], original_shape[0]))
        depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
        return depth_colored

    def calculate_fps(self):
        self.fps_counter += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.start_time = time.time()
            return fps
        return None

    def run(self, display_mode="side_by_side", camera_id=0):
        if not self.initialize_camera(camera_id):
            return

        print("🎥 Running optimized Depth Anything (press 'q' to quit)")
        display_modes = ["side_by_side", "overlay", "depth_only"]
        mode_index = display_modes.index(display_mode)
        current_fps = 0.0

        while True:
            if self.frame_queue.empty():
                continue
            frame = self.frame_queue.get()

            pil_image, undistorted = self.preprocess_frame(frame)

            try:
                # Depth inference
                inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device, torch.float16)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    depth_pred = self.model(**inputs).predicted_depth

                # Convert safely to float32 for OpenCV
                depth = depth_pred.squeeze().detach().cpu().float().numpy()

                depth_colored = self.postprocess_depth(depth, undistorted.shape[:2])


                fps = self.calculate_fps()
                if fps is not None:
                    current_fps = fps

                mode = display_modes[mode_index]
                if mode == "side_by_side":
                    disp = np.hstack((undistorted, depth_colored))
                elif mode == "overlay":
                    disp = cv2.addWeighted(undistorted, 0.6, depth_colored, 0.4, 0)
                else:
                    disp = depth_colored

                # Scale window by 1/4
                disp_small = cv2.resize(disp, (disp.shape[1] // 4, disp.shape[0] // 4))
                cv2.putText(disp_small, f"FPS: {current_fps:.1f}", (10, disp_small.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.imshow("Depth Anything + Undistortion (Optimized)", disp_small)

            except Exception as e:
                print(f"⚠️ Depth processing error: {e}")
                cv2.imshow("Depth Anything + Undistortion (Optimized)", undistorted)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mode_index = (mode_index + 1) % len(display_modes)

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["small", "base", "large"], default="small")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--display-mode", choices=["side_by_side", "overlay", "depth_only"], default="side_by_side")
    parser.add_argument("--calib-file", default="Undistort-and-Depth/fisheye_calibration_data.json")
    parser.add_argument("--gpu-undistort", action="store_true", help="Enable GPU-accelerated undistortion")
    parser.add_argument("--inference-size", type=int, nargs=2, default=[384, 384], help="Inference image size (w h)")
    args = parser.parse_args()

    cam = DepthAnythingCamera(
        model_size=args.model_size,
        calib_file=args.calib_file,
        use_gpu_undistort=args.gpu_undistort,
        inference_size=tuple(args.inference_size)
    )
    cam.run(display_mode=args.display_mode, camera_id=args.camera_id)


if __name__ == "__main__":
    main()
