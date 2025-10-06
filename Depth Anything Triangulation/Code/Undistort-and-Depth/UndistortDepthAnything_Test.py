import cv2
import numpy as np
import torch
import json
from transformers import pipeline
from PIL import Image
import time
import argparse
import sys

class DepthAnythingCamera:
    def __init__(self, model_size="small", device=None, calib_file="fisheye_calibration_data.json"):
        """Initialize depth estimation + fisheye undistortion"""
        self.model_size = model_size

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

        # Precompute undistortion maps
        new_K = self.K.copy()
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, self.image_size, cv2.CV_16SC2
        )

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using GPU acceleration")
            else:
                self.device = torch.device("cpu")
                print("Using CPU (consider GPU for better performance)")
        else:
            self.device = device

        # Initialize depth model
        model_name = f"LiheYoung/depth-anything-{model_size}-hf"
        print(f"Loading model: {model_name}")
        try:
            self.depth_estimator = pipeline(
                "depth-estimation",
                model=model_name,
                device=0 if self.device.type == "cuda" else -1
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

        self.cap = None
        self.fps_counter = 0
        self.start_time = time.time()

    def initialize_camera(self, camera_id):
        # """Open camera"""
        # self.cap = cv2.VideoCapture(camera_id)
        # if not self.cap.isOpened():
        #     print(f"❌ Could not open camera {camera_id}")
        #     return False
        # print(f"✅ Camera initialized (ID: {camera_id})")

        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return False
            
        # Set camera properties for better performance
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"Camera initialized successfully (ID: {camera_id})")
        return True

    def preprocess_frame(self, frame):
        """Undistort + convert OpenCV frame to PIL Image"""
        frame_resized = cv2.resize(frame, self.image_size)
        undistorted = cv2.remap(frame_resized, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame), undistorted

    def postprocess_depth(self, depth_result, original_shape):
        """Convert depth map to visualization"""
        depth = np.array(depth_result["depth"])
        depth_resized = cv2.resize(depth, (original_shape[1], original_shape[0]))
        depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
        return depth_colored

    def calculate_fps(self):
        self.fps_counter += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.start_time = time.time()
            return fps
        return None

    def run(self, display_mode="side_by_side", camera_id=0):
        if not self.initialize_camera(camera_id):
            return

        print("🎥 Running real-time fisheye-undistorted Depth-Anything (press 'q' to quit)")

        display_modes = ["side_by_side", "overlay", "depth_only"]
        mode_index = display_modes.index(display_mode)
        current_fps = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # Undistort & preprocess
            pil_image, undistorted = self.preprocess_frame(frame)

            try:
                # Depth estimation
                depth_result = self.depth_estimator(pil_image)
                depth_colored = self.postprocess_depth(depth_result, undistorted.shape[:2])

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

                # 🔹 Scale window by 1/2
                disp_small = cv2.resize(disp, (disp.shape[1] // 2, disp.shape[0] // 2))

                cv2.putText(disp_small, f"FPS: {current_fps:.1f}", (10, disp_small.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.imshow("Depth Anything + Undistortion", disp_small)

            except Exception as e:
                print(f"⚠️ Depth processing error: {e}")
                cv2.imshow("Depth Anything + Undistortion", undistorted)

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
    parser.add_argument("--calib-file", default="Undistort-and-Depth/fisheye_calibration_data.json", help="Calibration JSON file")
    args = parser.parse_args()

    cam = DepthAnythingCamera(model_size=args.model_size, calib_file=args.calib_file)
    cam.run(display_mode=args.display_mode, camera_id=args.camera_id)


if __name__ == "__main__":
    main()
