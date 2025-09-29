import time
import torch
import cv2
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys

# --- Adjust paths ---
repo_path = r"C:/Users/Torenia/SGDepth"  # path to repo root
model_path = r"./models/model.pth"

sys.path.insert(0, repo_path)

from models.sgdepth import SGDepth
from arguments import InferenceEvaluationArguments

class VideoInference:
    def __init__(self, opt):
        self.opt = opt
        self.model_path = opt.model_path
        self.source = opt.image_path  # webcam index or video file
        self.num_classes = 20
        self.depth_min = opt.model_depth_min
        self.depth_max = opt.model_depth_max
        self.output_path = opt.output_path
        self.output_format = opt.output_format
        self.all_time = []

        # Cityscapes colormap
        self.labels = (
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232)),
            ('building', (70, 70, 70)),
            ('wall', (102, 102, 156)),
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32)),
        )

    def init_model(self):
        print("Initializing model...")
        sgdepth = SGDepth

        with torch.no_grad():
            self.model = sgdepth(
                self.opt.model_split_pos, self.opt.model_num_layers, self.opt.train_depth_grad_scale,
                self.opt.train_segmentation_grad_scale,
                self.opt.train_weights_init, self.opt.model_depth_resolutions, self.opt.model_num_layers_pose,
            )

            state = self.model.state_dict()
            to_load = torch.load(self.model_path)
            for k in state.keys():
                if k in to_load:
                    state[k] = to_load[k]
                else:
                    print(f"Warning: key {k} missing in checkpoint")

            self.model.load_state_dict(state)
            self.model = self.model.eval().cuda()

    def normalize(self, tensor):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return transforms.Normalize(mean, std)(tensor)

    def prepare_batch(self, frame_bgr):
        """Convert OpenCV frame to batch dict for model"""
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        resize = transforms.Resize((self.opt.inference_resize_height, self.opt.inference_resize_width))
        image = resize(pil_img)
        tensor = transforms.ToTensor()(image)
        tensor = self.normalize(tensor).unsqueeze(0).float().cuda()

        batch = {
            ('color_aug', 0, 0): tensor,
            ('color', 0, 0): tensor,
            'domain': ['cityscapes_val_seg'],
            'purposes': [['segmentation'], ['depth']],
            'num_classes': torch.tensor([self.num_classes]),
            'domain_idx': torch.tensor(0),
        }
        return (batch,), pil_img.size

    def scale_depth(self, disp):
        min_disp = 1 / self.depth_max
        max_disp = 1 / self.depth_min
        return min_disp + (max_disp - min_disp) * disp

    def render_predictions(self, segs_pred, depth_pred, orig_size, frame_bgr):
        """Return stacked [original | segmentation | depth]"""
        ow, oh = orig_size

        # Segmentation
        segs_pred = segs_pred[0]
        seg_img = np.zeros((segs_pred.shape[0], segs_pred.shape[1], 3), dtype=np.uint8)
        for lab, (_, color) in enumerate(self.labels):
            seg_img[segs_pred == lab] = color
        seg_img = cv2.resize(seg_img, (ow, oh), interpolation=cv2.INTER_NEAREST)

        # Depth
        depth_pred = np.array(depth_pred[0][0].cpu())
        depth_pred = self.scale_depth(depth_pred)
        depth_pred = (depth_pred * (255.0 / depth_pred.max())).clip(0, 255).astype(np.uint8)
        depth_img = cv2.applyColorMap(depth_pred, cv2.COLORMAP_PLASMA)
        depth_img = cv2.resize(depth_img, (ow, oh))

        # Original
        orig_resized = cv2.resize(frame_bgr, (ow, oh))

        return np.concatenate((orig_resized, seg_img, depth_img), axis=0)

    def run(self):
        self.init_model()

        # Source selection
        if str(self.source).lower() == "camera" or str(self.source).isdigit():
            cap = cv2.VideoCapture(int(self.source) if str(self.source).isdigit() else 0)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print("Could not open video source:", self.source)
            return

        # Optional video writer
        writer = None
        if self.output_path and self.output_format:
            os.makedirs(self.output_path, exist_ok=True)
            out_file = os.path.join(self.output_path, "output" + self.output_format)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h * 3))
            print("Saving output video to:", out_file)

        print("Starting video. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            batch, orig_size = self.prepare_batch(frame)
            start = time.time()
            with torch.no_grad():
                output = self.model(batch)
            self.all_time.append(time.time() - start)

            disp = output[0]["disp", 0]
            seg = output[0]['segmentation_logits', 0].exp().cpu().numpy().argmax(1)

            stacked = self.render_predictions(seg, disp, orig_size, frame)

            cv2.imshow("SGDepth - Realtime", stacked)
            if writer:
                writer.write(stacked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        if len(self.all_time) > 1:
            print("Average inference time per frame:", np.mean(self.all_time[1:]))


if __name__ == "__main__":
    opt = InferenceEvaluationArguments().parse()
    VideoInference(opt).run()
