from ultralytics import YOLO
import numpy as np
from src.schemas import SegmentorConfig
import torch


class Detector():
    def __init__(self, config: SegmentorConfig):
        self.checkpoint = config.detector_path
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def load(self):
        self.detector = YOLO(self.checkpoint).to(self.device)
        print(f"Detector loaded and use {self.device} device")

    def infer(self, img):
        result = self.detector.predict(img, retina_masks=True)
        mask = result[0].masks.data.cpu().numpy()[0]
        mask = (mask*255).astype(np.uint8)
        return mask
