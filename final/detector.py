from ultralytics import YOLO
import numpy as np
from final.schemas import SegmentorConfig
import torch


class Detector():
    def __init__(self, config: SegmentorConfig):
        self.checkpoint = config.detector_path
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def load(self):
        self.detector = YOLO(self.checkpoint).to(self.device)

    def infer(self, img):
        result = self.detector.predict(img, retina_masks=True)
        mask = result[0].masks.data.cpu().numpy()[0]
        mask = (mask*255).astype(np.uint8)
        # masked_img=img*np.stack((mask.astype(np.uint8),)*3, axis=-1)
        # masked_img[np.where((masked_img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        return mask
