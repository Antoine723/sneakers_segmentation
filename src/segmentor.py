from src.detector import Detector
from src.mask_predictor import MaskPredictor
import cv2
import numpy as np
from pathlib import Path

from src.schemas import SegmentorConfig


class AutomaticSegmentor():
    def __init__(self, config: SegmentorConfig):
        self.detector = Detector(config)
        self.mask_predictor = MaskPredictor(config)

    def load(self):
        self.detector.load()
        self.mask_predictor.load()

    def post_process(self, mask: np.ndarray, img: np.ndarray):
        height, width = img.shape[:2]
        resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)

        binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        blurred = cv2.GaussianBlur(resized_mask, (5, 5), 0)
        binary_mask = (blurred > 0.4).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(binary_mask)
        largest = max(contours, key=cv2.contourArea)
        binary_mask = cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
        mask_blur = cv2.GaussianBlur(binary_mask, (9, 9), sigmaX=3)
        alpha_mask = mask_blur / 255.0  # Entre 0 et 1
        foreground = (img * alpha_mask[..., None]).astype(np.uint8)
        background = (255 * (1 - alpha_mask[..., None])).astype(np.uint8)
        final_image = foreground + background
        return final_image

    def infer(self, img: np.ndarray, output_dir: Path):
        first_mask = self.detector.infer(img)
        mask = self.mask_predictor.infer(img, first_mask, output_dir)
        print("Infer done")
        return self.post_process(mask, img)
