from ultralytics import YOLO
import numpy as np

class Detector():
    def __init__(self):
        self.checkpoint = "/home/antoine/Documents/Segmentation/custom-yolo-detect/train/weights/best.pt"
        self.device = "cuda"

    def load(self):
        self.detector = YOLO(self.checkpoint).to(self.device)
        self.pose = YOLO("yolo11n-pose.pt")  # YOLOv8-Pose

    
    def infer(self, img):
        result = self.detector.predict(img)
        box = result[0].boxes.xyxy.cpu().numpy()
        height, width = img.shape[:2]
        box = box[0]
        r = self.pose.predict(img)
        # box = np.array([box[0][0]/width, box[0][1]/height, box[0][2]/width, box[0][3]/height])
        # box = np.array([int(box[0][0]), int(box[0][1]), int(box[0][2]), int(box[0][3])])
        return box , r[0].keypoints.xyn