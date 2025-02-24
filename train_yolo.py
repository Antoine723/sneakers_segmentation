from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")
model.to("cuda")
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

if __name__ == "__main__":
   # Training.
   results = model.train(
      data="/home/antoine/Documents/Segmentation/yolo_detect_data/data.yaml",
      imgsz=640,
      epochs=300,
      batch=16,
      project='/home/antoine/Documents/Segmentation/custom-yolo-detect/',
      close_mosaic=0,
      patience=0,
      amp=False
      )
   


   