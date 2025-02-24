import cv2
import numpy as np
from pathlib import Path
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from ultralytics import YOLO
import os
import shutil
from rembg.bg import remove

# def load_sam(checkpoint: Path, model_cfg:str, device: str="cpu"):
#     sam = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
#     return sam
    

# def load_yolo(checkpoint:Path, device:str="cpu"):
#     yolo_detector = YOLO(checkpoint).to(device) 
#     return yolo_detector   


def infer_yolo(model, img):
    result = model.predict(img)
    box = result[0].boxes.xyxy.cpu().numpy()
    box = np.array([int(box[0][0]), int(box[0][1]), int(box[0][2]), int(box[0][3])])
    return box

def infer_sam(model, img, bbox):
    model.set_image(img)
    mask, logit, low_res_mask = model.predict(
        box=bbox,
        multimask_output=False,
    )
    mask = np.uint8(mask)[0]

    mask = post_process(img, mask)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    masked_img[np.where(masked_img == 0)] = 255
    return masked_img

def post_process(image, mask):
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.erode(mask, kernel, iterations=3)
    # kernel = np.ones((4, 4), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=13)

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    return mask


src_dir = "/home/antoine/Documents/Segmentation/dataset/full"
target_dir = "target"
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)

import time
start_time = time.time()

for i, img in enumerate(os.listdir(src_dir)):
    path = f"{src_dir}/{img}"
    img = cv2.imread(str(path))
    output = remove(img)
    output = cv2.cvtColor(output, cv2.COLOR_RGBA2RGB)
    cv2.imwrite(f"{target_dir}/{i}.jpg", output)
    break
end_time = time.time()
print(f"Execution time : {end_time-start_time} seconds")
# yolo_checkpoint = "C:/Users/nonoa/Documents/Segmentation/custom-yolo-detect/train/weights/best.pt"
# sam_checkpoint = "C:/Users/nonoa/Documents/Segmentation/sam2_hiera_large.pt"
# sam_cfg = "sam2_hiera_l.yaml"

# yolo = load_yolo(yolo_checkpoint, "cuda")
# sam = load_sam(sam_checkpoint, sam_cfg, "cuda")

# for i, img in enumerate(os.listdir(src_dir)):
#     path = f"{src_dir}/{img}"
#     img = cv2.imread(str(path))
#     box = infer_yolo(yolo, img)
#     masked_img = infer_sam(sam, img, box) 
    # cv2.imwrite(f"{target_dir}/{i}.jpg", masked_img)








