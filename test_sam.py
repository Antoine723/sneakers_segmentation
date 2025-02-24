from ultralytics import YOLO
import cv2
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import os
import typer
import torch
from segment_anything.utils.transforms import ResizeLongestSide

# def sam_infer(yolo_detector_model, mask_predictor, img):
#     result = yolo_detector_model.predict(img)
#     box = result[0].boxes.xyxy.cpu().numpy()
#     mask_predictor.set_image(img)
#     masks, scores, logits = mask_predictor.predict(
#         box=box,
#         multimask_output=True
#     )
#     mask = masks[np.argmax(scores)]
#     masked_img=img*np.stack((mask.astype(np.uint8),)*3, axis=-1)
#     masked_img[np.where((masked_img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
#     return mask

# MODEL_TYPE = "vit_h"
# sam_model = sam_model_registry[MODEL_TYPE](checkpoint="C:/Users/nonoa/Documents/Segmentation/sam.pth")
# sam_model.to("cuda")
# yolo_detector_model = YOLO("C:/Users/nonoa/Documents/Segmentation/shoes/train3/weights/best.pt ").to("cuda")
# batched_input = []
# sam_predictor = SamPredictor(sam_model)

# for i in range(2):
#     current_input = {}
#     img_path = f"C:/Users/nonoa/Downloads/shoes_valid_dataset/BEFOREBRUT/{54+i}.jpg"
#     img = cv2.imread(img_path)
#     # c.append(sam_infer(yolo_detector_model, sam_predictor, img))
#     result = yolo_detector_model.predict(img)
#     input_image = ResizeLongestSide(sam_model.image_encoder.img_size).apply_image(img)
#     input_image_torch = torch.as_tensor(input_image, device="cuda")
#     input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
#     box = result[0].boxes.xyxy
#     current_input["original_size"] = img.shape[:2]
#     current_input["image"] =input_image_torch
#     current_input["boxes"] = box
#     batched_input.append(current_input)

# res = sam_model(batched_input, True)
# mask = res[0]["masks"].reshape(2920,5184,3)
# img = cv2.imread("C:/Users/nonoa/Downloads/shoes_valid_dataset/BEFOREBRUT/54.jpg")
# masked_img = img*mask.cpu().numpy()
# # masked_img[np.where((masked_img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
# cv2.imwrite("C:/Users/nonoa/Documents/Segmentation/eval/2/test.jpg", masked_img)

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

url = "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80"
img = cv2.imread("C:/Users/nonoa/Downloads/shoes_valid_dataset/BEFOREBRUT/27.jpg")

# image = Image.open("C:/Users/nonoa/Downloads/shoes_valid_dataset/BEFOREBRUT/27.jpg")
inputs = processor(images=img, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(
    logits,
    size=img.shape[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]
# plt.imshow(pred_seg)
pred_seg = pred_seg.cpu().numpy().astype(np.uint8)
cv2.imwrite("C:/Users/nonoa/Documents/Segmentation/teston.jpg", cv2.bitwise_and(img, img, mask=pred_seg))
