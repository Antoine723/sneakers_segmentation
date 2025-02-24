from ultralytics import YOLO
import cv2
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import os
import typer
import torch
import pathlib
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = typer.Typer()


def draw_mask(image, mask_generated):
    masked_image = image.copy()

    masked_image = np.where(mask_generated.astype(int),
                            np.array([0,255,0], dtype='uint8'),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


def sam_infer(yolo_detector_model, mask_predictor, img):
    result = yolo_detector_model.predict(img)
    # img = np.float32(img)
    box = result[0].boxes.xyxy.cpu().numpy()
    box = np.array([int(box[0][0]), int(box[0][1]), int(box[0][2]), int(box[0][3])])
    mask_predictor.set_image(img)
    masks, scores, logits= mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    # cv2.imwrite("C:/Users/nonoa/Documents/Segmentation/test.jpg", cv2.rectangle(img, (int(box[0][0]),int(box[0][1])), (int(box[0][2]),int(box[0][3])), color=(255,0,0)))
    mask = masks[np.argmax(scores)]
    # mask = mask.astype(np.uint8)
    mask = np.uint8(mask)
    # mask = np.stack((mask.astype(np.float32),)*3, axis=-1)
    # masked_img=img*mask
    # masked_img[np.where((masked_img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    # return masked_img
    return cv2.bitwise_and(img, img, mask=mask)





@app.command()
def generate_masked_img(img_path:str = typer.Option("--img_path"), output_folder:str = typer.Option("--output_folder"), yolo_detector_weights_path:str = typer.Option("--yolo_detector_weights_path"), sam_weights_path:str = typer.Option("--sam_weights_path")):
    img = cv2.imread(img_path)
    cv2.imwrite(f"{output_folder}/original.jpg", img)
    yolo_detector = YOLO(yolo_detector_weights_path).to("cuda")    
    # sam_model = sam_model_registry["vit_h"](checkpoint=sam_weights_path)
    # sam_model.load_state_dict(torch.load("C:/Users/nonoa/Documents/Segmentation/finetuned_sam.pth"))
    # sam_model.to("cuda")
    sam2_checkpoint = "/home/antoine/Documents/Segmentation/sam2_hiera_large.pt"
    model_cfg = "/home/antoine/Documents/Segmentation/sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    mask_predictor = SAM2ImagePredictor(sam2_model)
    output_name = "sam2.jpg"
    cv2.imwrite(f"{output_folder}/{output_name}", sam_infer(yolo_detector, mask_predictor, img.copy()))


if __name__ == "__main__":
    app()


# python infer.py 
# --img-path C:/Users/nonoa/Downloads/shoes_valid_dataset/BEFOREBRUT/54.jpg 
# --output-folder C:/Users/nonoa/Documents/Segmentation/eval/2 
# --yolo-seg-weights-path C:/Users/nonoa/Documents/Segmentation/shoes/yolo-seg/train3/weights/best.pt
# --yolo-detector-weights-path C:/Users/nonoa/Documents/Segmentation/shoes/train3/weights/best.pt 
# --sam-weights-path C:/Users/nonoa/Documents/Segmentation/sam.pth


# python infer.py generate-masked-img --img-path C:/Users/nonoa/Downloads/shoes_real_dataset/BEFOREBRUT/27.jpg --output-folder C:/Users/nonoa/Documents/Segmentation/eval/3 --yolo-detector-weights-path C:/Users/nonoa/Documents/Segmentation/custom-yolo-detect/train/weights/best.pt --sam-weights-path C:/Users/nonoa/Documents/Segmentation/sam.pth

# python infer.py generate-masked-img-yolo-seg --img-path C:/Users/nonoa/Downloads/shoes_real_dataset/BEFOREBRUT/27.jpg --output-folder C:/Users/nonoa/Documents/Segmentation/eval/3 --yolo-seg-weights-path C:/Users/nonoa/Documents/Segmentation/custom-yolo-seg/train/weights/best.pt

