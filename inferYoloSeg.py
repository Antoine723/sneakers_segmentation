from ultralytics import YOLO
import cv2
from segment_anything import SamPredictor, sam_model_registry
from mobile_sam import sam_model_registry as mobile_sam_model_registry, SamPredictor as MobileSamPredictor
import numpy as np
import os
import typer
import torch
import pathlib
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = typer.Typer()
checkpoint = "C:/Users/nonoa/Documents/Segmentation/sam2_hiera_tiny.pt"

def yolo_seg_infer(model, img):
    result = model.predict(img, retina_masks=True)
    mask = result[0].masks.data.cpu().numpy()[0]
    mask = mask.astype(np.uint8)
    # masked_img=img*np.stack((mask.astype(np.uint8),)*3, axis=-1)
    # masked_img[np.where((masked_img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    return cv2.bitwise_and(img, img, mask=mask)
    # return masked_img

def draw_mask(image, mask_generated) :
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
def compare_yolo_seg_and_sam(img_path:str = typer.Option("--img_path"), output_folder:str = typer.Option("--output_folder"), yolo_seg_weights_path:str = typer.Option("--yolo_seg_weights_path"), yolo_detector_weights_path:str = typer.Option("--yolo_detector_weights_path"), sam_weights_path:str = typer.Option("--sam_weights_path")):
    cv2.imwrite(f"{output_folder}/original.jpg", cv2.imread(img_path))
    yolo_seg = YOLO(yolo_seg_weights_path)
    yolo_seg.to("cuda")
    img = cv2.imread(img_path)
    cv2.imwrite(f"{output_folder}/yolo_seg.jpg", yolo_seg_infer(yolo_seg, img.copy()))
    del yolo_seg
    yolo_detector = YOLO(yolo_detector_weights_path)
    MODEL_TYPE = "vit_h"
    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=sam_weights_path)
    sam_model.to("cuda")
    mask_predictor = SamPredictor(sam_model)
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    cv2.imwrite(f"{output_folder}/sam.jpg", sam_infer(yolo_detector, predictor, img.copy()))




@app.command()
def generate_masked_img(img_path:str = typer.Option("--img_path"), output_folder:str = typer.Option("--output_folder"), yolo_detector_weights_path:str = typer.Option("--yolo_detector_weights_path"), sam_weights_path:str = typer.Option("--sam_weights_path"), mobile:str=typer.Option("--mobile")):
    mobile = mobile == "True"
    img = cv2.imread(img_path)
    cv2.imwrite(f"{output_folder}/original.jpg", img)
    yolo_detector = YOLO(yolo_detector_weights_path).to("cpu")    
    # sam_model = sam_model_registry["vit_h"](checkpoint=sam_weights_path) if not mobile else mobile_sam_model_registry["vit_t"](checkpoint=sam_weights_path)
    # sam_model.load_state_dict(torch.load("C:/Users/nonoa/Documents/Segmentation/finetuned_sam.pth"))
    # sam_model.to("cuda")
    # mask_predictor = SamPredictor(sam_model) if not mobile else MobileSamPredictor(sam_model)
    sam2_checkpoint = "C:/Users/nonoa/Documents/Segmentation/sam2_hiera_large.pt"
    model_cfg = "C:/Users/nonoa/Documents/Segmentation/sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")

    mask_predictor = SAM2ImagePredictor(sam2_model)
    output_name = "sam2.jpg" if not mobile else "mobile_sam.jpg"
    cv2.imwrite(f"{output_folder}/{output_name}", sam_infer(yolo_detector, mask_predictor, img.copy()))


@app.command()
def generate_masked_img_yolo_seg(img_path:str = typer.Option("--img_path"), output_folder:str = typer.Option("--output_folder"), yolo_seg_weights_path:str = typer.Option("--yolo_seg_weights_path")):
    img = cv2.imread(img_path)
    cv2.imwrite(f"{output_folder}/original.jpg", img)
    model_name = pathlib.Path(yolo_seg_weights_path).parent.parent.parent.name
    yolo_seg = YOLO(yolo_seg_weights_path).to("cuda")    
    yolo_seg.to("cuda")
    cv2.imwrite(f"{output_folder}/yolo_seg_{model_name}.jpg", yolo_seg_infer(yolo_seg, img.copy()))

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

