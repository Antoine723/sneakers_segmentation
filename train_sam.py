import torch
from torch.utils.data import DataLoader, Dataset
from segment_anything import SamPredictor, sam_model_registry
from torch.optim import Adam
import monai
import os
import glob
import pathlib
import json
import cv2
import numpy as np
from utils import get_masked_img
from tqdm import tqdm
# Définition du répertoire racine contenant les données d'entraînement et de validation
root_dir = "C:/Users/nonoa/Downloads/sam_dataset_small"
train_dir = f"{root_dir}/train"
valid_dir = f"{root_dir}/valid"

# Définition du type de modèle à utiliser
MODEL_TYPE = "vit_h"


class CustomDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.labels_file = f"{dir}/annotations.json"
        self.images = self._load_images()

    def _load_images(self):
        images = []
        with open(self.labels_file, "r") as f:
            labels = json.load(f)
        self.categories = labels["categories"]
        for annot in labels["annotations"]:
            images.append((self.dir+ "/" + [img for img in labels["images"] if img["id"] == annot["image_id"]][0]["file_name"], np.array(annot["bbox"]), annot["segmentation"]))
        return images
    
    def _convert_label_to_mask(self, label, size):
        mask = np.zeros(size)
        final_label = []
        for i in range(0,len(label[0]), 2):
            x = round(label[0][i])
            y = round(label[0][i+1])
            final_label.append([x,y])
            # final_label[x][y] = True
        cv2.fillPoly(mask, pts=[np.array(final_label)], color=(255))
        # return final_label
        return torch.from_numpy(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, bbox, label = self.images[idx]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        # label = self._convert_label_to_mask(label, img.shape)
        return img, bbox, self._convert_label_to_mask(label, img.shape[:2])
    
def create_mask_from_points(points, mask_shape):
   
    mask = np.zeros(mask_shape)
    # Convertir les points en un tableau numpy
    pts = np.array(points)    # Remplir le polygone avec des valeurs de 1 dans le masque
    cv2.fillPoly(mask, [pts], color=255)
    return mask.astype(np.uint8)

train_dataset = CustomDataset(train_dir)

# seg = train_dataset.__getitem__(1)[2]
# pts = train_dataset._convert_label_to_mask(seg, None)
# img = train_dataset.__getitem__(1)[0]
# mask = np.zeros(img.shape)
# cv2.fillPoly(mask, pts=[np.array(pts)], color=(255, 255, 255))
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.imshow("Mask", mask)
# cv2.waitKey(0)
# cv2.imshow("Masked img", img*mask.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Chargement du modèle SAM pré-entraîné
sam_model = sam_model_registry[MODEL_TYPE](checkpoint="C:/Users/nonoa/Documents/Segmentation/sam.pth")
sam_model.to("cuda")  # Utiliser le GPU pour l'entraînement
mask_predictor = SamPredictor(sam_model)
optimizer = Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
# #Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# def get_mask(img, bbox):
#     input_image = sam_model.preprocess(torch.from_numpy(img.astype(np.float32).reshape(1, 3,640,640)).to("cuda"))
#     with torch.no_grad():
#         image_embedding = sam_model.image_encoder(input_image)

#     with torch.no_grad():
#         sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
#             points=None,
#             boxes=torch.from_numpy(bbox).to("cuda").reshape((1,1,4)),
#             masks=None,
#         )
#     low_res_masks, iou_predictions = sam_model.mask_decoder(
#     image_embeddings=image_embedding,
#     image_pe=sam_model.prompt_encoder.get_dense_pe(),
#     sparse_prompt_embeddings=sparse_embeddings,
#     dense_prompt_embeddings=dense_embeddings,
#     multimask_output=False,
#     )

#     upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_image.shape[-2:], img.shape[:2]).to("cuda")

#     from torch.nn.functional import threshold, normalize

#     binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to("cuda")
#     return binary_mask
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0
    for img, bbox, label in tqdm(train_dataset):
        optimizer.zero_grad()
        mask_predictor.set_image(img)
        masks, scores, logits = mask_predictor.predict(
            box=bbox,
            multimask_output=True
        )

        mask = torch.from_numpy(masks[np.argmax(scores)].astype(np.float32))
        mask.requires_grad = True
        label.requires_grad = True
        loss = seg_loss(mask, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss at epochs {epoch + 1} = {total_loss/len(train_dataset)}")
    torch.save(sam_model.state_dict(), "C:/Users/nonoa/Documents/Segmentation/finetuned_sam.pth")
# masked_img = img*create_mask_from_points(label, img.shape[:2])

# cv2.imwrite(f"C:/Users/nonoa/Documents/Segmentation/masked.jpg", masked_img)


#x,y,x,y => segmentation format
