from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2

class MaskPredictor():
    def __init__(self):
        self.checkpoint = "/home/antoine/Documents/Segmentation/sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"
    
    def load(self):
        sam2_model = build_sam2(self.model_cfg, self.checkpoint, device="cuda")
        self.mask_predictor = SAM2ImagePredictor(sam2_model)
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        self.sp = SuperPointForKeypointDetection.from_pretrained(
            "magic-leap-community/superpoint"
        )

    def infer_with_points(self, img, output_dir):
        height, width = img.shape[:2]
        inputs = self.processor(img, return_tensors="pt")
        self.mask_predictor.set_image(img)

        with torch.no_grad():
            outputs = self.sp(**inputs)
            image_sizes = [(height, width)]
            outputs = self.processor.post_process_keypoint_detection(outputs, image_sizes)
        output = outputs[0]
        num_kp = 30 #TODO A affiner comme pour le n_clusters
        indices = np.argpartition(output["scores"], -num_kp)[-num_kp:]
        sscores = output["scores"][indices]
        pts = output["keypoints"][indices]
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(pts)
        selected_points = kmeans.cluster_centers_
        input_label = np.array([1]*len(selected_points))  # 1 pour un point positif
        masks, scores, logits= self.mask_predictor.predict(
            # box=box,
            multimask_output=True,
            point_coords=selected_points,
            point_labels=input_label
        )
        f_mask, _, _ = self.mask_predictor.predict(
            point_coords=selected_points,
            point_labels=input_label,
            mask_input=logits[np.argmax(scores), :, :][None, :, :],
            multimask_output=False,
        )
        final_mask = f_mask[0] 
        cv2.imwrite(f"{output_dir}/mask.jpg", (final_mask*255).astype(np.uint8))

        # for i, m in enumerate(masks):
        #     cv2.imwrite(f"mask_{i}.jpg", (m*255).astype(np.uint8))
        # kernel = np.ones((7, 7), np.uint8)  # Taille du noyau à ajuster
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # selected_points = np.array([pts[np.argmax(sscores)].tolist()])
        plt.imshow(img, cmap='gray')
        plt.scatter(selected_points[:, 0], selected_points[:, 1], c='r', marker='x')
        plt.title("Points d'intérêt sélectionnés")
        plt.savefig(f"{output_dir}/keypoints.jpg")
        plt.clf()
        return final_mask

    def infer_with_box(self, img, box):
        height, width = img.shape[:2]

        # box = np.array([int(box[0]*width), int(box[1]*height), int(box[2]*width), int(box[3]*height)])
        self.mask_predictor.set_image(img)
        masks, scores, logits= self.mask_predictor.predict(
            box=box,
            multimask_output=True,
        )
        mask = masks[scores.argmax()]
        # mask = np.uint8(mask)
        # kernel = np.ones((3, 3), np.uint8)
        # # mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # from scipy.ndimage import binary_fill_holes
        # import matplotlib.pyplot as plt

        # mask = binary_fill_holes(mask).astype(np.uint8)
        # plt.axis("off")
        # plt.imshow(img)
        # plt.scatter(
        #     pts[:, 0],
        #     pts[:, 1],
        #     c=sscores * 100,
        #     s=sscores * 50,
        #     alpha=0.8,
        # )
        # plt.savefig(f"output_image.png")
        print("Seg done")
        return mask



