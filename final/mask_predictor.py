from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import numpy as np
from sklearn.cluster import KMeans
import cv2
from pathlib import Path

from final.schemas import SegmentorConfig


class MaskPredictor():
    def __init__(self, config: SegmentorConfig):
        self.checkpoint = config.segmentor_path
        self.model_cfg = config.segmentor_config
        self.num_clusters = config.num_clusters
        self.num_kp = config.num_kp
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def load(self):
        sam2_model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
        self.mask_predictor = SAM2ImagePredictor(sam2_model)

        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        self.keypoints_detector = SuperPointForKeypointDetection.from_pretrained(
            "magic-leap-community/superpoint"
        )

    def get_keypoints_in_mask(self, keypoints: np.ndarray, mask: np.ndarray):
        filtered_points = []
        indices = []
        for i, (x,y) in enumerate(keypoints):
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
                filtered_points.append((x,y))
                indices.append(i)
        return np.stack(filtered_points), indices

    def infer_with_mask(self, img, first_mask, output_dir):
        height, width = img.shape[:2]

        self.mask_predictor.set_image(img)
        points = self.get_random_point_from_mask(first_mask)
        labels = np.array([1]*len(points))
        cv2.imwrite(f"{output_dir}/yolo_seg_mask.jpg", first_mask)
        cv2.imwrite(f"{output_dir}/yolo_seg_final.jpg", cv2.bitwise_and(img, img, mask=first_mask))

        masks, scores, logits = self.mask_predictor.predict(
            multimask_output=True,
            point_coords=points,
            point_labels=labels
        )

        f_mask, _, _ = self.mask_predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=logits[np.argmax(scores), :, :][None, :, :],
            multimask_output=False,
        )
        mask = f_mask[0]*255
        cv2.imwrite(f"{output_dir}/mask.jpg", mask.astype(np.uint8))
        return mask

    def infer(self, img: np.ndarray, mask: np.ndarray, output_dir: Path):
        height, width = img.shape[:2]
        inputs = self.processor(img, return_tensors="pt")
        self.mask_predictor.set_image(img)

        with torch.no_grad():
            outputs = self.keypoints_detector(**inputs)
            image_sizes = [(height, width)]
            outputs = self.processor.post_process_keypoint_detection(outputs, image_sizes)
        output = outputs[0]
        keypoints_in_mask, indices = self.get_keypoints_in_mask(output["keypoints"], mask)
        if self.num_kp > 0:
            indices = np.argpartition(output["scores"][indices], -self.num_kp)[-self.num_kp:]
            selected_points = keypoints_in_mask[indices]

        if self.num_clusters > 0:
            kmeans = KMeans(n_clusters=self.num_clusters)
            kmeans.fit(selected_points)
            selected_points = kmeans.cluster_centers_
        selected_points = keypoints_in_mask
        input_label = np.array([1]*len(selected_points))  # 1 = positive dot
        _, scores, logits = self.mask_predictor.predict(
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

        cpy_img = img.copy()
        for point in selected_points:
            x, y = int(point[0]), int(point[1])
            cv2.drawMarker(cpy_img, (x, y), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=50, thickness=3)
        cv2.imwrite(output_dir / "keypoints.jpg", cpy_img)
        return final_mask
