from final.detector import Detector
from final.mask_predictor import MaskPredictor
import cv2
import numpy as np

class AutomaticSegmentor():
    def __init__(self):
        # self.detector = Detector()
        self.mask_predictor = MaskPredictor()

    def load(self):
        # self.detector.load()
        self.mask_predictor.load()

    def post_process(self, mask, img):
        height, width = img.shape[:2]
        # masked_img = img * np.stack((mask,)*3, axis=-1)
        # cv2.imwrite("leTest.jpg", masked_img)
        resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)
        
        binary_mask = (resized_mask > 0.5).astype(np.uint8) *255
        
        basket_extracted = cv2.bitwise_and(img, img, mask=binary_mask)
        
        background_white = np.ones_like(img) * 255  # Multiplie par 255 pour un fond blanc pur
        inverse_mask = cv2.bitwise_not(binary_mask)
        
        background_with_mask = cv2.bitwise_and(background_white, background_white, mask=inverse_mask)

        return cv2.add(basket_extracted, background_with_mask)
    
    def infer(self, img, output_dir):
        # small_img = cv2.resize(img, (1024,1024))

        # box, kps = self.detector.infer(small_img)
        # box[1] = 0
        # cpy = img.copy()
        # for kp in kps[0]:
        #     x = int(kp[0]*width)
        #     y = int(kp[1]*height)
        #     cpy = cv2.circle(cpy, (x,y),50, color=(0,0,255))
        # cv2.imwrite("test.jpg", cpy)
        # mask_with_box = self.mask_predictor.infer_with_box(small_img, box)
        mask_with_pts = self.mask_predictor.infer_with_points(img, output_dir)
        # res_with_box = self.post_process(mask_with_box, img)
        res_with_pts = self.post_process(mask_with_pts, img)

        return res_with_pts


