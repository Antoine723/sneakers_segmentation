from final.segmentor import AutomaticSegmentor
import cv2
import time

if __name__ == "__main__":
    start_time = time.time()
    path = "/home/antoine/Documents/Segmentation/dataset/full/240.jpg"
    img = cv2.imread(str(path))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg = AutomaticSegmentor()
    seg.load()
    masked_img_with_pts = seg.infer(img)
    # cv2.imwrite(f"{target_dir}/{i}_w_box.jpg", masked_img_with_box)
    cv2.imwrite(f"240_w_pts.jpg", masked_img_with_pts)
