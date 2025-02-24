from final.segmentor import AutomaticSegmentor
import cv2
import os
import shutil
import time

if __name__ == "__main__":
    start_time = time.time()
    src_dir = "/home/antoine/Documents/Segmentation/dataset/full"
    target_dir = "final/target"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    seg = AutomaticSegmentor()
    seg.load()
    # i = 22
    # img = os.listdir(src_dir)[22]
    for i, img in enumerate(os.listdir(src_dir)):
        path = f"{src_dir}/{img}"
        img = cv2.imread(str(path))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        os.mkdir(f"{target_dir}/{i}")
        cv2.imwrite(f"{target_dir}/{i}/original.jpg", img)
        masked_img_with_pts = seg.infer(img, f"{target_dir}/{i}")
        # cv2.imwrite(f"{target_dir}/{i}_w_box.jpg", masked_img_with_box)
        cv2.imwrite(f"{target_dir}/{i}/final.jpg", masked_img_with_pts)
    end_time = time.time()
    print(f"Execution time : {end_time-start_time} seconds")
