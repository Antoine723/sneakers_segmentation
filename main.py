from src.segmentor import AutomaticSegmentor
import cv2
import os
import shutil
import time
from tqdm import tqdm
from pathlib import Path
from src.schemas import SegmentorConfig
from src.config import settings

if __name__ == "__main__":
    start_time = time.time()
    src_dir = "/home/antoine/sneakers_segmentation/dataset/full"
    target_dir = "final/target"
    with open(settings.config_file_path, "r") as f:
        config = SegmentorConfig.model_validate_json(f.read())
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    seg = AutomaticSegmentor(config)
    seg.load()
    for i, img in tqdm(enumerate(os.listdir(src_dir))):
        path = f"{src_dir}/{img}"
        img = cv2.imread(str(path))
        os.mkdir(f"{target_dir}/{i}")
        cv2.imwrite(f"{target_dir}/{i}/original.jpg", img)
        masked_img = seg.infer(img, Path(f"{target_dir}/{i}"))
        cv2.imwrite(f"{target_dir}/{i}/final.jpg", masked_img)
    end_time = time.time()
    print(f"Execution time : {end_time-start_time} seconds")
