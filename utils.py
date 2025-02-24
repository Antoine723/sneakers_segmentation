import numpy as np

def get_masked_img(img, mask):
    return img*np.stack((mask.astype(np.uint8),)*3, axis=-1)


# import os

# root_path = "C:/Users/nonoa/Documents/Segmentation/yolo_seg/"
# for mode in os.listdir(root_path):
#     if os.path.isdir(f"{root_path}/{mode}"):
#         label_path = f"{root_path}/{mode}/labels/"
#         for label in os.listdir(label_path):
#             with open(f"{label_path}/{label}", "r") as f:
#                 content = f.readlines()
#                 new_content = []
#                 for line in content:
#                     if line[1] == " ":
#                         line = "0" + line[1:]
#                     else:
#                         line = "0" + line[2:]
#                     new_content.append(line)
#             with open(f"{label_path}/{label}", "w") as f:
#                 f.write("".join(new_content))


