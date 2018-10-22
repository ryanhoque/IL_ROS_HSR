import cv2
import numpy as np
from fast_grasp_detect.data_aug.augment_lighting import get_lighting, get_depth_aug
import os

def flip_data_vertical(img,label,clss):
    v_img = cv2.flip(img,1)
    label[0] = w-label[0]
    return {'img': v_img, 'pose': label, 'class': clss}

IMG_DIR = "/nfs/diskstation/ryanhoque/corner_data/depth_images" # depth images only
LABEL_FILE_PATH = "/nfs/diskstation/ryanhoque/corner_data/labels.txt" # label line number is for the corresponding depth image
with open(LABEL_FILE_PATH, 'r') as file:
	labels = file.read().split('\n')
OUTPUT_DIR = "/nfs/diskstation/ryanhoque/corner_data/augmented_depth_images"

fnames = sorted(os.listdir(IMG_DIR), key=lambda x: int(x[6:-4])) # this is not alphabetical
for i in range(len(fnames)):
	img = cv2.imread(os.path.join(IMG_DIR, fnames[i]))
	img = cv2.resize(img, (448, 448))
	augmented = get_depth_aug(img)
	if int(labels[i]):
		subfolder = 'success'
	else:
		subfolder = 'failure'
	for a in range(len(augmented)):
		cv2.imwrite(os.path.join(OUTPUT_DIR, subfolder, "image-" + str(i) + "-" + str(a) + ".png"), augmented[a])
		v_img = cv2.flip(augmented[a], 1)
		cv2.imwrite(os.path.join(OUTPUT_DIR, subfolder, "image-" + str(i) + "-" + str(a) + "2" + ".png"), v_img)
