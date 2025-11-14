import os
import cv2
import numpy as np
from tqdm import tqdm

img_path = './data/DT/processed_images'
img_path2 = './data/DT/processed_images'
names = os.listdir(img_path)

if not os.path.exists(img_path2):
    os.makedirs(img_path2)    

for name in tqdm(names):
    if name[-3:]=='png':
        img = cv2.imread(os.path.join(img_path,name),cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(img_path2,name),img)

