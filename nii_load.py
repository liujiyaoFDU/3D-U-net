import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from skimage import io
import cv2
import random
import nibabel as nib

path='./data/rectal_cancer/label_all/'

#获取文件路径列表
origin_dir=glob(path+"*/*-O.nii*")
C_mask_dir=glob(path+"*/*-C-label.nii*")
H_mask_dir=glob(path+"*/*-H-label.nii*")
TL_mask_dir=glob(path+"*/*-TL*")

print(origin_dir)
data_img = pd.DataFrame.from_dict({
    "image_path":origin_dir,
    "C_mask_path":C_mask_dir
})


# #显示data_img
# #显示所有列
# pd.set_option('display.max_columns', None)
# #显示所有行
# pd.set_option('display.max_rows', None)
# #设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth',100)
# print(data_img.head())



#加载训练集
def train_images(img_path,img_width,img_height):
    train_data = []
    for path in tqdm(img_path):
        img =nib.load(path)
        img= img.dataobj[:,:,:]
        # img = cv2.resize(img,(img_width,img_height))
        train_data.append(img)
    return train_data
train_image = train_images(origin_dir,512,512)