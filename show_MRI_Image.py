# encoding=utf8
'''
查看和显示nii文件
'''
import matplotlib
import numpy as np
from collections import  Counter
matplotlib.use('TkAgg')
  
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
  
example_filename = r'C:\Users\Jiyao\Desktop\直肠分割\data\rectal_cancer\label_all\120-JIANG LUO SHAN\120-O.nii'
  
img = nib.load(example_filename)
print (img)
print (img.header['db_name'])  #输出头信息
width,height,queue=img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()
  
num = 1
for i in range(0,queue,2):
  
  img_arr = img.dataobj[:,:,i]
  plt.subplot(5,4,num)
  plt.imshow(img_arr,cmap='gray')
  print(img_arr.shape)
  num +=1
plt.show()


#统计第20个切片的像素值个数
img_arr = img.dataobj[:,:,19]
print(img_arr.shape)  #512x512
print(type(img_arr))
print(Counter(img_arr.flatten()))

