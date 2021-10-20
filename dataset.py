import nibabel as nib
import pydicom
import torch
import scipy
import glob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms
from medicaltorch import losses as mt_losses
from medicaltorch import metrics as mt_metrics
from medicaltorch import filters as mt_filters

from transforms3ds import *

import os

'''========参数设置========'''
num_workers=0
aug=False #是否进行数据增强（并不增加数据）

'''===================='''


class MRIDataset(Dataset):

    #获取图像路径与数据增广
    def __init__(self, root=None, flag_3d=False, mode='train',channel_size_3d=32, mri_slice_dim=512, aug=False):

        '''root:data根目录'./data/rectal_cancer/label_all/', aug=False:是否进行数据增强，默认非否'''

        Dataset.__init__(self)
        self.flag_3d=flag_3d
        self.root=root
        self.channel_size_3d=channel_size_3d

        "1.加载获取图像-mask路径列表"
        self.folders = glob.glob(self.root+'/*') #多个文件路径


        assert self.folders!=[] #是否成功读入


        #获取文件路径列表
        origin_dir=glob.glob(self.root+"*/*-O.nii*")
        C_mask_dir=glob.glob(self.root+"*/*-C-label.nii*")
        H_mask_dir=glob.glob(self.root+"*/*-H-label.nii*")
        TL_mask_dir=glob.glob(self.root+"*/*-TL*")
        # print(origin_dir)
        self.segmentation_pairs=[]
        "======================================="
        mask_dir=C_mask_dir
        "======================================="
        #获取图像-mask路径列表
        for idx in range(len(origin_dir)):
            self.segmentation_pairs.append([origin_dir[idx],mask_dir[idx]])
        # print(len(self.segmentation_pairs))

        "2.数据规整与增广"

        spl = [.8,.1,.1] #训练集验证集测试集划分比例

        train_ptr = int(spl[0]*len(self.segmentation_pairs))
        val_ptr = train_ptr + int(spl[1]*len(self.segmentation_pairs))

        #定义数据增广
        if not flag_3d:
            if aug:
                train_transforms = transforms.Compose([
                                MTResize((mri_slice_dim,mri_slice_dim)),
                                transforms.RandomChoice([
                                    mt_transforms.RandomRotation(30),
                                    mt_transforms.ElasticTransform(alpha=2000, sigma=50),
                                    mt_transforms.AdditiveGaussianNoise(mean=0.05, std=0.01),]),
                                mt_transforms.ToTensor(),
                                MTNormalize()])

            else:
                train_transforms = transforms.Compose([
                                MTResize((mri_slice_dim,mri_slice_dim)),
                                mt_transforms.ToTensor(),
                                MTNormalize()])

            val_transforms = transforms.Compose([
                                transforms.Resize((mri_slice_dim,mri_slice_dim)),
                                mt_transforms.ToTensor(),
                                MTNormalize()])

            train_unnormalized = train_transforms

        else:

            if aug:
                train_transforms = transforms.Compose([
                                    ToPILImage3D(),
                                    Resize3D((mri_slice_dim, mri_slice_dim)),
                                    transforms.RandomChoice([
                                        RandomHorizontalFlip3D(),
                                        RandomVerticalFlip3D(),
                                        RandomRotation3D(30),]),
                                    ToTensor3D(),
                                    Normalize3D('min_max')])

                train_unnormalized = transforms.Compose([
                                    ToPILImage3D(),
                                    Resize3D((mri_slice_dim, mri_slice_dim)),
                                    transforms.RandomChoice([
                                        RandomHorizontalFlip3D(),
                                        RandomVerticalFlip3D(),
                                        RandomRotation3D(30)]),
                                    ToTensor3D(),])
            else:
                train_transforms = transforms.Compose([
                                    ToPILImage3D(),
                                    Resize3D((mri_slice_dim, mri_slice_dim)),
                                    ToTensor3D(),
                                    Normalize3D('min_max')])

                train_unnormalized = transforms.Compose([
                                    ToPILImage3D(),
                                    Resize3D((mri_slice_dim, mri_slice_dim)),
                                    ToTensor3D(),])

            val_transforms = transforms.Compose([
                                    ToPILImage3D(),
                                    Resize3D((mri_slice_dim, mri_slice_dim)),
                                    ToTensor3D(),
                                    IndividualNormalize3D(),])
        #初始化数据增广方法
        if mode == 'train':
            self.segmentation_pairs = self.segmentation_pairs[:train_ptr]
            self.transforms = train_transforms
            self.seg_transforms = train_unnormalized
        elif mode == 'val':
            self.segmentation_pairs = self.segmentation_pairs[train_ptr:val_ptr]
            self.transforms = val_transforms
            self.seg_transforms = train_unnormalized
        else:
            self.segmentation_pairs = self.segmentation_pairs[val_ptr:]
            self.transforms = val_transforms
            self.seg_transforms = train_unnormalized

        if not flag_3d:
            self.twod_slices_dataset = mt_datasets.MRI2DSegmentationDataset(self.segmentation_pairs, transform=self.transforms)

    #获取数据集大小
    def __len__(self):
        if not self.flag_3d:
            return len(self.twod_slices_dataset)
        else:
            return len(self.segmentation_pairs)

    #获取数据集
    def __getitem__(self, item):

        #加载数据
        if not self.flag_3d:        #2D情况（没用到可以删除此代码）
            mt_dict = self.twod_slices_dataset.__getitem__(item)
            return mt_dict['input'], mt_dict['gt']
        else: #3D情况

            #获取原图和mask并转为tensor
            vox_fname, seg_fname = self.segmentation_pairs[item]
            fobj = nib.load(vox_fname)
            sobj = nib.load(seg_fname)
            inp, out = torch.tensor(fobj.get_fdata()), torch.tensor(sobj.get_fdata())

            #维度变换：(H,W,T)-->(T,H,W)
            inp = inp.permute(2, 0, 1)
            out = out.permute(2, 0, 1)

            #修剪/规范化3D图片厚度
            if inp.size(0)<=self.channel_size_3d:#厚度不够——将3D图零填充至指定的厚度
                batch_size = (self.channel_size_3d,) + inp.size()[1:]
                temp1, temp2 = torch.zeros(batch_size), torch.zeros(batch_size)
                temp1[:inp.size(0),:,:] = inp
                temp2[:out.size(0),:,:] = out
                inp, out = temp1, temp2
            else: #厚度超出——将3D图随机截取指定厚度
                r = np.random.randint(0, inp.size(0)-self.channel_size_3d)
                inp, out = inp[r:r+self.channel_size_3d,:,:], out[r:r+self.channel_size_3d,:,:]

            #数据增广和变换
            if self.transforms:
                # print('已执行数据增广')
                inp, out = self.transforms(inp), self.seg_transforms(out)

            out[out>0] = 1
            return inp, out

    #展示加载的数据序列
    def show_slices(self, slices, save=None):

        num_rows = len(slices)

        fig, axes = plt.subplots(len(slices), len(slices[0]))
        for i, mri_grp in enumerate(slices):
            for j, img in enumerate(mri_grp):
                if num_rows == 1:
                    axes[j].imshow(img.T, cmap="gray", origin="lower")
                    axes[j].axis('off')
                else:
                    axes[i, j].imshow(img.T, cmap="gray", origin="lower")
                    axes[i, j].axis('off')
        if not save:
            plt.show()
        else:
            plt.savefig(save)




if __name__=='__main__':


    # root='./data/rectal_cancer/label_all/' #我的电脑数据地址
    root=r'/home2/HWGroup/zhengrc/rectal_cancer/label_all/'  #服务器数据地址
    flag_3d = True
    mri_slice_dim = 512

    rectal_cancer=MRIDataset(root=root, mode='train',flag_3d=flag_3d, mri_slice_dim=mri_slice_dim,aug=aug)

    print('The size of dataset is:   {}'.format(rectal_cancer.__len__()))
    # prep = lambda x: 'Number of %s in  study for %s: %d\n'%('patients' if flag_3d else 'slices',  len(x))
    load_rectal_cancer=DataLoader(rectal_cancer, batch_size=32, shuffle=True,
                                num_workers=num_workers, collate_fn=mt_datasets.mt_collate)

    #展示第一个batch的第一个病人的MRI序列
    for i, (i1, i2) in enumerate(load_rectal_cancer):
        print (i1.shape, i2.shape, [torch.min(i1), torch.max(i1)], [torch.min(i2), torch.max(i2)])
        rows,clos=i1.shape[1]//6+1,6
        for num in range(1,i1.shape[1]+1):
            # print(rows,clos,num,i1[0,0,:,:,:].numpy().reshape(512,512).shape)
            plt.subplot(rows,clos,num)
            plt.imshow(i1[0,num-1,:,:,:].numpy().reshape(512,512),cmap='gray')
        plt.show()
        break



# root='./data/rectal_cancer/label_all/'
# M=MRIDataset(root=root)
# print(M.segmentation_pairs)

