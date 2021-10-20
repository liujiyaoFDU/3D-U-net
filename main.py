import sys

from dataset import MRIDataset
from Unet3D import UNet3D
from cumulative_average import CumulativeAverager
from medicaltorch import datasets as mt_datasets
import argparse  # 命令行选项、参数和子命令解析器
import math
import random
import shutil #文件和文件集合的高级操作
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau #当某指标不再变化（下降或升高），调整学习率
from torch.nn import BCELoss #Cross Entropy
from torch.nn.parallel import DataParallel #并行计算
from torch.nn.utils import clip_grad_norm_#梯度剪裁

'''Command Line Arguments'''
ap = argparse.ArgumentParser()
ap.add_argument('--batch_size', type=int, help='Number of 3D voxel batches',default=16)
ap.add_argument('--lr', type=float, help='Initial Learning rate', default=0.001)
ap.add_argument('--lr_decay', type=float, help='Learning Rate Decay', default=0.1)
ap.add_argument('--optimizer', help='sgd, adam', default='adam')
ap.add_argument('--epochs', help='Total number of epochs to train on data', default=200, type=int)
ap.add_argument('--iters', help='Number of training batches per epoch', default=None, type=int)
ap.add_argument('--aug', action='store_true', help='Flag to decide about input augmentations',default=False)
ap.add_argument('--demo', action='store_true', help='Flag to indicate testing',default=False)
ap.add_argument('--load_from', help='Path to checkpoint dict', default=None)
args = ap.parse_args()



'''设置根目录'''
# root='./data/rectal_cancer/label_all/'
root=r'/home2/HWGroup/zhengrc/rectal_cancer/label_all/'   #服务器数据地址

'''获取数据集'''
#如果显存溢出，就要resize一下输入的图片尺寸
def get_model(mode, flag_3d = True, channel_size_3d = 32, mri_slice_dim = 128):

    assert math.log(mri_slice_dim, 2).is_integer() # Image dims must be powers of 2

    #是否数据增广
    if mode == 'train' and args.aug:
        aug = True
    else:
        aug = False

    t1_lgg = MRIDataset(root=root,  mode=mode, channel_size_3d=channel_size_3d, flag_3d=flag_3d, mri_slice_dim=mri_slice_dim, aug=aug)
    dataset = t1_lgg

    return dataset

#保存训练损失日志
log_str = ''
def add_to_log(st):
    global log_str
    print (st)
    log_str = log_str+'\n'+st
    with open('log.txt', 'w') as f:
        f.write(log_str)
    return log_str

if not args.demo:
    primary_dataset = get_model(mode='train')
    val_dataset =  get_model(mode='val')
    primary_data_loader = DataLoader(primary_dataset, args.batch_size, shuffle=True,num_workers=0, collate_fn=mt_datasets.mt_collate)
    val_data_loader = DataLoader(val_dataset, args.batch_size, shuffle=True,num_workers=0, collate_fn=mt_datasets.mt_collate)
    add_to_log("training on %d samples"%len(primary_dataset))
else:
    primary_dataset = get_model('test')
    primary_data_loader = DataLoader(primary_dataset, shuffle=False, batch_size=1)

#
# print(len(primary_dataset))
# for i, (i1, i2) in enumerate(primary_data_loader):
#     print(i.shape,j.shape)
'''训练设置'''
device = torch.device("cuda:3")
net = UNet3D().to(device)
net.train()

net = DataParallel(net,device_ids=[3])
bce_criterion = BCELoss()


def get_optimizer(st, lr, momentum=0.9):
    if st == 'sgd':
        return SGD(net.parameters(), lr = lr, momentum=momentum)
    elif st == 'adam':
        return Adam(net.parameters(), lr = lr)

optimizer = get_optimizer(args.optimizer, args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)#当某指标不再变化调整学习率

losstype = [0,1,2,3,4,5], 'bce'

def dice_loss(y,pred):
    #parametes: y and pred are all inary matrix

    smooth=1.
    yflat=y.view(-1)
    predflat=pred.view(-1)

    intersection=(yflat*predflat).sum()

    return 1-(2*intersection+smooth)/(yflat.sum()+predflat.sum()+smooth)



def train(epoch,losstype='dice'):

    #set Number of training batches per epoch
    if args.iters is not None:
        primary_data_loader.dataset.segmentation_pairs = primary_data_loader.dataset.segmentation_pairs[:args.iters]
    print(primary_data_loader)
    for idx,(inp,seg) in enumerate(primary_data_loader) :
        print(idx,inp.shape,seg.shape)

        optimizer.zero_grad()
        inp, seg = torch.tensor(inp).cuda(3), torch.tensor(seg, requires_grad=False).cuda(3)
        out = net.forward(inp)

        if losstype=='dice':
            loss = dice_loss(seg, out)
        else:
            loss = bce_criterion(out, seg)

        avg_tool.update(loss)#将loss加入avgtool中列表，之后进行平均
        log_str = add_to_log("Epoch %d, Batch %d/%d: Loss=%0.6f"%(epoch, idx+1, len(primary_data_loader), avg_tool.get_average()))
        loss.backward()
        optimizer.step()

        clip_grad_norm_(net.parameters(), 5.0)

def validate(losstype): #num_patients=20):

    net.eval()
    val_loss_avg = CumulativeAverager()

    add_to_log("Performing validation test on %d samples"%len(val_dataset))

    if args.iters is not None:
        val_data_loader.dataset.segmentation_pairs = val_data_loader.dataset.segmentation_pairs[:args.iters]

    with torch.no_grad():

        for inp, seg in val_data_loader:

            inp, seg = torch.tensor(inp).cuda(3), torch.tensor(seg).cuda(3)
            out = net.forward(inp)
            if losstype == 'bce':
                loss = bce_criterion(out, seg)
            else:
                loss = dice_loss(seg, out)
            val_loss_avg.update(loss)

    val_loss = val_loss_avg.get_average()
    log_str = add_to_log('Validation Loss=%0.6f'%(val_loss))

    return val_loss

def test():
    pass

#保存最好的模型
def saver_fn(net_params, is_best, name='checkpt.pth.tar'):
    torch.save(net_params, name)

    if is_best is not None:
        shutil.copyfile(name, 'checkpt_best_%d.pth.tar'%(is_best))

if not args.demo:

    avg_tool = CumulativeAverager()

    vloss, is_best = torch.tensor(float(np.inf)), None
    if args.load_from is not None:
        if os.path.isfile(args.load_from):
            log_str = add_to_log("=> loading checkpoint '{}'".format(args.load_from))
            checkpoint = torch.load(args.load_from)
            start = checkpoint['epoch']
            vloss = checkpoint['best_val_loss']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.lr = checkpoint['learning_rate']
            log_str = add_to_log("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load_from, checkpoint['epoch']))
        else:
            log_str = add_to_log("=> no checkpoint found at '{}'".format(args.load_from))
    else:
        start = 0

    for epoch in range(start, args.epochs):

        train(epoch, losstype=losstype)
        val_loss = validate(losstype=losstype).cpu()
        scheduler.step(val_loss)

        if vloss > val_loss:
            vloss = val_loss
            is_best = epoch+1

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        saver_fn({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_val_loss': vloss,
            'optimizer' : optimizer.state_dict(),
			'learning_rate': lr
        }, is_best)

else:
    test()