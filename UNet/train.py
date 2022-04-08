import argparse
import os
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.utils import *
from utils.dataset import *

from unet import UNet

from torchvision import transforms, datasets

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="/workspace/medical_deeplearning/data/", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="test", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

# HyperParameter
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
log_dir = args.log_dir
ckpt_dir = args.ckpt_dir
result_dir = args.result_dir
data_dir = args.data_dir

mode = args.mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([Normaization(mean=0.5, std = 0.5), RandomFlip(), ReSize(), ToTensor()])

dataset = UDataset(data_dir = data_dir, mode = mode, transform = transform)
data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False,  num_workers=0, pin_memory=True)

num_data = len(dataset)
num_batch = np.ceil(num_data / batch_size)

# 네트워크 설정
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

# function 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

# Tensorboard 사용을 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
# writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
st_epoch = 0

def dice_loss(pred, target, smooth = 1e-5):
    # binary cross entropy loss
    # bce = F.binary_cross_entropy_with_logits(pred, target, reduction='sum')
    
    pred = torch.sigmoid(pred)
    # intersection = (pred * target).sum(dim=(2,3))
    # union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    pred = pred.view(-1).to(device)
    target = target.view(-1).to(device)

    intersection = (pred * target).sum()                            
    dice = (2.*intersection + smooth)/(pred.sum() + target.sum() + smooth) 

    return dice

if mode == 'train':
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    for epoch in range(num_epoch):
        net.train()
        loss_arr = []
        for batch, data in enumerate(data_loader):
            image = data['image'].to(device)
            mask = data['mask'].to(device)

            output = net(image)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, mask)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch, np.mean(loss_arr)))

            # # Tensorboard 저장하기
            # label = fn_tonumpy(mask)
            # input = fn_tonumpy(fn_denorm(image, mean=0.5, std=0.5))
            # output = fn_tonumpy(fn_class(output))

            # print(mask.shape)

            # writer_train.add_image('mask', mask, num_batch * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('image', image, num_batch * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('output', output, num_batch * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 10 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(data_loader):
            # forward pass
            mask = data['mask'].to(device)
            image = data['image'].to(device)

            output = net(image)

            # 손실함수 계산하기
            # loss = fn_loss(output, mask)
            dice = dice_loss(output, mask)

            loss_arr += [dice.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch, np.mean(loss_arr)))

            # # Tensorboard 저장하기
            # label = fn_tonumpy(label)
            # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            # output = fn_tonumpy(fn_class(output))

            for j in range(mask.shape[0]):
                id = num_batch * (batch - 0) + j
                print(np.transpose(image[j].squeeze(), (1,2,0)).shape)
                plt.imsave(os.path.join(result_dir, 'png', 'mask_%04d.png' % id), image[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'image_%04d.png' % id), np.transpose(image[j].squeeze(), (1,2,0))[:,:,0], cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'mask_%04d.npy' % id), mask[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'image_%04d.npy' % id), image[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch, np.mean(loss_arr)))