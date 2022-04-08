import matplotlib.pyplot as plt
import os
import numpy as np

import torch
import torch.nn as nn


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

# save Network
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net' : net.state_dict(), 'optim' : optim.state_dict()}, '%s/model_epoch%d.pth' % (ckpt_dir, epoch))

# load network
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list = sorted(ckpt_list)
    print('ckpt_list', ckpt_list)
    # ckpt_dir.sort(key=lambda f : int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_list[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_list[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch