import os
from PIL import Image
import numpy as np
import tqdm
import shutil
import torch.nn as nn
import torch
import cv2

"""
    用于演示
    EdgeAcc的计算范围，提取边缘附近区域
"""

ignore_idx = 100
edge_pool_kernel = 18

label_dir = "./data/gid_water/val/label"
out_dir = f'./edge_{edge_pool_kernel}'


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    

def get_edge_conv2d(channel=1):
    conv_op = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, channel, axis=0)
    sobel_kernel = np.repeat(sobel_kernel, channel, axis=1)
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    return conv_op
edge_conv2d = get_edge_conv2d()
edge_pool = nn.MaxPool2d(kernel_size=edge_pool_kernel, stride=1, padding=edge_pool_kernel//2)
ignore_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=5//2)


def get_edge_map(label):
    label = label.unsqueeze(0)
    ignore_mask = (label == ignore_idx).float()
    ignore_mask = ignore_pool(ignore_mask).bool()
    label = edge_conv2d(label.float())
    edge = (label > 0.1) & ~ignore_mask
    edge = edge_pool(edge.float()).bool().squeeze()
    return edge


for f in tqdm.tqdm(os.listdir(label_dir)):
    fn = os.path.splitext(f)[0]
    label = torch.from_numpy(np.array(Image.open(label_dir+"/"+f)))
    edge = get_edge_map(label)


    cv2.imwrite(out_dir+'/'+f+"_edge.png", (edge*255).numpy().astype(np.uint))

