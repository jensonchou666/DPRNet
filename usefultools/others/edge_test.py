img_path = '0-Test/edge_1/1_1_label.png'


import torch, os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import kornia
import cv2
import time


import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelEdgeDetection:
    def __init__(self, edge_pool_kernel=6, to_cuda=True):
        def get_edge_conv2d(channel=1):
            conv_op = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)
            sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
            sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
            sobel_kernel = np.repeat(sobel_kernel, channel, axis=0)
            sobel_kernel = np.repeat(sobel_kernel, channel, axis=1)
            conv_op.weight.data = torch.from_numpy(sobel_kernel)
            return conv_op
        self.edge_conv2d = get_edge_conv2d()
        if to_cuda:
            self.edge_conv2d = self.edge_conv2d.cuda()
        self.edge_pool = nn.MaxPool2d(kernel_size=edge_pool_kernel, stride=1, padding=edge_pool_kernel//2)

    def detect(self, mask):
        edge_range = self.edge_conv2d(mask.float()) > 0.1
        edge_range = self.edge_pool(edge_range.float()).squeeze(1).bool()
        return edge_range

dec = SobelEdgeDetection(to_cuda=False)

image = Image.open(img_path)
image = np.array(image)
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
edge = dec.detect(image)


dir1, name = os.path.split(img_path)
name, ext = os.path.splitext(name)
outf1 = os.path.join(dir1, name + str(int(time.time())) + ext)

cv2.imwrite(outf1, (edge.squeeze().int() * 255).numpy())