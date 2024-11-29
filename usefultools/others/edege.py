import torch, os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def edge_detect(input: torch.Tensor, out_channel=1):
    # 用nn.Conv2d定义卷积操作
    # global conv_op
    # if conv_op is None:

    in_channel = input.shape[1]

    conv_op = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, in_channel, axis=0)
    sobel_kernel = np.repeat(sobel_kernel, out_channel, axis=1)

    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op.to(input.device)(input)
    # print(torch.max(edge_detect))
    # 将输出转换为图片格式
    #edge_detect = edge_detect.squeeze()
    return edge_detect

path = "./a/1.png"
out = "./a/a1.png"

image = Image.open(path)
image = np.array(image)
image = torch.from_numpy(image)

print(image.shape)