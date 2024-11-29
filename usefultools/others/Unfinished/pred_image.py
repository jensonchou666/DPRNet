import os, sys
import re
import torch
from torch import nn
import numpy as np
import argparse
from pathlib import Path
import os.path as osp
from tqdm import tqdm
import cv2
import time, random
from PIL import Image
import albumentations as albu


from jscv.utils.cfg import *
import jscv.utils.analyser as analyser
from torch.utils.data import DataLoader
from jscv.models.pred_stat_t import *



model_path = "work_dir/vaihingen/ft-unetformer-e150/final/epoch=85@val_mIoU=77.66@ft-unetformer-vaihingen.ckpt"
# model_path = "work_dir/vaihingen/pvtv2_b5-unetformer-e150/final/epoch=144@val_mIoU=77.75@pvtv2_b5-unetformer-vaihingen.ckpt"
imgs_path = "a/images"
masks_path = "a/masks"
out_path = "a/out"

#? 这三项需要修改成对应数据集的！！！ 
train_mean = [0.3187, 0.3204, 0.4805]
train_std = [0.1405, 0.1448, 0.2086]
num_classes = 6
#?






def seed_everything(seed, do_init=True):
    if seed == "time":
        seed = int(time.time())
    elif seed == "random":
        seed = random.randint(10, 100000)
    elif isinstance(seed, str):
        seed = int(seed)
    else:
        assert isinstance(seed, int)
    if do_init:
        print("seed:", seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    return seed




def load_data(img_path, mask_path):
    global train_mean, train_std
    # load
    img = np.array(Image.open(img_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # albumentations,  only Normalize
    data = albu.Normalize(train_mean, train_std, max_pixel_value=255)(image=img, mask=mask)

    # to_tensor
    img = torch.from_numpy(data["image"]).permute(2, 0, 1).float()
    mask = torch.from_numpy(data["mask"]).long()
    return img, mask




if not osp.exists(out_path):
    os.makedirs(out_path)

''' model '''
ckptdict = torch.load(model_path, map_location='cpu')
if "seed" in ckptdict:
    seed = ckptdict["seed"]
    seed_everything(seed)
    
model = ckptdict['model'].cuda().eval()

with torch.no_grad():
    for f1 in os.listdir(imgs_path):
        n, e = osp.splitext(f1)
        f1 = osp.join(imgs_path, f1)
        f2 = osp.join(masks_path, f"{n}.png")
        img, mask = load_data(f1, f2)

        result = model(img.cuda().unsqueeze(0))
        if isinstance(result, dict):
            pred = result["pred"][0]
        else:
            pred = result[0]

        top1 = nn.Softmax(dim=0)(pred).argmax(dim=0)
        
        fpath = osp.join(out_path, f"{n}-[0]-image.png")
        orgimg = cv2.cvtColor(np.array(Image.open(f1).convert('RGB')), cv2.COLOR_RGB2BGR)
        cv2.imwrite(fpath, orgimg)

        fpath = osp.join(out_path, f"{n}-[1]-pred.png")
        map = analyser.label2rgb(top1.cpu().numpy(), num_classes)
        cv2.imwrite(fpath, cv2.cvtColor(map, cv2.COLOR_RGB2BGR))

        fpath = osp.join(out_path, f"{n}-[2]-target.png")
        map = analyser.label2rgb(mask.numpy(), num_classes)
        cv2.imwrite(fpath, cv2.cvtColor(map, cv2.COLOR_RGB2BGR))
