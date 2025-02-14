import os
from PIL import Image
import numpy as np
import tqdm
import shutil
import torch

PH, PW = 8, 8

dir_imgs = "work_dir/GID_Water/dbnet_Gr101_Lr18-e80/version_3/2024-07-01-21-29-11"

dir_out = 'B'

pathes_dirs = []
for i in range(PH):
    D = []
    for j in range(PW):
        D.append(os.path.join(dir_out, f'{i}_{j}'))
        if not os.path.exists(D[-1]):
            os.makedirs(D[-1])
    pathes_dirs.append(D)

for f in tqdm.tqdm(os.listdir(dir_imgs)):
    x = np.array(Image.open(dir_imgs+"/"+f))
    H, W = x.shape[:2]
    x = x.reshape(PH, H//PH, PW, W//PW, -1).squeeze()
    for i in range(PH):
        for j in range(PW):
            xij = Image.fromarray(x[i,:,j])
            xij.save(os.path.join(pathes_dirs[i][j], f))
