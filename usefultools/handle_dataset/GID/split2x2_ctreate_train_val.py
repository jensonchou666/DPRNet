'''

Useage:
python usefultools/handle_dataset/GID/create_train_val.py data/gid_water_merged \
    --out data/gid_water --val_list_file data/gid_water/val_list.txt


-----------------------------------

Linux OS

in:    
    -data_dir
        -image
        -label
    
out:
    -out_dir
        val_list.txt
        -train
            -image
            -label
        -val
            -image
            -label
'''


import os
import os.path as osp
import random
import argparse
from pathlib import Path
from datetime import datetime
import time
import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--dir", type=Path, default="data/gid_water_merged")
    arg("-o", "--out", type=Path, default="data/GID_Water_3d5K")
    arg("--val-ratio", type=float, default=0.25, help="")
    arg("--max-subfigures", type=int, default=4, help="防止(x+1)张子图同时出现在了验证集里, 算法有一点问题")
    # arg("--vallist-path", type=Path, help="如果找到该文件, 按文件的列表来划分 train/val, \
    #     找不到，划分后将划分保存至该文件， 请不要忽略此选项，否则你要手动去Stdout找划分")

    # arg("--vallist-path", type=Path, default="data/GID_Water_3d5K/val_list_2024-12-12 20:06:17.txt", 
    #     help="如果指定，将按文件的列表来划分 train/val, \
    #     如果不指定， 将会自动划分")
    arg("--vallist-path", type=Path, help="如果指定，将按文件的列表来划分 train/val, \
        如果不指定， 将会自动划分")

    arg("--not-create", default=True, action='store_true')

    return parser.parse_args()


args = get_args()

data_dir = args.dir
out_dir = args.out
val_ratio = args.val_ratio
vallist_path = args.vallist_path
subfigures_N = args.max_subfigures + 1
delayTime = 4

print("from:")
print(data_dir)
print("to:")
print(out_dir)


assert osp.exists(data_dir)


images = os.listdir(data_dir / "image")

images_patches = []
for im in images:
    images_patches.append((im, 0))
    images_patches.append((im, 1))
    images_patches.append((im, 2))
    images_patches.append((im, 3))



val_dict = {}

if vallist_path is not None:
    if not os.path.exists(vallist_path):
        for i in range(delayTime):
            print(f"\r文件不存在，将自动划分({delayTime-i})", end='', flush=True)
            time.sleep(1)
        vallist_path = None

if vallist_path is None:
    val_nums = int(val_ratio * len(images_patches))
    
    val_list = random.sample(images_patches, val_nums)
    train_list = [item for item in images_patches if item not in val_list]

    # val_dict = {}
    # for im, idx in val_list:
    #     if im not in val_dict:
    #         val_dict[im] = []
    #     val_dict[im].append(idx)
    # 
    # for key, val in val_dict.items():
    #     print(f"{key}: {val}")
    # print(len(val_list))
    # print()
    # print('防止4张子图同时出现在了验证集')
    # print('--------------------------------------------')


    D1 =  {}
    '''  防止4张子图同时出现在了验证集 '''
    for _i, (im, idx) in enumerate(val_list):
        if im not in D1:
            D1[im] = 1
        else:
            D1[im] += 1
            if D1[im] >= subfigures_N:
                while True:
                    _j = random.randint(0, len(train_list) - 1)
                    im_train, idx_train = train_list[_j]
                    print(f"val集里的 {im}-{idx} 与 {im_train}-{idx_train} 交换")
                    if im == im_train:
                        print(f"冲突，重新选择")
                        continue
                    else:
                        train_list[_j] = (im, idx)
                        val_list[_i] = (im_train, idx_train)
                        D1[im] -= 1
                        break
    for im, idx in val_list:
        if im not in val_dict:
            val_dict[im] = []
        val_dict[im].append(idx)
    val_dict = {key: sorted(value) for key, value in val_dict.items()}
    count = 0
    for key, val in val_dict.items():
        print(f"{key}: {val}")
        count += len(val)
    print('val solo images: ', len(val_dict.keys()))
    
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    fname = f"val_list_{time_str}.txt"

    with open(fname, "w") as f:
        for key, val in val_dict.items():
            f.write(f"{key}: {val}\n")
    print(f"划分结果已保存在 {fname}")
else:
    with open(vallist_path) as f:
        lines = f.readlines()
        for line in lines:
            im, idxes = line.strip().split(":")
            idxes = idxes.strip()[1:-1].split(",")
            idxes = [int(si.strip()) for si in idxes]
            val_dict[im] = idxes
            
        count = 0
        for key, val in val_dict.items():
            print(f"{key}: {val}")
            count += len(val)
        print('val solo images: ', len(val_dict.keys()))

print('train:val ', len(images_patches)-count, ":", count)
 
if args.not_create:
    exit()

out_train_dir = out_dir  / 'train'
out_val_dir = out_dir  / 'val'
assert not osp.exists(out_train_dir)
assert not osp.exists(out_val_dir)
os.makedirs(out_train_dir / "image")
os.makedirs(out_train_dir / "label")
os.makedirs(out_val_dir / "image")
os.makedirs(out_val_dir / "label")

from PIL import Image
import numpy as np
import cv2

for img in tqdm.tqdm(images):
    img_path = data_dir / "image" / img
    name, ext = os.path.splitext(img)
    label = name + '_label.tif'
    label_path = data_dir / "label" / label
    assert os.path.exists(label_path)
    # print(img_path, label_path)

    img_numpy = np.array(Image.open(img_path))
    label_numpy = np.array(Image.open(label_path))
    
    H, W, C = img_numpy.shape
    img_numpy = img_numpy.reshape(2, H // 2, 2, W // 2, C).transpose(0, 2, 1, 3, 4)
    label_numpy = label_numpy.reshape(2, H // 2, 2, W // 2).transpose(0, 2, 1, 3)
    # print("变换后形状:", img_numpy.shape, label_numpy.shape)
    
    for i in range(2):
        for j in range(2):
            k = i*2 + j
            is_train = True
            
            img_ij = img_numpy[i, j]
            label_ij = label_numpy[i, j]
            # print(f"img_ij shape: {img_ij.shape}, label_ij shape: {label_ij.shape}")
            if img in val_dict and k in val_dict[img]:
                is_train = False

            if is_train:
                cv2.imwrite(str(out_train_dir / "image" / f"{name}_{i}_{j}{ext}"), img_ij)
                cv2.imwrite(str(out_train_dir / "label" / f"{name}_{i}_{j}_label{ext}"), label_ij)
            else:
                cv2.imwrite(str(out_val_dir / "image" / f"{name}_{i}_{j}{ext}"), img_ij)
                cv2.imwrite(str(out_val_dir / "label" / f"{name}_{i}_{j}_label{ext}"), label_ij)
