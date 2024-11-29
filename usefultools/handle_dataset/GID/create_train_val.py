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

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("dir", type=Path, default="data/gid_water_merged")
    arg("-o", "--out", type=Path, default="data/gid_water")
    arg("--val-nums", type=int, default=30, help="")
    
    arg("--val_list_file", type=Path, help="如果找到该文件, 按文件的列表来划分 train/val")
    
    return parser.parse_args()


args = get_args()

data_dir = args.dir
out_dir = args.out
val_file_nums = args.val_nums
val_list_file = args.val_list_file


print("from:")
print(data_dir)
print("to:")
print(out_dir)

assert osp.exists(data_dir)

images = os.listdir(data_dir / "image")
if val_list_file is None:
    val_img_names = random.sample(images, val_file_nums)
    print("Auto Create Val Dataset")
else:
    with open(val_list_file) as f:
        val_img_names = [s.strip() for s in f.readlines()]
    print("Create Val Dataset from:", val_list_file)

val_mask_names = [osp.splitext(s)[0]+"_label.tif" for s in val_img_names]


print("Train:Val", len(images)-len(val_img_names), len(val_img_names))
print("Val Dataset Images:")
print((val_img_names), '\n')


out_train_dir = out_dir  / 'train'
out_val_dir = out_dir  / 'val'
assert not osp.exists(out_train_dir)
assert not osp.exists(out_val_dir)
os.makedirs(out_val_dir / "image")
os.makedirs(out_val_dir / "label")


for a,b in zip(val_img_names, val_mask_names):
    os.rename(data_dir / "image" / a, out_val_dir / "image" / a)
    os.rename(data_dir / "label" / b, out_val_dir / "label" / b)

with open(out_dir / "val_list.txt", 'w') as f:
    for s in val_img_names:
        f.write(s+"\n")

os.rename(data_dir, out_train_dir)
