import os
from PIL import Image
import numpy as np
import tqdm
import argparse
from pathlib import Path

'''
Useage:

python usefultools/handle_dataset/GID/merge_GID2.py \
    --image water_seg/image_1024 --label water_seg/label_1024 \
        --out data/gid_water_merged


'''

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image", type=Path, default="data/water_seg/image_1024")
    arg("-g", "--label", type=Path, default="data/water_seg/label_1024")
    arg("-o", "--out", type=Path, default="data/gid_water_merged")
    arg("--not-ignore-bottom", default=False, action="store_true")
    arg("--bottom-ignore-y", type=int, default=6798)
    

    return parser.parse_args()



args = get_args()

img_path = args.image
label_path = args.label
out_path = args.out
name_dict = {}


print("from:")
print(img_path)
print(label_path)
print("to:")
print(out_path)



def reverse_lebel_name(f):
    f = os.path.splitext(f)[0]
    a = f.find("-MSS")
    s1, s2 = f[:a+5], f[a+5:]

    if not s2.startswith("_"):
        raise NameError(f"name wrong!: {f}")
    
    if s2.startswith("_label"):
        s3 = s2[6:] + "_label"
    elif s2.endswith("_label"):
        s3 = "_label" + s2[:-6]
    else:
        raise NameError(f"name wrong!: {f}")
    return s1 + s3


for f in os.listdir(label_path):
    f1 = reverse_lebel_name(f)

    i = f1.find("_label_")
    name = f1[:i]
    idx = f1[i+len("_label_"):]
    if name not in name_dict:
        name_dict[name] = [idx]
    else:
        name_dict[name].append(idx)

def imgfile(name, idx):
    return os.path.join(img_path, name + "_" + idx + ".tif")

def labelfile(name, idx):
    return os.path.join(label_path, name + "_" + idx + "_label" +".tif")


for name, idx_list in name_dict.items():
    for idx in idx_list:
        imgf = imgfile(name, idx)
        labelf = labelfile(name, idx)
        if not os.path.exists(imgf):
            raise Exception(imgf, "not exists")
        if not os.path.exists(labelf):
            raise Exception(labelf, "not exists")

out_path_image = os.path.join(out_path, "image")
out_path_label = os.path.join(out_path, "label")

if not os.path.exists(out_path_image):
    os.makedirs(out_path_image)
if not os.path.exists(out_path_label):
    os.makedirs(out_path_label)


for (name, idx_list) in tqdm.tqdm(name_dict.items()):
    mx, my = 0, 0
    for idx in idx_list:
        x, y = idx.split("_")
        x, y = int(x), int(y)
        mx, my = max(mx, x), max(my, y)


    img_total, label_total = None, None

    for x in range(mx+1):
        
        img0, label0 = None, None

        for y in range(my+1):
            idx = f"{x}_{y}"
            imgf = imgfile(name, idx)
            labelf = labelfile(name, idx)
            
            imgi = np.array(Image.open(imgf))
            labeli = np.array(Image.open(labelf))
            
            
            if img0 is None:
                img0, label0 = imgi, labeli
            else:
                img0 = np.concatenate((img0, imgi), axis = 1) # 横向拼接
                label0 = np.concatenate((label0, labeli), axis = 1) # 横向拼接
                del imgi, labeli
            
        if img_total is None:
            img_total, label_total = img0, label0
        else:
            img_total = np.concatenate((img_total, img0), axis = 0)
            label_total = np.concatenate((label_total, label0), axis = 0)
            del img0, label0
    
    # print(img_total.shape, label_total.shape)

    img = Image.fromarray(img_total)
    label = Image.fromarray(label_total)
    
    
    img.save(os.path.join(out_path_image, name + ".tif"))
    label.save(os.path.join(out_path_label, name + "_label.tif"))


if not args.not_ignore_bottom:
    N = args.bottom_ignore_y
    print("\n2nd. Set the label of the bottom pixels to ignore_index...")
    print("bottom_ignore_y:", N)

    label_dir = out_path_label

    for f in tqdm.tqdm(os.listdir(label_dir)):
        x = (np.array(Image.open(os.path.join(label_dir, f))))
        H,W = x.shape
        x[N:,:] = 100
        x = Image.fromarray(x)
        x.save(os.path.join(label_dir, f))
