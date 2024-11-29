import argparse
import os, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing.pool as mpp
import multiprocessing as mp

from jscv.utils.analyser import label2rgb, set_rgb
import jscv.utils.analyser as Ana


'''
    标签的 值图 (0,1,2,3....ignore_index)
    转换成 RGB图
    
    用法：
    python usefultools/handle_dataset/lable_to_color.py \
        data/gid_water/train/label -o data/gid_water/train/label_color \
            --dataset_name gid_water


'''

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("lable_dir_path", type=Path, help="Path to  lable image folder")
    arg("-o", "--output", type=Path, help="Path to  output rgb image folder")
    arg("--dataset_name", default="gid_water")
    arg("-n", "--num_classes", default='0', help="num_classes, int or 'auto'")
    return parser.parse_args()



palette = None
def get_palette(dataset_name):
    global palette
    if dataset_name == "gid_water":
        from jscv.datasets.gid_water import rgb_dict
        palette = rgb_dict.values()
    elif dataset_name == "vaihingen":
        from jscv.datasets.vaihingen_dataset import rgb_dict
        palette = rgb_dict.values()
    elif dataset_name == "potsdam":
        from jscv.datasets.potsdam_dataset import rgb_dict
        palette = rgb_dict.values()
    elif dataset_name == "loveda":
        from jscv.datasets.loveda_dataset import rgb_dict
        palette = rgb_dict.values()
    elif dataset_name == "cityscapes":
        import jscv.datasets.cityscapes_dataset as cityscapes_dataset
        palette = cityscapes_dataset.palette
    elif dataset_name == "ade20k":
        import jscv.datasets.ade20k as ade20k
        palette = ade20k.palette


auto_num_cls = True
num_classes = 1

def img_writer(input):
    global num_classes, auto_num_cls, palette
    lable_path, output_path = input

    set_rgb(palette)

    lable = cv2.imread(lable_path, cv2.IMREAD_GRAYSCALE)
    if lable is None:
        return

    lable = lable.astype(np.uint8)
    if auto_num_cls:
        num_classes = max(num_classes, np.max(lable) + 1)
    rgb = label2rgb(lable, num_classes)
    cv2.imwrite(output_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    


if __name__ == "__main__":
    args = get_args()

    get_palette(args.dataset_name)
    
    lable_path = args.lable_dir_path
    output_path = args.output
    if output_path is None:
        output_path = Path(str(lable_path) + '_color')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if args.num_classes == '0':
        auto_num_cls = False
        num_classes = len(palette)
        print(f"@ num_classes={num_classes}")
    elif args.num_classes != 'auto':
        auto_num_cls = False
        num_classes = int(args.num_classes)

    data_list = []

    for fname in os.listdir(lable_path):
        Lpath = os.path.join(lable_path, fname)
        data_list.append((Lpath, str(output_path / os.path.splitext(fname)[0]) + ".png"))
        
    print("from:")
    print(lable_path)
    print("to:")
    print(output_path)

    print("begin write!")
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, data_list)
