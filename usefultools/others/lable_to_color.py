import argparse
import os, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing.pool as mpp
import multiprocessing as mp

from jscv.utils.analyser import label2rgb, set_rgb


#region 数据集对应的palette(颜色表)
from jscv.datasets.vaihingen_dataset import rgb_dict
palette = rgb_dict.values()
#endregion


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("lable_dir_path", type=Path, help="Path to  lable image folder")
    arg("-o", "--output", type=Path, help="Path to  output rgb image folder")

    arg("-n", "--num_classes", default='0', help="num_classes, int or 'auto'")

    return parser.parse_args()

# LoveDa 7
# ISPRS 6

auto_num_cls = True
num_classes = 1



def img_writer(input):
    global num_classes, auto_num_cls
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
        data_list.append((Lpath, str(output_path / fname)))
    print("begin write!")
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, data_list)
