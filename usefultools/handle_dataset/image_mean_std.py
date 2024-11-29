import torch
import os
from time import time

import argparse
from pathlib import Path
from tqdm import tqdm
import cv2


"""
    遍历数据集，计算所有图片的平均值和方差
"""

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("images_dir", type=Path, help="Path to images folder")
    arg('-t', "--traverse", default=False, action="store_true", help='traverse floder')

    arg('-a', "--accurate", default=False, action="store_true", help='accurate float')
    arg("--no-div", default=False, action="store_true", help='do not div 255')
    arg('-k', "--keep-history", default=False, action="store_true", help='do not clear screan')


    return parser.parse_args()

t0 = time()
mean_sum, std_sum, num_imgs = torch.zeros(3), torch.zeros(3), 0
print_end = '\n'

def func(img_path, i, all, div=255, accurate=True):
    global mean_sum, std_sum, num_imgs, print_end, t0
    im = cv2.imread(img_path)
    if im is None:
        print(f'\ncan not read {img_path}')
        return

    img = torch.from_numpy(im).float() / div
    mean_sum += torch.mean(img, dim=[0, 1])
    std_sum += torch.std(img, dim=[0, 1])
    num_imgs += 1
    mean = mean_sum / num_imgs
    std = std_sum / num_imgs

    t = int(time() - t0)
    m = t // 60
    s = t % 60
    if m > 0:
        st = f'{m}m{s}s'
    else:
        st = f'{s}s'

    if accurate:
        mean = [x.item() for x in mean]
        std = [x.item() for x in std]
        s = f'{i}/{all}'
        print(f'{s:<10} mean:', mean, '\n', f'{st:<10} std:', std, end=print_end)
    else:
        print(f'{i}/{all} {st} ', '  mean:', mean, '  std:', std, end=print_end)


if __name__ == "__main__":
    args = get_args()
    images_dir = args.images_dir
    accurate = args.accurate
    div = 255
    if args.no_div:
        div = 1
    if not args.keep_history:
        print_end = '\r'

    if args.traverse:
        all_mean_sum, all_std_sum = torch.zeros(3), torch.zeros(3)
        all_num_imgs = 0
        for root, dir, files in os.walk(images_dir):
            allsz = len(files)
            if allsz > 0:
                print(f"\nIn {root}")

            for i, f in enumerate(files):
                func(os.path.join(root, f), i + 1, allsz, div, accurate)
            
            if allsz > 0:
                all_mean_sum += mean_sum
                all_std_sum += std_sum
                all_num_imgs += num_imgs
                mean_sum = torch.zeros(3)
                std_sum = torch.zeros(3)
                num_imgs = 0
            
        mean_sum, std_sum = all_mean_sum, all_std_sum
        num_imgs = all_num_imgs
            

    else:
        files = os.listdir(images_dir)
        allsz = len(files)
        for i, fname in enumerate(files):
            func(os.path.join(images_dir, fname), i + 1, allsz, div, accurate)

    print("\nResult:")
    mean = mean_sum / num_imgs
    std = std_sum / num_imgs
    print('mean:', mean, '  std:', std)
    print('Accurate:')
    mean = [x.item() for x in mean]
    std = [x.item() for x in std]
    print('mean:', mean)
    print('std:', std)