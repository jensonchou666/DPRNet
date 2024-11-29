#! /opt/conda/bin/python


import argparse
import os, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing.pool as mpp
import multiprocessing as mp

from jscv.utils.analyser import label2rgb


'''
    统计数据集里类别个数
'''

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("dir_path", type=Path, help="Path to image folder")
    arg('-s', "--test-shape", default=False, action="store_true", help='test shape')
    arg('-t', "--traverse", default=False, action="store_true", help='traverse floder')
    arg('-c', "--compare", default=False, action="store_true", help='if -s, show(min, max shape)')

    return parser.parse_args()

# LoveDa 7
# ISPRS 6

classes = set()
shape_h = set()
shape_w = set()
shapes = []

max_h, max_w, min_h, min_w = 0, 0, 11111, 11111
def compare(h, w):
    global max_h, max_w, min_h, min_w
    ret = False
    if h > max_h:
        ret = True
        max_h = h
    if h < min_h:
        ret = True
        min_h = h
    if w > max_w:
        ret = True
        max_w = w
    if w < min_w:
        ret = True
        min_w = w
    return ret
    # max_h = max(max_h, h)
    # max_w = max(max_w, w)
    # min_h = min(min_h, h)
    # min_w = min(min_w, w)

if __name__ == "__main__":
    args = get_args()

    lable_path = args.dir_path

    test_shape = args.test_shape

    #img_suffix = ['png', 'tif', 'jpg']

    def func(fpath):
        #Lpath = os.path.join(lable_path, fname)
        img = cv2.imread(fpath)
        if img is None:
            return

        if not test_shape:
            items = 1
            for shp in img.shape:
                items = items * shp
            a = 1
            for v in img.reshape((items)):
                if v not in classes:
                    classes.add(v)
                    print(classes)
        elif args.compare:
            h, w = img.shape[0], img.shape[1]
            shapes.append((h, w))
            if compare(h, w):
                print(f'min_h {min_h}, min_w {min_w}, max_h {max_h}, max_w {max_w}')
        else:
            h, w = img.shape[0], img.shape[1]
            _add = False
            if h not in shape_h:
                shape_h.add(h)
                _add = True
            if w not in shape_w:
                shape_w.add(w)
                _add = True
            if _add:
                shapes.append((h, w))
                print([h, w])

    if args.traverse:
        for root, dir, files in os.walk(lable_path):
            for f in files:
                func(os.path.join(root, f))
    else:
        for fname in os.listdir(lable_path):
            func(os.path.join(lable_path, fname))

    #计算平均 h, w
    count = 0
    ch, cw = 0, 0
    meanhw = []

    def _compute_():
        global ch, cw, count, meanhw
        if count == 0:
            return
        meanhw.append((float(ch) / count, float(cw) / count))
        count = 0
        ch, cw = 0, 0

    for h, w in shapes:
        ch += h
        cw += w
        count += 1
        if count == 100:
            _compute_()
    _compute_()

    for h, w in meanhw:
        ch += h
        cw += w
    meanhw = (float(ch) / len(meanhw), float(cw) / len(meanhw))
    print(f'mean shape: {meanhw}')