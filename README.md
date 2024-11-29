# Introduction

---

Difficult Patch Refinement Network (**DPRNet**) is an
ultra-high-resolution model for binary water body segmentation.
Paper will be published soon.

It is based on a deep learning framework I developed personally: **jscv**,
which is mainly used for semantic segmentation training.


[//]: # (## Major Features)

[//]: # ()
[//]: # (---)


## Main directories and files

---

```Tree
.
|-- config
|   |-- DPRNet
|   |   |-- DPRNet_GR101_LR18.py    (main config)
|   |   `-- DPRNet_pvt_LR18.py      (replace backbone)
|   |-- datasets
|   |   `-- GID_Water.py            (dataset config)
|   `-- repair_dataset
|       `-- ex_PTDR.py              (dataset repair config)
|   |-- _base_
|   |-- Down4x                      (Downsample Models)
|-- data
|   `-- gid_water                   (Gaofen-2 water-seg)
|       |-- train
|       |   |-- image               (train-images)
|       |   `-- label               (train-label)
|       `-- val
|           |-- image               (val-images)
|           `-- label               (val-label)
|-- jscv
|   |-- backbone                    (verious backbone)
|   |-- datasets                    (dataset base code)
|   |   `-- gid_water.py
|   |-- hr_models
|   |   |-- base_models.py
|   |   `-- pathes_segmentor.py     (Normal UHR model)
|   |   |-- DPRNet.py            <--- DPRNet model code
|   |-- losses
|   `-- utils
|       |-- load_checkpoint.py
|       |-- logger.py               (Draw indicator table)
|       `-- trainer.py              (Trainer,Evaluator,SaveCkpt)
|-- usefultools
|   `-- handle_dataset
|       `-- GID                     (Processing Gaofen-2)
|-- work_dir
    `-- GID_Water
        `-- DPRNet_pvt_LR18-e80
            `-- version_0           (Save Training files)
|-- pretrain
|-- easytrain.sh
`-- train.py
```


## Pretrained Weights

---

resnet18: <https://download.pytorch.org/models/resnet18-5c106cde.pth>

resnet101: <https://download.pytorch.org/models/resnet101-63fe2227.pth>

#### Not necessary:

pvt_v2_b1.pth <https://github.com/whai362/PVTv2-Seg/blob/master/configs/sem_fpn/PVTv2/fpn_pvtv2_b1_ade20k_40k.py>

focalnet_small_lrf.pth <https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_small_lrf.pth>

Put them in directory: `./pretrain/`

## Prepare Dataset

---

1. Download a Gaofen-2 water dataset from here (already labeled, only contains water and land 2 classes)
   <https://aistudio.baidu.com/datasetdetail/104157>
2. Unzip as directory: `./water_seg`
3. These are divided into small 1024×1024 pictures.
   The direct segmentation does not perform well.
   We need to process them.

Merge to 7168×7168
```aiignore
python usefultools/handle_dataset/GID/merge_GID2.py \
    --image water_seg/image_1024 --label water_seg/label_1024 \
        --out data/gid_water_merged
```

Split Dataset, Train : Val (120:30)
```aiignore
python usefultools/handle_dataset/GID/create_train_val.py data/gid_water_merged \
    --out data/gid_water --val_list_file data/gid_water/val_list.txt
```


## Build Environment

---

1. In Linux OS. Create anaconda environment: 
```aiignore
conda create -n DPRNet python=3.7
```
2. `conda activate DPRNet`
3. Install torch, torchvision:
```
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
torchaudio==0.10.2 -f https://download.pytorch.org/whl/torch_stable.html
```
Other versions should also work. Domestic mirror: `-i https://pypi.tuna.tsinghua.edu.cn/simple`

4. ` pip install -r requirements.txt `

Ignore (no need): ` python ./setup.py install`



## Train

---

```aiignore
python train.py config/DPRNet/DPRNet_GR101_LR18.py -c config/datasets/GID_Water.py
```

Work Dir is `./work_dir/GID_Water/DPRNet_GR101_LR18-e80/version_0`

Evaluation matrix per epoch is in : `{Work Dir}/log-DPRNet_GR101_LR18-GID_Water.txt`

The printed log file is: `{Work Dir}/outputs.log`

Trained weight and extra-data in: `{Work Dir}/epoch=0@val_mIoU=?@....ckpt`

Images saved in: `{Work Dir}/2024-?-?-?-?-?`

---

Resume training (see `-r` detail description):
```aiignore
python train.py config/DPRNet/DPRNet_GR101_LR18.py -c config/datasets/GID_Water.py -r 0/{xxx}
```

---

Put it in background, print and error are in `./work_dir/STD_LOG/last-{GPU-ID}.txt`

```aiignore
./easytrain.sh @ {GPU-ID} {cfg} -c {cfg} {other_args} &
```

### Another training mode (recommend)

In the previous training, only G-Branch was training in
the first 10 epochs because L-Branch depends on it.

1. First, train a normal downsampling model for dozens of epochs.

```aiignore
python train.py config/Down4x/deeplab101_d4.py -c config/datasets/GID_Water.py
```
Suppose your trained weight file (Just few epochs) is:
`work_dir/GID_Water/deeplab101_d4-e80/version_0/epoch=21@val_mIoU=88.93@deeplab101_d4-GID_Water.ckpt`

2. Write this path to `config/DPRNet/DPRNet_GR101_LR18.py` on:

   `global_net_ckpt_dict['stage1_GNet_pretrained'] = {ckpt path}`

3. Set `global_net_ckpt_from = 'stage1_GNet_pretrained'`

Now you can re-execute the previous command, and the model will start training from the pre-trained G-Branch.

## PTDR dataset repair

---

Please train for dozens of epoches before continuing.


PTDR usage: After `-c`, add: `config/repair_dataset/ex_pred_stat.py`.

Dataset must use `GID_Water_no_aug.py`.  add `-r {well-trained-ckpt}`, Eg.

```aiignore
python train.py config/DPRNet/DPRNet_GR101_LR18.py \
    -c config/datasets/GID_Water_no_aug.py \
        config/repair_dataset/ex_pred_stat.py -r ?/?
```
```aiignore
#python train.py config/DPRNet/DPRNet_GR101_LR18.py \
#    -c config/datasets/GID_Water_no_aug.py \            # Must no augument
#        config/repair_dataset/ex_pred_stat.py -r ?/?    # Must resume
```
Well-trained-ckpt: `-r version_0/epoch=71@val_mIoU=89.**@**.ckpt`

Results are demonstrated in: `data/gid_water_train_repaired/debug`, you can
remove it.

Corrected labels are in: `data/gid_water_train_repaired/refined`.

Do:
```aiignore
mv -r data/gid_water_train_repaired/refined data/gid_water/train/label_repaired
```

In future model training, you can use this corrected labels. Eg.

```aiignore
python train.py {model} -c config/datasets/GID_Water_repaired.py
```


## Problem solved:

---
To be continued...

## Tools:

---
To be continued...