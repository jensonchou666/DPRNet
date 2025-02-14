#### 一个比较通用的语义分割模型训练的项目，基于自己开发的jscv框架。
主要发布了模型： **DPRNet** 《DPRNet: Dynamic Patch Refinement Network for Water Body Extraction and Ultra-High Resolution Segmentation》
论文后续发表

---


暂时不提供 环境搭建、模型训练等教程 (没时间写)

训练模型的命令： `./easytrain.py config/DPR_WDCI_PVT_L.py -c config/GID_Water_3d5K_60e.py`

请自行阅读源码理解算法。

核心文件：

1. `train.py` (模型训练脚本)
1. `config/DPR_WDCI_PVT_L.py` (论文里用的配置)
1. `config/DPRNet/DPRNet.py` (核心配置文件)
1. `jscv/hr_models/DPRNet/DPRNet.py` (DPRNet的模型结构)
1. `jscv/hr_models/DPRNet/dynamic_patches.py` (动态补丁算法)
