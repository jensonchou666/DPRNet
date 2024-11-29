
#use:
 #'train.py config/DPRNet/DPRNet_GR101_LR18.py -c config/DPRNet/ex_cuda_over.py ...'

global_patches          = {'train':(4,4), 'val':(2,2)}
local_batch_size        = {'train': 1, 'val': 1}
