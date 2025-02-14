'''
#TODO 测一下速度，如果觉得kernels太多了，考虑用 filter 过掉一些 kernel (sz < 80% batch_patches_max)

#TODO 阈值 动态学习 (无梯度)

#TODO 让大尺寸窗口 stride大于1，减少计算

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time

def get_kernel_dict(h_bigger=True):
    #TODO 替换这种方式，改用优先级
    kernel_dict = {}
    # 上面的就省略了，否则计算量太大
    # 按需要可以减少一些分配, 加快速度
    kernel_dict[49] = [(7,7)]
    kernel_dict[45] = [(9,5), (5,9)]
    kernel_dict[42] = [(7,6), (6,7)]
    kernel_dict[36] = [(6,6)]
    kernel_dict[35] = [(7,5), (5,7)]
    kernel_dict[32] = [(8,4), (4,8)]
    kernel_dict[30] = [(6,5), (5,6)]
    kernel_dict[28] = [(7,4), (4,7)]
    kernel_dict[27] = [(9,3), (3,9)]
    kernel_dict[25] = [(5,5)]
    kernel_dict[24] = [(6,4), (4,6), (8,3), (3,8)]
    kernel_dict[21] = [(7,3), (3,7)]
    kernel_dict[20] = [(5,4), (4,5)] # 2x10是不期望的
    # kernel_dict[18] = [(6,3), (3,6), (9,2), (2,9)]
    kernel_dict[18] = [(6,3), (3,6)]
    kernel_dict[16] = [(4,4), (8,2), (2,8)] # 1x16是不期望的
    kernel_dict[15] = [(5,3), (3,5)]
    kernel_dict[14] = [(7,2), (2,7)]
    kernel_dict[12] = [(4,3), (3,4), (6,2), (2,6)]
    kernel_dict[10] = [(5,2), (2,5)]
    kernel_dict[9] = [(3,3)]
    kernel_dict[8] = [(4,2), (2,4)]
    kernel_dict[6] = [(3,2), (2,3)]
    kernel_dict[5] = [(5,1), (1,5)]
    kernel_dict[4] = [(2,2), (4,1), (1,4)]
    kernel_dict[3] = [(3,1), (1,3)]
    kernel_dict[2] = [(2,1), (1,2)]
    for k, v in kernel_dict.items():
        i = 0
        while i < len(v) - 1:
            if v[i][0] == v[i+1][1]:
                assert v[i][1] == v[i+1][0]
                swap = (h_bigger and v[i][0] < v[i][1]) or \
                    (not h_bigger and v[i][0] > v[i][1])
                if swap:
                    tmp = v[i]
                    v[i] = v[i+1]
                    v[i+1] = tmp
                i += 1
            i += 1

    return kernel_dict

# def get_kernel_dict_full(h_bigger=True):
#     kernel_dict = {}
#     # 上面的就省略了，否则计算量太大
#     # 按需要可以减少一些分配, 加快速度
#     kernel_dict[49] = [(7,7), (7,7)]
#     kernel_dict[45] = [(9,5), (5,9)]
#     kernel_dict[42] = [(7,6), (6,7)]
#     kernel_dict[36] = [(6,6), (6,6)]
#     kernel_dict[30] = [(6,5), (5,6)]
#     kernel_dict[28] = [(7,4), (4,7)]
#     kernel_dict[27] = [(9,3), (3,9)]
#     kernel_dict[25] = [(5,5)]
#     kernel_dict[24] = [(6,4), (4,6), (8,3), (3,8)]
#     kernel_dict[21] = [(7,3), (3,7)]
#     kernel_dict[20] = [(5,4), (4,5)] # 2x10是不期望的
#     kernel_dict[18] = [(6,3), (3,6), (9,2), (2,9)]
#     kernel_dict[16] = [(4,4), (8,2), (2,8)] # 1x16是不期望的
#     kernel_dict[15] = [(5,3), (3,5)]
#     kernel_dict[14] = [(7,2), (2,7)]
#     kernel_dict[12] = [(4,3), (3,4), (6,2), (2,6)]
#     kernel_dict[10] = [(5,2), (2,5)]
#     kernel_dict[9] = [(3,3)]
#     kernel_dict[8] = [(4,2), (2,4)]
#     kernel_dict[6] = [(3,2), (2,3)]
#     kernel_dict[5] = [(5,1), (1,5)]
#     kernel_dict[4] = [(2,2), (2,2), (4,1), (1,4)]
#     kernel_dict[3] = [(3,1), (1,3)]
#     kernel_dict[2] = [(2,1), (1,2)]
#     for k, v in kernel_dict.items():
#         i = 0
#         while i < len(v) - 1:
#             if v[i][0] == v[i+1][1]:
#                 assert v[i][1] == v[i+1][0]
#                 swap = (h_bigger and v[i][0] < v[i][1]) or \
#                     (not h_bigger and v[i][0] > v[i][1])
#                 if swap:
#                     tmp = v[i]
#                     v[i] = v[i+1]
#                     v[i+1] = tmp
#                 i += 1
#             i += 1

#     return kernel_dict




def batch_partition_possible(max_pathes_nums):
    batch_partition_list = []
    i = 2
    while True:
        batch_size = max_pathes_nums // i
        k = max_pathes_nums // (i + 1)
        if batch_size != k:
            batch_partition_list.append((batch_size, (i,i)))
        else:
            break
        i += 1
    while batch_size > 0:
        batch_partition_list.append(
            (batch_size, (max_pathes_nums//(batch_size+1)+1, max_pathes_nums//batch_size)))
        batch_size -= 1
    return reversed(batch_partition_list)


def batch_partition_kernels(max_pathes_nums, kernel_dict: dict =None, h_bigger=True):
    batch_partition_list = batch_partition_possible(max_pathes_nums)
    if kernel_dict is None:
        kernel_dict = get_kernel_dict(h_bigger)
    batch_kernels = []
    for (BZ, (i, j)) in batch_partition_list:
        batch_kernels_BZ = []
        for k in range(j, i-1, -1):
            # print(k, end=' ')
            if k in kernel_dict:
                for v in kernel_dict[k]:
                    batch_kernels_BZ.append(v)
        batch_kernels.append((BZ, (i, j), batch_kernels_BZ))
    return batch_kernels

def demo_batch_partition_possible(max_pathes_nums):
    print("max_pathes_nums:", max_pathes_nums)
    print("possible batch partition:")
    possible_list = batch_partition_possible(max_pathes_nums)
    for i in possible_list:
        print(i)

# print(torch.__version__)


def _create_filt_(nbidx, idx):
    _filt = nbidx[:,idx].view(-1)
    _filt = torch.unique(_filt[_filt != -1], sorted=False)
    return _filt





def boxes_selection(scores: torch.Tensor, kernel_size, threshold:float, indices: torch.Tensor=None):
    '''
        重叠框 消除 算法

        scores: [1,1,H,W] 滑动窗口 处理 pathes_difficulty_tabel(PDT) 后的结果
            表示所有可能的分配的分数
        kernel_size: 平均池化层的kernel大小,    [H维度, W维度]
        threshold: PDT threshold
        indices: [1,1,H,W]   由 torch.arange() 生成   可以不给（速度稍慢）
        
        返回:  boxes [ [y1,y2,x1,x2], score ] 
    '''
    
    mask0 = scores > threshold
    if not mask0.any():
        return []

    scores = scores * mask0
    
    # if kernel_size[0] * kernel_size[1] >= 28:
    #     print(kernel_size)
    #     print(scores)
    
    if scores.dim() == 2:
        scores = scores.unsqueeze(0).unsqueeze(0)

    assert scores.size(0) == 1 and scores.size(1) == 1
    
    if indices is None:
        indices = torch.arange(scores.numel(), device=scores.device)
    indices = indices.view_as(scores).long()

    ''' overlap boxes elimination '''

    k_1, k_2 = kernel_size  # 补丁组尺寸
    k1, k2 = k_1*2-1, k_2*2-1  # 影响域大小

    # 使用 unfold 生成滑动窗口
    neighbors = F.unfold(F.pad(scores, (k2//2, k2//2, k1//2, k1//2), value=-1), (k1, k2), stride=1)
    # neighbors idx
    # nbidx = F.unfold(indices.float(), (k1, k2), stride=1, padding=(k1//2, k2//2)).int()
    nbidx = F.unfold(F.pad(indices.float(), (k2//2, k2//2, k1//2, k1//2), value=-1), (k1, k2), stride=1).long()

    indices = indices.view(-1)
    neighbors = neighbors.squeeze(0)
    nbidx = nbidx.squeeze(0)

    _scores = scores.view(-1)
    idx1 = indices[ _scores > 0]

    max_values, _max_indices = torch.max(neighbors[:,idx1], dim=0)
    neighbors_idx_1 = nbidx[:,idx1]
    max_indices = neighbors_idx_1[_max_indices, torch.arange(neighbors_idx_1.size(1))]

    idx0 = idx1[max_indices == idx1]
    idx1  = idx1[torch.isin(idx1, _create_filt_(nbidx,idx0), invert=True, assume_unique=True)]

    # TODO 用 层次max 替换 sort 的耗时操作
    _, _idx = torch.sort(neighbors[:,idx1], dim=0, descending=True, stable=True)
    nbidx[:,idx1] = torch.gather(nbidx[:,idx1], dim=0, index=_idx)

    idx0 = [idx0]
    # print('#######', torch.concat(idx0), idx1)

    row = 0
    while idx1.numel() != 0:
        row += 1
        # print('????', row, torch.concat(idx0))
        # print(idx1)
        # print(nbidx[row, idx1])
        maskeq = nbidx[row, idx1] == idx1
        eq_idx = idx1[maskeq]
        if eq_idx.numel() == 0: continue
        
        # print('====', eq_idx)
        idx1 = idx1[~maskeq]
        filt = []
        while eq_idx.numel() > 1:
            ''' 处理冲突 '''
            any2 = torch.isin(nbidx[:row, eq_idx], eq_idx).any(dim=0)
            ''' 不可能全为 True '''
            _idx_1 = eq_idx[~any2]
            _idx_2 = eq_idx[any2]
            idx0.append(_idx_1)

            _filt = _create_filt_(nbidx, _idx_1)
            filt.append(_filt)
            eq_idx = _idx_2[torch.isin(_idx_2, _filt, invert=True, assume_unique=True)]

        idx0.append(eq_idx)
        _filt = _create_filt_(nbidx, eq_idx)

        if len(filt) > 0:
            filt.append(_filt)
            _filt = torch.unique(torch.concat(filt))

        idx1  = idx1[torch.isin(idx1, _filt, invert=True, assume_unique=True)]

    W = scores.shape[-1]

    idx0 = torch.concat(idx0)
    ''' 按分数从大到小排序 '''
    _scores, indices = torch.sort(_scores[idx0], descending=True)
    idx0 = idx0[indices]


    y_0 = torch.div(idx0, W, rounding_mode='floor').long()
    x_0 = torch.remainder(idx0, W).long()
    y_1 = y_0 + k_1
    x_1 = x_0 + k_2
    # results = torch.stack([y_0, y_1, x_0, x_1], dim=1)
    # return results, _scores, idx0
    indices = torch.stack([y_0, y_1, x_0, x_1], dim=1)

    return create_boxes(indices, _scores)

def create_boxes(indices: torch.Tensor, scores: torch.Tensor):
    return [(indice.tolist(), round(float(sc), 6)) for indice, sc in zip(indices, scores)]




class PNScoreComputer:
    '''
        Positive-Negative-based Score calculation function

        positive_rate ( for kernel [m,n] ( m>=n ) )
            = A([m,n]) = f(m*n) g(m/n)

        f(x) = k/(x^p)  f(x1)=y1, f(x2)=y2,   如:  f(2)=0.5, f(32)=0.2

        g(x) = 1/(x^p)  g(x3)=y3,   如:  f(4)=0.5

        
        调参    
                增大 y1 -------- 增加小补丁被选中的概率
                增大 y2 -------- 增加大补丁被选中的概率
                减小 y3 -------- 降低 非正方形框 被选中的概率
                
                suggest: 0<y2<y1<1   y3<1
    '''

    def __init__(self, fx_points=[(2, 0.8), (36, 0.2)], gx_point=(4, 0.6)):
        x1, y1 = fx_points[0]
        x2, y2 = fx_points[1]
        x3, y3 = gx_point
        self.p = math.log(y2 / y1) / math.log(x1 / x2)
        self.k = y1 * (x1 ** self.p)
        self.q = -math.log(y3) / math.log(x3)

    def positive_rate(self, kernel_size):
        '''
            kernel_size: [k1, k2]
                m = max(k1,k2), n = min(k1,k2)
            return: k / ( (mn)^p (m/n)^q )
        '''
        k1, k2 = kernel_size
        m, n = max(k1, k2), min(k1, k2)
        rate = self.k / ((m*n)**self.p * (m/n)**self.q)
        if rate <= 0:
            print('rate is negative, please check the parameters of PNScoreComputer.__init__')
            return 0
        return rate


    def compute_scores(self, PDT: torch.Tensor, threshold, kernel_size, stride=1):
        '''
            PDT: patch_difficulty_table [H, W]  确认 值在0~1
            #TODO 增加注释
        '''
        H, W = PDT.shape
        k1, k2 = kernel_size
        positive_rate = self.positive_rate(kernel_size)
        max_v = (1+positive_rate) * threshold
        if max_v < 1:
            mask_positive = PDT > threshold
            PDT = PDT.clone()
            ratio = (PDT[mask_positive] - threshold) / (1 - threshold)
            PDT[mask_positive] = threshold + ratio * (max_v - threshold)
        PDT = PDT.unsqueeze(0).unsqueeze(0)
        unfolded = F.unfold(PDT, kernel_size=kernel_size, stride=stride, padding=0)
        scores = unfolded.mean(dim=1)
        return scores.view(1, 1, (H - k1) // stride + 1, (W - k2) // stride + 1), positive_rate


class DynamicPatchesGrouping:
    
    flag_full = 'full'
    flag_half = 'half'
    flag_1x1 = '1x1'
    flag_inverse = 'inverse'    # [inverse_idx:] 为 inversed kernels

    def __init__(self, 
                 max_pathes_nums, 
                 PDT_size=(16,16),
                 threshold_rate_1x1=0.6, #? 降低 1x1分配 的阈值
                 min_pathes_nums=1, # 补丁数量 在 ?% 以下的分配被跳过 -- 防止补丁数过少的分配 影响速度
                 score_compute_classtype=PNScoreComputer,   #? 计算分数的方法
                 score_compute_kargs={},
                 kernel_dict: dict =None,
                 h_bigger=False,
                 toCUDA=True,
                 do_init=True,
                 extra_mode='default',  # 'default' / '1x1' / 'GRNet'
                #  extra_mode='1x1',  # 'default' / '1x1' / 'GRNet'
                #  extra_mode='GRNet',  # 'default' / '1x1' / 'GRNet'
                 **kargs
                 ):
        self.max_pathes_nums = max_pathes_nums
        self.PDT_size = PDT_size
        self.h_bigger = h_bigger
        self.toCUDA = toCUDA
        self.kernel_dict = kernel_dict
        self.score_computer = score_compute_classtype(**score_compute_kargs)
        self.min_pathes_nums = min_pathes_nums
        self.threshold_rate_1x1 = threshold_rate_1x1
        self.minfloat32 = torch.finfo(torch.float32).min
        self.tmp_boxes = []
        self.time_count = 0
        self.extra_mode = extra_mode
        # self.minfloat32 = -111111
        assert min_pathes_nums < 0.5 * max_pathes_nums
        if do_init:
            self.init()


    def get_size_meanpooled(self, kernel_size):
        H,W = self.PDT_size
        K1,K2 = kernel_size
        return (H-K1+1, W-K2+1)

    def create_batch_groups(self, max_pathes_nums, kernel_dict: dict =None, h_bigger=True, toCUDA=True):
        groups_info = batch_partition_kernels(max_pathes_nums, kernel_dict, h_bigger)
        batch_groups = []
        indices_dict = {}
        for BZ, pathes_num_range, kernels in groups_info:
            items = []
            D = dict(batch_size=BZ, num_range=pathes_num_range, items=items)
            for k in kernels:
                item = {}
                # conv = MeanConv2d(*k)
                M = self.get_size_meanpooled(k)
                m = M[0] * M[1]
                if m not in indices_dict:
                    arange = torch.arange(m, dtype=torch.long)
                    if toCUDA:
                        arange = arange.cuda()
                    indices_dict[m] = arange
                # if toCUDA:
                #     conv = conv.cuda()
                # item['avg'] = conv
                item['kernel'] = k
                item['indices'] = indices_dict[m]
                items.append(item)
            batch_groups.append(D)
        self.batch_groups = batch_groups
        self.indices_dict = indices_dict
        H,W = self.PDT_size
        arange = torch.arange(H*W, dtype=torch.long)
        if toCUDA: 
            arange = arange.cuda()
        self.device = arange.device
        self.indices_dict[H*W] = arange
        self.groups_info = [dict(batch_size=a, num_range=b, kernels=c) for (a,b,c) in groups_info]


    def init(self):
        self.create_batch_groups(self.max_pathes_nums, self.kernel_dict, self.h_bigger, self.toCUDA)

    def clear(self):
        del self.batch_groups
        self.batch_groups = None
        del self.indices_dict
        self.indices_dict = None

    def pdt_allocated(self, PDT:torch.Tensor, boxes:torch.Tensor):
        # boxes 是 [N, 5]
        for box in boxes:
            # print(box)
            y1, y2, x1, x2 = box[0]
            PDT[y1:y2, x1:x2] = self.minfloat32

    def pdt_allocated_temp(self, PDT:torch.Tensor, boxes:torch.Tensor):
        # boxes 是 [N, 5]
        for box in boxes:
            # print(box)
            y1, y2, x1, x2 = box[0]
            self.tmp_boxes.append(([y1, y2, x1, x2], PDT[y1:y2, x1:x2].clone()))
            PDT[y1:y2, x1:x2] = self.minfloat32

    def pdt_allocated_restore(self, PDT:torch.Tensor):
        for box, PDT_value in self.tmp_boxes:
            y1, y2, x1, x2 = box
            PDT[y1:y2, x1:x2] = PDT_value
        self.tmp_boxes = []


    def scan_patches_difficulty_table(self, PDT:torch.Tensor, threshold: float):
        with torch.no_grad():
            time0 = time.time()
            res = self._scan_patches_difficulty_table(PDT, threshold)
            self.time_count += time.time() - time0
        return res

    def _scan_patches_difficulty_table(self, PDT:torch.Tensor, threshold: float):
        '''
            # TODO  

            PDT: [16,16]
            threshold: 0~1
        '''
        # print('threshold: ', threshold) # 0.0370
        # threshold = 0.055
        # threshold = 0.15


        PDT = PDT.clone()
        if self.toCUDA:
            PDT = PDT.cuda()
        assert PDT.dim() == 2

        max_p_num = self.max_pathes_nums
        omit_num = self.min_pathes_nums
        # print('max_pathes_nums', max_p_num, 'omit_rate', omit_rate, 'omit_num', omit_num)

        allocated_groups = []
        def allocate_group(flag, kernel, boxes, PDT_erase=True,**kwargs):
            D = {
                'flag': flag,
                'kernel': kernel,
                'batch_size': len(boxes),
                'boxes': boxes,
                'num_patches': kernel[0]*kernel[1]* len(boxes),
            }
            D.update(kwargs)
            allocated_groups.append(D)
            if PDT_erase:
                self.pdt_allocated(PDT, boxes)

        batch_groups = self.batch_groups
        if self.extra_mode != 'default':
            batch_groups = []

        for i, batchsetting in enumerate(batch_groups):
            batch_size = batchsetting['batch_size']
            items = batchsetting['items']
            len0 = len(items)

            inversed = False
            
            for j, item in enumerate(items):
                assert isinstance(item, dict)
                kernel = item['kernel']
                indices = item['indices']
                ksz = kernel[0]*kernel[1]

                scores, positive_rate = self.score_computer.compute_scores(PDT, threshold, kernel)
                boxes = boxes_selection(scores, kernel, threshold, indices)
                sz = len(boxes)

                if sz == 0 and not inversed:
                    continue

                if not (sz == 0 and inversed):
                    batches_full = sz // batch_size
                    for j in range(batches_full):
                        # print(kernel,j, j*batch_size, (j+1)*batch_size)
                        box = boxes[j*batch_size:(j+1)*batch_size]
                        allocate_group(self.flag_full, kernel, box)
                    rest = sz % batch_size
                    num_rest = rest*ksz
                    if rest == 0: 
                        if inversed:
                            boxes = []
                        else:
                            continue
                    else:
                        boxes = boxes[-rest:]
                if inversed:
                    inversed = False
                    N1, N2 = len(boxes), len(last_boxes)
                    # print("###2", kernel, N1, N2)
                    nums = (N1+N2)*ksz
                    self.pdt_allocated_restore(PDT)
                    # print('!!!!!!!!!', kernel)
                    if nums <= omit_num: continue
                    if nums > max_p_num + omit_num:
                        # print("#########", kernel, last_kernel)
                        allocate_group(self.flag_half, kernel, boxes, positive_rate=positive_rate)
                        allocate_group(self.flag_half, last_kernel, last_boxes, positive_rate=positive_rate)
                        continue
                    if N1 == 0:
                        # print("$$$$$$$$$", last_kernel)
                        allocate_group(self.flag_half, last_kernel, last_boxes, positive_rate=positive_rate)
                        continue
                    if nums > max_p_num:
                        boxes, last_boxes = filter_lists(boxes, last_boxes, max_p_num)
                        N1, N2 = len(boxes), len(last_boxes)
                    if N1 > N2:
                        boxes = boxes + last_boxes
                        N = N1
                        # kernel = kernel
                    else:
                        boxes = last_boxes + boxes
                        N = N2
                        kernel = last_kernel
                    # print("@@@@@@@@@@", kernel)
                    allocate_group(self.flag_inverse, kernel, boxes, inverse_idx=N, positive_rate=positive_rate)
                else:
                    if (j < len0-1):
                        kernel2 = items[j+1]['kernel']
                        if kernel[0] == kernel2[1] and kernel[1] == kernel2[0]:
                            # print('#1', kernel, kernel2)
                            inversed = True
                            last_kernel = kernel
                            last_boxes = boxes
                            self.pdt_allocated_temp(PDT, boxes)
                            continue
                    inversed = False
                    #TODO 方形框的 omit 应该放宽一点
                    if num_rest > omit_num: #- 0.1*max_p_num:
                        allocate_group(self.flag_half, kernel, boxes, positive_rate=positive_rate)

        threshold_1x1 = threshold * self.threshold_rate_1x1

        if self.extra_mode == 'GRNet':
            threshold_1x1 = torch.median(PDT.view(-1))
            # print(threshold_1x1, (PDT > threshold_1x1).sum()/((PDT > 0).sum()))

        ''' 1x1 '''
        H, W = PDT.shape
        PDT = PDT.view(-1)
        mask0 = PDT > threshold_1x1
        if mask0.any():
            #TODO 是否要 排序
            indices = self.indices_dict[H*W][mask0]
            scores = PDT[indices]
            y_0 = torch.div(indices, W, rounding_mode='floor').long()
            x_0 = torch.remainder(indices, W).long()
            indices = torch.stack([y_0, y_0+1, x_0, x_0+1], dim=1)
            boxes = create_boxes(indices, scores)
            batch_size = self.max_pathes_nums
            batches = len(boxes) // batch_size
            for j in range(batches):
                boxes_ = boxes[j*batch_size:(j+1)*batch_size]
                allocate_group(self.flag_1x1, (1,1), boxes_, False)
            rest = len(boxes) % batch_size
            if rest > 0:
                #TODO 是否要过滤最后一个batch的 1x1补丁
                boxes_ = boxes[-rest:]
                allocate_group(self.flag_1x1, (1,1), boxes_, False)

        return allocated_groups

    def crop_images_one_batch(self, images:list, allocated_batch):
        if (allocated_batch['flag'] == self.flag_inverse):
            return self._crop_images_inverse_(images, allocated_batch)
        boxes = allocated_batch['boxes']
        pdtH, pdtW = self.PDT_size
        outputs = []
        for img in images:
            # assert isinstance(img, torch.Tensor)
            H, W = img.shape[-2:]
            pH, pW = H//pdtH, W//pdtW
            cropped = []
            for box in boxes:
                y1, y2, x1, x2 = box[0]
                y1, y2, x1, x2 = y1*pH, y2*pH, x1*pW, x2*pW
                cropped.append(img[...,y1:y2,x1:x2])
            cropped = torch.concat(cropped, dim=0)#.contiguous()
            outputs.append(cropped)
        return outputs

    def _crop_images_inverse_(self, images:list, allocated_batch):
        boxes = allocated_batch['boxes']
        pdtH, pdtW = self.PDT_size
        inverse_idx = allocated_batch['inverse_idx']
        outputs = []
        for img in images:
            # assert isinstance(img, torch.Tensor)
            H, W = img.shape[-2:]
            pH, pW = H//pdtH, W//pdtW
            cropped = []
            for box in boxes[:inverse_idx]:
                y1, y2, x1, x2 = box[0]
                y1, y2, x1, x2 = y1*pH, y2*pH, x1*pW, x2*pW
                cropped.append(img[...,y1:y2,x1:x2])
            for box in boxes[inverse_idx:]:
                y1, y2, x1, x2 = box[0]
                y1, y2, x1, x2 = y1*pH, y2*pH, x1*pW, x2*pW
                ''' 旋转 '''
                cropped.append(torch.rot90(img[...,y1:y2,x1:x2], 1, (-2,-1)))
            cropped = torch.concat(cropped, dim=0)#.contiguous()
            outputs.append(cropped)
        return outputs

    def replace_cropped(self, images:list, cropped, allocated_batch):
        if (allocated_batch['flag'] == self.flag_inverse):
            return self._replace_cropped_inverse_(images, cropped, allocated_batch)
        boxes = allocated_batch['boxes']
        pdtH, pdtW = self.PDT_size
        # outputs = []
        for img, crop in zip(images, cropped):
            # assert isinstance(img, torch.Tensor)
            H, W = img.shape[-2:]
            pH, pW = H//pdtH, W//pdtW
            crop_list = torch.split(crop, img.size(0), 0)
            for box, cp in zip(boxes, crop_list):
                y1, y2, x1, x2 = box[0]
                y1, y2, x1, x2 = y1*pH, y2*pH, x1*pW, x2*pW
                img[...,y1:y2,x1:x2] = cp #.clone()
            # outputs.append(img.contiguous())

    def _replace_cropped_inverse_(self, images:list, cropped, allocated_batch):
        boxes = allocated_batch['boxes']
        pdtH, pdtW = self.PDT_size
        inverse_idx = allocated_batch['inverse_idx']
        for img, crop in zip(images, cropped):
            # assert isinstance(img, torch.Tensor)
            H, W = img.shape[-2:]
            pH, pW = H//pdtH, W//pdtW
            crop_list = torch.split(crop, img.size(0), 0)
            for box, cp in zip(boxes[:inverse_idx], crop_list[:inverse_idx]):
                y1, y2, x1, x2 = box[0]
                y1, y2, x1, x2 = y1*pH, y2*pH, x1*pW, x2*pW
                img[...,y1:y2,x1:x2] = cp #.clone()
            for box, cp in zip(boxes[inverse_idx:], crop_list[inverse_idx:]):
                y1, y2, x1, x2 = box[0]
                y1, y2, x1, x2 = y1*pH, y2*pH, x1*pW, x2*pW
                img[...,y1:y2,x1:x2] = torch.rot90(cp, 3, (-2,-1))

    def analyze_init(self):
        self.num_patches_counter = 0
        self.total_pathes_counter = 0
        self.batches_counter = 0
        self.time_count = 0

    def analyze_once(self, allocated_groups, PDT: torch.Tensor, do_return=False):
        num_patches = 0
        for D in allocated_groups:
            num_patches += D['num_patches']
        total_pathes = PDT.numel()
        batches = len(allocated_groups)
        self.num_patches_counter += num_patches
        self.total_pathes_counter += total_pathes
        self.batches_counter += batches
        if do_return:
            return self.analyze_items(num_patches, total_pathes, batches)

    def analyze_all(self):
        D = {'spend': f'{self.time_count:.2f}s'}
        D.update(self.analyze_items(self.num_patches_counter, self.total_pathes_counter, self.batches_counter))
        return D


    def analyze_items(self, num_patches, total_pathes, batches):
        #TODO 继续统计增加各种 信息
        if batches == 0:
            memory_use_rate = 1
        else:
            memory_use_rate = round(num_patches/(self.max_pathes_nums*batches), 4)
        result = {
            'select_rate': round(num_patches/total_pathes, 4), #? 比例: 待细化补丁数 / PDT总补丁数 
            'memory_use_rate': memory_use_rate, #? 显存利用率， 越大越好
            'num_patches': num_patches,  #? 待细化补丁的总数 ( 以PDT的最小块为单位，即最小补丁 )
            'batches': batches,  #? 细化阶段 依次进行 batches 次, 越小越好
            }
        return result


class ThresholdLearner:
    def __init__(self, select_rate_target=0.25, threshold_init=0.05,
                 base_factor=0.5, adjustment_decay=0.95, window_size=3, 
                 max_adjustment=0.01, positive_relation=False,
                 base_factor_min=0.001, base_factor_max=0.1, convergence_threshold=0.001):
        """
        初始化 ThresholdLearner 类，设置超参数。

        :param select_rate_target: 期望的选择率（target select rate），模型调整目标
        :param threshold_init: 初始的阈值，模型从此开始调整
        :param base_factor: 初始的调整步长因子，控制每次调整的幅度，默认为 0.5。
        :param adjustment_decay: 步长衰减因子，每次调整后，步长会乘以此因子以平滑调整，默认为 0.95。
        :param window_size: 平滑窗口大小，控制历史误差和调整步长的计算范围，默认为 3。
        :param max_adjustment: 最大调整步长，用于限制每次调整的最大幅度，默认为 0.01。
        :param positive_relation: 是否选择正相关模式（阈值增大时，选择率也增大），默认为 False。
        :param base_factor_min: base_factor 的最小值，控制步长最小值，默认为 0.001。
        :param base_factor_max: base_factor 的最大值，控制步长最大值，默认为 0.1。
        :param convergence_threshold: 收敛阈值，当选择率与目标选择率的差异小于此值时，认为已经收敛，默认为 0.001。
        """
        # 初始化超参数
        self.threshold = threshold_init  # 初始阈值
        self.select_rate_target = select_rate_target  # 目标选择率
        self.base_factor = base_factor  # 初始步长
        self.adjustment_decay = adjustment_decay  # 步长衰减因子
        self.window_size = window_size  # 平滑窗口大小
        self.max_adjustment = max_adjustment  # 最大调整步长
        self.positive_relation = positive_relation  # 是否为正相关关系
        self.convergence_threshold = convergence_threshold  # 收敛阈值
        self.base_factor_min = base_factor_min  # base_factor 的最小值
        self.base_factor_max = base_factor_max  # base_factor 的最大值
        
        # 历史记录
        self.history_diff = []  # 记录过去几次的误差差距
        self.history_steps = []  # 记录过去几次的阈值调整步长

    def adjust_threshold(self, select_rate):
        """
        单步调整阈值。

        :param select_rate: 当前阈值下计算得到的选择率。
        :return: 调整后的阈值和选择率差异。
        """
        # 计算选择率与期望值的差距
        diff = select_rate - self.select_rate_target

        # 计算调整步长
        adjustment_factor = self.base_factor * diff

        # 保证步长不会过大
        adjustment_factor = np.clip(adjustment_factor, -self.max_adjustment, self.max_adjustment)

        # 根据关系是否为正来调整阈值
        if self.positive_relation:
            self.threshold -= adjustment_factor
        else:
            self.threshold += adjustment_factor

        # 记录每次的调整误差和步长
        self.history_diff.append(diff)
        self.history_steps.append(adjustment_factor)

        # 计算窗口大小：如果历史数据少于 N，使用可用的最小数据量
        window_size = min(self.window_size, len(self.history_diff))  # 动态窗口大小，最大为 N，最小为当前已有数据量

        # 取最近 `window_size` 次的误差的平均值
        recent_diff_avg = np.mean(np.abs(self.history_diff[-window_size:]))
        # 计算最近 `window_size` 次调整步长的平均值
        recent_steps_avg = np.mean(np.abs(self.history_steps[-window_size:]))

        # 使用误差的平均值和步长的平均值来调整 base_factor
        self.base_factor = recent_diff_avg / (recent_steps_avg + self.base_factor_min)

        # 限制 base_factor 的最大值和最小值
        self.base_factor = np.clip(self.base_factor, self.base_factor_min, self.base_factor_max)

        # 步长衰减：每次调整后逐步减小步长，保持调整平滑
        self.base_factor *= self.adjustment_decay

        return self.threshold, diff  # 返回调整后的阈值和差距


            

def analyze_allocated_once(allocated_groups, manager: DynamicPatchesGrouping, PDT: torch.Tensor):
    '''
        #? 弃用
    '''
    num_patches = 0
    for D in allocated_groups:
        num_patches += D['num_patches']
    
    max_pathes_nums = manager.max_pathes_nums   #? 一个批次最大允许的 最小补丁数量
    total_pathes = PDT.numel()
    batches = len(allocated_groups)
    if batches == 0:
        memory_use_rate = 1
    else:
        memory_use_rate = round(num_patches/(max_pathes_nums*batches), 4)

    result = {'num_patches': num_patches,  #? 待细化补丁的总数 ( 以PDT的最小块为单位，即最小补丁 )
              'select_rate': round(num_patches/total_pathes, 4),    #? 比例: 待细化补丁数 / PDT总补丁数 
              'batches': batches,  #? 细化阶段 依次进行 batches 次, 越小越好
              'memory_use_rate': memory_use_rate, #? 显存利用率， 越大越好
              }
    return result




def confirm_no_overlap(allocated_groups, PDT_size):
    H, W = PDT_size
    Map = torch.zeros(H, W).bool()
    
    index = torch.arange(H*W).view_as(Map)

    for i, D in enumerate(allocated_groups):
        for box in D['boxes']:
            [y1, y2, x1, x2], scores = box
            mask = Map[y1:y2,x1:x2]
            if mask.any():
                print('\n!!! overlap detected, idx:', i, 'kernel:', D['kernel'])
                idx1 = index[y1:y2,x1:x2][mask]
                idx1 = torch.stack([torch.div(idx1, W, rounding_mode='floor').int(), idx1 % W], dim=1)
                print(idx1)
                for j, D2 in enumerate(allocated_groups[:i+3]):
                    print(j, D2)
                return False
            Map[y1:y2, x1:x2] = True
    print('\nSuccessed, no overlap.')
    return True



def filter_lists(x1, x2, N):
    """
    从两个已经排序的列表 x1 和 x2 中移除元素，使得总长度等于 N。
    被移除的元素的 score 必须比保留下来的元素的 score 都低。
    
    Args:
    - x1, x2: List[Tuple[str, float]]，按 score 从大到小排序的列表
    - N: int，总保留的元素数量
    
    Returns:
    - x1_filtered, x2_filtered: 过滤后的两个列表
    """
    # 当前总长度
    L1, L2 = len(x1), len(x2)
    total_length = L1 + L2
    to_remove = total_length - N  # 需要移除的元素数量

    # 双指针操作，从尾部移除元素
    while to_remove > 0:
        # 比较两个列表最后一个元素的 score
        if not x1:  # x1 空了，移除 x2 的元素
            x2.pop()
        elif not x2:  # x2 空了，移除 x1 的元素
            x1.pop()
        elif x1[-1][1] < x2[-1][1]:  # x1 的最后一个元素分数较低
            x1.pop()
        else:  # x2 的最后一个元素分数较低
            x2.pop()
        to_remove -= 1

    return x1, x2




def enhanced_color_transition(x: float, x255=True) -> tuple:
    """
    根据 x 的值 (0~1) 返回高对比度的颜色。
    - 颜色范围：青色、绿色、蓝色到紫色（避免红色）。
    - 输入: x (0 ~ 1)
    - 输出: RGB 值 (0 ~ 1)
    """
    assert 0 <= x <= 1, "x 必须在 [0, 1] 范围内"
    
    # 通过 x 的值划分不同的颜色范围
    if x < 0.33:
        hue = 0.5 + x * 0.3  # 青色(0.5) 到 绿色(0.6)
    elif x < 0.66:
        hue = 0.6 + (x - 0.33) * 0.3  # 绿色(0.6) 到 蓝色(0.7)
    else:
        hue = 0.7 + (x - 0.66) * 0.2  # 蓝色(0.7) 到 紫色(0.8)
    
    saturation = 1.0  # 高饱和度
    value = 1.0       # 高亮度

    # 转换为 RGB
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    # print(r, g, b)
    if x255:
        r, g, b = r*255, g*255, b*255
    return r, g, b

def paint_on_image(image: torch.Tensor, 
                   allocated_groups, 
                   manager:DynamicPatchesGrouping,
                   min_thick_rate = 0.12,
                   max_thick_rate = 0.3,
                   thick_compare_to = 'h',
                   RGB2BGR=False,
                   do_crop=False, #? 把补丁裁剪出来
                   fill_mode=False,
                   ):
    '''
        image: [H,W]灰度图  或 [H,W,3]
    '''
    if image.dim() == 2:
        image = image.unsqueeze(2).repeat(1,1,3)
    H,W,C = image.shape
    image = image.to(manager.device)
    assert C == 3
    pdtH, pdtW = manager.PDT_size
    max_pathes_nums = manager.max_pathes_nums
    pH, pW = H//pdtH, W//pdtW
    thick_compare_to = pH if thick_compare_to == 'h' else pW
    min_thick = min_thick_rate * thick_compare_to
    max_thick = max_thick_rate * thick_compare_to
    def get_thick(rate):
        return int(min_thick + rate * (max_thick - min_thick))
    
    croped_list = []
    
    for i, g in enumerate(allocated_groups):
        kernel = g['kernel']
        boxes = g['boxes']
        rate = float(kernel[0]*kernel[1]) / max_pathes_nums
        thick = get_thick(rate)
        color = enhanced_color_transition(rate)
        if RGB2BGR:
            r,g,b = color
            color = b,g,r
        color = torch.tensor(color, device=manager.device)
        for box in boxes:
            y1, y2, x1, x2 = box[0]
            y1, y2, x1, x2 = y1*pH, y2*pH, x1*pW, x2*pW
            # print(y1, y2, x1, x2, scores)
            if fill_mode:
                image[y1:y2, x1:x2] = color
                thick = max(thick//2, 1)
                image[y1:y1+thick, x1:x2] = 0   # Top
                image[y2-thick:y2, x1:x2] = 0   # Down
                image[y1:y2, x1:x1+thick] = 0   # Left
                image[y1:y2, x2-thick:x2] = 0   # Right
            else:
                image[y1:y1+thick, x1:x2] = color   # Top
                image[y2-thick:y2, x1:x2] = color   # Down
                image[y1:y2, x1:x1+thick] = color   # Left
                image[y1:y2, x2-thick:x2] = color   # Right
            if do_crop:
                croped_list.append(image[y1:y2, x1:x2].clone())
    if do_crop:
        return image, croped_list
    return image



def gray_map_strengthen(gray: np.ndarray):
    gray = gray.astype(np.float64)
    gray = gray - np.min(gray)
    rate = 255.0 / np.max(gray)
    gray = gray * rate
    # print(np.max(gray), np.average(gray))
    gray = gray.astype(np.uint8)
    return gray

def demo_simple():
    PDT_size = (16,16)
    threshold = 0.

    PDT = torch.rand(*PDT_size)

    dynamicPatchesGrouping = DynamicPatchesGrouping(32, PDT_size=PDT_size, kernel_dict=get_kernel_dict(), toCUDA=True)
    # dynamicPatchesGrouping = DynamicPatchesGrouping(16, PDT_size=PDT_size, kernel_dict={16:[(8,2)]}, toCUDA=True)
    for x in dynamicPatchesGrouping.groups_info:
        print(x)
    print()
    
    allocated_groups = dynamicPatchesGrouping.scan_patches_difficulty_table(PDT, threshold)
    
    for i, g in enumerate(allocated_groups):
        print(i, g)

    print('\n', analyze_allocated_once(allocated_groups))

    confirm_no_overlap(allocated_groups, PDT_size)


def auto_generate_pdt(height=32, width=32, continuity_sigma=1.2):
    """
    模拟生成一张包含一定比例大值（0.6以上）的损失图 lossmap（16x16）。
    
    参数：
    - height: 图像高度，默认 16。
    - width: 图像宽度，默认 16。
    - threshold: 大部分值集中在该值以下，默认 0.1。
    - continuity_sigma: 控制空间连续性的高斯滤波参数，值越大连续性越强。
    
    返回：
    - lossmap: 模拟生成的损失图，形状为 [height, width]。
    """
    from scipy.ndimage import gaussian_filter
    # Step 1: 使用伽马分布生成随机值（以保证部分值能较大）
    random_values = np.random.gamma(shape=2.0, scale=0.1, size=(height, width))  # 伽马分布
    random_values = np.clip(random_values, 0, 1)  # 截断到 [0, 1] 范围
    
    # Step 2: 添加空间连续性（高斯平滑）
    lossmap = gaussian_filter(random_values, sigma=continuity_sigma)

    # Step 3: 归一化到 [0, 1]
    lossmap = (lossmap - np.min(lossmap)) / (np.max(lossmap) - np.min(lossmap))
    
    # Step 4: 调整大值的数量（确保有一些大于 0.6 的值）
    # 生成一些随机值大于 0.6
    large_values = np.random.uniform(0.6, 1.0, size=(height, width))
    lossmap[lossmap < 0.6] = np.random.uniform(0, 0.6, size=(lossmap < 0.6).sum())  # 大部分值小于 0.6
    lossmap[lossmap >= 0.6] = large_values[lossmap >= 0.6]  # 少数值大于 0.6

    return lossmap


def demo_paint(path='0-Test/动态补丁测试1(随机PDT)'):
    #TODO 图片生成有问题

    import cv2
    import os
    if not os.path.exists(path):
        os.makedirs(path)

    H, W = 1024, 1024
    # pdtH, pdtW = PDT_size = (16,16)
    pdtH, pdtW = PDT_size = (32,32)
    max_num_patches = 49
    threshold = 0.7

    score_compute_classtype = PNScoreComputer
    score_compute_kargs = dict(
        fx_points=[(2, 0.4), (36, 0.12)],
        gx_point=(4, 0.5)
    )

    min_pathes_nums = int(max_num_patches*0.05)


    # PDT = torch.rand(*PDT_size)
    PDT = torch.tensor(auto_generate_pdt(pdtH, pdtW)).float()

    dynamicPatchesGrouping = DynamicPatchesGrouping(
        max_num_patches, PDT_size=PDT_size,
        score_compute_classtype=score_compute_classtype,
        score_compute_kargs=score_compute_kargs,
        min_pathes_nums=min_pathes_nums,
        kernel_dict=get_kernel_dict(), toCUDA=True)
    
    allocated_groups = dynamicPatchesGrouping.scan_patches_difficulty_table(PDT, threshold)
    
    # for i, g in enumerate(allocated_groups):
    #     print(i, g)

    PDT_image = PDT.reshape(pdtH,1,pdtW,1).repeat(1,H//pdtH, 1, W//pdtW).reshape(H,W)
    PDT_image = torch.tensor(gray_map_strengthen(PDT_image.numpy()))

    map1 = paint_on_image(PDT_image, allocated_groups, dynamicPatchesGrouping)
    
    k = 0
    for ff in os.listdir(path):
        idx = ff.find("@")
        if idx>0 and ff[:idx].isdigit():
            k = max(int(ff[:idx]), k)

    name1 = f"{pdtH}x{pdtW}-N_{min_pathes_nums}_{max_num_patches}-T{threshold}\
-y_{score_compute_kargs['fx_points'][0][1]}\
_{score_compute_kargs['fx_points'][1][1]}\
_{score_compute_kargs['gx_point'][1]}"
    name = f'{k+1}@{name1}'
    i = 0
    fname = path + "/" + name + ".png"
    while os.path.exists(fname):
        i += 1
        fname = path + "/" + name + f"{i}.png"

    cv2.imwrite(fname, cv2.cvtColor(map1.cpu().numpy(), cv2.COLOR_BGR2RGB))
    print(analyze_allocated_once(allocated_groups, dynamicPatchesGrouping, PDT))
    confirm_no_overlap(allocated_groups, PDT_size)


def demo_threshold_learner():
    from scipy.interpolate import interp1d
    # 给定的一些阈值和选择率
    thresholds = np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.1, 0.4])
    select_rates = np.array([0.4241, 0.3583, 0.2836, 0.2359, 0.2104, 0.1915, 0.104])

    # 使用插值来构建 get_select_rate 关系函数
    get_select_rate = interp1d(thresholds, select_rates, kind='cubic', fill_value="extrapolate")

    # 创建 ThresholdLearner 实例
    learner = ThresholdLearner(
        select_rate_target=0.25,  # 期望的选择率
        threshold_init=0.05,  # 初始阈值
        base_factor=0.5,  # 初始步长
        adjustment_decay=0.95,  # 步长衰减因子
        window_size=3,  # 平滑窗口大小
        max_adjustment=0.01,  # 最大调整步长
        positive_relation=False,  # 是否为正相关（选择率随阈值增大而增大）
        convergence_threshold=0.001,  # 设置收敛阈值
        base_factor_min=0.001,  # 设置 base_factor 的最小值
        base_factor_max=0.1  # 设置 base_factor 的最大值
    )

    # 假设每次计算后的选择率是由外部计算得到的
    for iteration in range(100):
        # 使用插值计算当前阈值下的选择率
        select_rate = get_select_rate(learner.threshold)  # 通过插值函数计算选择率
        
        # 调整阈值，并获得差距
        new_threshold, diff = learner.adjust_threshold(select_rate)
        
        # 输出当前阈值调整后的状态
        print(f"Iteration {iteration+1}: Threshold={new_threshold:.4f}, "
              f"Select Rate={select_rate:.4f}, Target={learner.select_rate_target:.4f}, "
              f"Base Factor={learner.base_factor:.4f}, Diff={diff:.4f}")
        
        # 检查是否已收敛
        if abs(diff) < learner.convergence_threshold:
            print(f"Convergence reached at iteration {iteration+1}.")
            break

    # 输出最终的阈值
    print(f"Final threshold: {learner.threshold:.4f}")

if __name__ == "__main__":

    # demo_simple()
    demo_paint()
    # demo_threshold_learner()







    # demo_batch_partition_possible(13)
    # demo_batch_partition_possible(25)
    # demo_batch_partition_possible(32)
    # demo_batch_partition_possible(54)
    # demo_batch_partition_possible(16)
    
    # for x in get_kernel_dict(True).items():
    #     print(x)
    # print()
    # for x in get_kernel_dict(False).items():
    #     print(x)

    # batch_kernels = batch_partition_kernels(16)
    # for x in batch_kernels:
    #     print(x)


