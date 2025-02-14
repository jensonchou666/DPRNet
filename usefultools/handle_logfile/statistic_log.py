import os

'''
统计log-model.txt的指标

'''

# log_file = 'work_dir/GID_Water/1_globalnet_r101_16x16_r18_stage3-e80/final/log-1_globalnet_r101_16x16_r18_stage3-GID_Water.txt'
# log_file = 'work_dir/GID_Water/dbnet_Gr101_Lr18-e120/a1/log-dbnet_Gr101_Lr18-GID_Water.txt'
# log_file = 'work_dir/GID_Water/01_globalnet_r101_aspp_16x16_r18_stage3-e120/v1/log-1_globalnet_r101_aspp_16x16_r18_stage3-GID_Water.txt'


log_file = 'work_dir/GID_Water/dbnet_Gr101_Lr18-e120/a1/log-dbnet_Gr101_Lr18-GID_Water.txt'
log_file = 'work_dir/GID_Water/1_globalnet_r101_16x16_r18_stage3-e80/final/log-1_globalnet_r101_16x16_r18_stage3-GID_Water.txt'
# log_file = ''
# log_file = 'work_dir/GID_Water/0_DPRNet_GR101_LR18_16x16_stage3-e80/version_0/log-0_DPRNet_GR101_LR18_16x16_stage3-GID_Water.txt'


# log_file = 'work_dir/GID_Water/1_globalnet_r101_aspp_16x16_r18_stage3_no_ffm-e120/version_2/log-1_globalnet_r101_aspp_16x16_r18_stage3_no_ffm-GID_Water.txt'
# log_file = 'work_dir/GID_Water/a/1 copy.txt'
# log_file = 'work_dir/GID_Water/a/012 copy.txt'
log_file = 'work_dir/GID_Water/DPRNet_GR101_LR18_16x16_stage3_fuse_GLType-e80/version_2/log-DPRNet_GR101_LR18_16x16_stage3_fuse_GLType-GID_Water.txt'

topk = 10
max_epoch = 80



print(log_file)
print('max_epoch:', max_epoch, ', topk:', topk)
sort_as = 'max'


void_value = '--'
spliter = '|'
title_margin = 1



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def str_lines(Ls):
    s = ''
    for L in Ls:
        for t in L:
            if is_number(t):
                t = round(float(t), 4)
                if int(t) == t:
                    t = int(t)
            s += f'{t:<12} '
        s += '\n'
    return s


def takeSecond(elem):
    return elem[1]

def step():
    global monitor, log_file, topk, max_epoch, sort_as
    titles = []
    values = []
    lines = [s.strip() for s in open(log_file).readlines()]

    for i, line in enumerate(reversed(lines)):
        if 'epoch' in line and 'stage' in line:
            break
    start_i = len(lines) - i - 1
    t = lines[start_i].split(spliter)

    for s in t:
        s = s.strip()
        if len(s) > 0:
            titles.append(s)


    for i in range(start_i+title_margin+1, len(lines)):
        t = lines[i].split(spliter)
        v = []
        for s in t:
            s = s.strip()
            if len(s) > 0:
                v.append(s)
        if not is_number(v[0]):
            break
        values.append((int(v[0]), v))

    midx = 1000
    s_titles = ''
    for i, t in enumerate(titles):
        if t == monitor:
            midx = i
        s_titles += f'{t:<12} '


    L1 = []

    for e, v in values:
        if e > max_epoch:
            break
        if v[midx] != void_value:
            L1.append((e, float(v[midx]), v))



    L1.sort(key=takeSecond, reverse=(sort_as=='max'))

    topv = []
    L2 = []


    for i, (e,v,v2) in enumerate(L1):
        if i >= topk:
            break
        topv.append(v)
        L2.append(v2)


    # print(s_titles)
    # print(str_lines([v for _,_, v in L1]))

    # print(f'Top{topk}-{monitor}')
    # print(s_titles)
    # print(str_lines(L2))

    # s = ''
    # for v in topv:
    #     s += f'{v:.4f}, '
    # print(s[:-2], '\n')

    print(f'AvgT{topk}-{monitor}: {(sum(topv)/len(topv)):.4f}')



monitor = 'train_mIoU'
step()
monitor = 'train_F1'
step()
monitor = 'train_OA'
step()

monitor = 'val_mIoU'
step()
monitor = 'val_F1'
step()
monitor = 'val_OA'
step()
print("-" * 60)
print()