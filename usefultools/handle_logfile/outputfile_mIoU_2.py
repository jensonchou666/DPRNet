import os

output_file = 'work_dir/GID_Water/1_globalnet_r101_16x16_r18_stage3-e80/final/outputs.log'  #15e
output_file = 'work_dir/GID_Water/1_globalnet_r101_aspp_16x16_r18_stage3-e120/final/outputs.log' #25e

# output_file = 'work_dir/GID_Water/1_globalnet_r101_16x16_r50_stage3-e80/version_1/outputs.log' #  max 43
# output_file = 'work_dir/GID_Water/1_globalnet_r101_aspp_16x16_r50_stage3-e120/v1/outputs.log' #  max 120

# output_file = 'work_dir/GID_Water/0_DPRNet_pvt_LR18_16x16_stage3-e80/version_1/outputs.log'

# dataset refine
# output_file = 'work_dir/GID_Water/0_DPRNet_GR101_LR18_16x16_stage3-e80/version_0/outputs.log'


topk = 10

epochs = 100

stage = 'val'
# stage = 'train'
sort_key = 'cmIoU'
sort_key = 'rmIoU'


print(output_file)
lines = [s.strip() for s in open(output_file).readlines()]

datas = []


epoach_flag = False
on_key_coarse_pred = False
on_key_pred = False
for i, line in enumerate((lines)):
    if f"'stage': '{stage}'" in line:
        d = {}
        a = line.find("'epoch':")
        b = line.find(", 'stage'")
        d['epoch'] = int(line[a+len("'epoch':"):b].strip())
        datas.append(d)
        epoach_flag = True
    
    if epoach_flag:
        d = datas[-1]
        line = line.strip()
        if line == 'Segment Result, key=coarse_pred':
            on_key_coarse_pred = True
        elif line == 'Segment Result, key=pred':
            on_key_pred = True
        else:
            if on_key_coarse_pred and 'mIoU: ' in line:
                on_key_coarse_pred = False
                a, b, c = line.find("mIoU:"), line.find(", F1:"),  line.find(", OA:")
                d['cmIoU'] = round(float(line[len('mIoU:'):b].strip()),2)
                d['cF1'] = round(float(line[b+len(', F1:'):c].strip()),2)
                d['cOA'] = round(float(line[c+len(', OA:'):].strip()),2)
            elif on_key_pred and 'mIoU: ' in line:
                on_key_pred = False
                a, b, c = line.find("mIoU:"), line.find(", F1:"),  line.find(", OA:")
                d['rmIoU'] = round(float(line[len('mIoU:'):b].strip()),2)
                d['rF1'] = round(float(line[b+len(', F1:'):c].strip()),2)
                d['rOA'] = round(float(line[c+len(', OA:'):].strip()),2)
        if line.startswith('boundary:   0.1'):
            epoach_flag = False
            s1, s2 = ', ratio:', '%, easy_err'
            a, b = line.find(s1),  line.find(s2)
            s = float(line[a+len(s1):b].strip())
            d['ratio_0.1'] = s
            

def takeElem(elem):
    return elem[sort_key]


if len(datas) > epochs:
    print("len(datas)", len(datas))
    datas = datas[:epochs]
    print(f"datas = datas[:{epochs}]")


datas.sort(key=takeElem, reverse=True)
datas = datas[:topk]

keys = list(datas[0].keys())[1:]
# print(keys)
count = [0] * len(keys)
for d in datas:
    print(d)
    for i, k in enumerate(keys):
        count[i] += d[k]
print(stage, sort_key, '\nMean:  ', end='')

for i, (c, k) in enumerate(zip(count, keys)):
    c = round(c/topk, 2)
    print(f'{k}: {c}, ', end='')

# print(f"coarse_mean: {coarse_mean/topk:.2f}, refine_mean: {refine_mean/topk:.2f}")

print("\n--------------------------------------------------------------------------------------\n\n")
