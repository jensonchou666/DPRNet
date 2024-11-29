import os

# output_file = 'work_dir/GID_Water/1_globalnet_r101_16x16_r18_stage3-e80/final/outputs.log'  #15e
# output_file = 'work_dir/GID_Water/1_globalnet_r101_aspp_16x16_r18_stage3-e120/final/outputs.log' #25e

# # output_file = 'work_dir/GID_Water/1_globalnet_r101_16x16_r50_stage3-e80/version_1/outputs.log' #  max 43
# output_file = 'work_dir/GID_Water/1_globalnet_r101_aspp_16x16_r50_stage3-e120/v1/outputs.log' #  max 120

# # BiSeNet
# # output_file = 'work_dir/GID_Water/dbnet_Gr101_Lr18-e120/a1/outputs.log' #  max 120


# output_file = 'work_dir/GID_Water/0_DPRNet_pvt_LR18_16x16_stage3-e80/version_1/outputs.log'

# output_file = 'work_dir/GID_Water/1_globalnet_r101_16x16_r18_stage3-e80/final/outputs.log'  #15e

# output_file = 'work_dir/GID_Water/DPRNet_GR101_LR18_16x16_stage3_fuse_GLType-e80/version_2/outputs.log'
# output_file = 'work_dir/GID_Water/a/3.txt'
# output_file = 'work_dir/GID_Water/1_globalnet_r101_aspp_16x16_r18_stage3_no_ffm-e120/version_2/outputs.log'

# output_file = 'work_dir/GID_Water/0_DPRNet_focal_LR18_16x16_stage3-e80/final/outputs.log'
# output_file = 'work_dir/GID_Water/0_DPRNet_pvt_LR18_16x16_stage3-e80/v1/outputs.log'

output_file = 'work_dir/GID_Water/0_DPRNet_swinT_LR18_16x16_stage3-e80/version_0/outputs.log'

topk = 10

epochs = 100

# stage = 'train'
stage = 'val'
sort_key = 'refine'
# sort_key = 'coarse'


print(output_file)
lines = [s.strip() for s in open(output_file).readlines()]

datas = []

epoach_flag = False

for i, line in enumerate((lines)):
    if f"'stage': '{stage}'" in line:
        d = {}
        a = line.find("'epoch':")
        b = line.find(", 'stage'")
        d['epoach'] = int(line[a+len("'epoch':"):b].strip())
        datas.append(d)
        epoach_flag = True
    
    if epoach_flag:
        d = datas[-1]
        if 'edge_acc(coarse_pred):' in line:
            a = line.find(":")
            d['coarse'] = float(line[a+1:].strip())
        elif "edge_acc(pred):" in line:
            a = line.find(":")
            d['refine'] = float(line[a+1:].strip())
            epoach_flag = False

def takeElem(elem):
    return elem[sort_key]


if len(datas) > epochs:
    print("len(datas)", len(datas))
    datas = datas[:epochs]
    print(f"datas = datas[:{epochs}]")


datas.sort(key=takeElem, reverse=True)
datas = datas[:topk]
coarse_mean, refine_mean = 0, 0
for d in datas:
    # print(d)
    coarse_mean += d['coarse']
    refine_mean += d['refine']
# print(stage, sort_key)
print(f"coarse_mean: {coarse_mean/topk:.2f}, refine_mean: {refine_mean/topk:.2f}")
print("--------------------------------------------------------------------------------------\n\n")
