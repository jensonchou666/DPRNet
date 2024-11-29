import os


file = "work_dir/GID_Water/1_globalnet_r101_aspp_16x16_r18_stage3-e120/final/log-1_globalnet_r101_aspp_16x16_r18_stage3-GID_Water.txt"

out_dir = "log_out"

f = open(file)

split = "|"

clumn = 6

begin = False

_s = 0
f2 = out_dir + "/" + os.path.split(file)[1]
if os.path.exists(f2):
    assert False, '文件存在'
f2 = open(f2, 'w')

while True:
    s = f.readline()
    if s == "":
        break
    
    s = s.strip()
    
    if s.startswith("| epoch") and not begin:
        if _s == 0:
            _s = 1
        else:
            begin = True
    
    if begin and s.startswith("|"):
        ss = s.split(split)
        ss = ss[clumn+1].strip()
        if "--" not in ss:
            print(ss, file=f2)