import os
from PIL import Image
import numpy as np
import tqdm
import shutil
import torch
# mask_value = 1

label_dir = "data/gid_water/train/label"

PH, PW = 8,8

ignore_idx = 100
LowV = [0.1, 0.5, 1]
HighV = [0.1, 0.5, 1]
countL = [0,0,0]
countH = [0,0,0]
count = 0


for f in tqdm.tqdm(os.listdir(label_dir)):
    x = torch.from_numpy(np.array(Image.open(label_dir+"/"+f)))
    H,W = x.shape
    x = x.reshape(PH, H//PH, PW, W//PW)
    for k in range(PH):
        for p in range(PW):
            xkp = x[k, :, p]
            ignore = xkp==ignore_idx
            n_ignore = ignore.sum()
            n_total = (H//PH)*(W//PW)
            rate = ((xkp!=0).sum()-n_ignore)/(n_total-n_ignore)  * 100
            for i, v in enumerate(LowV):
                if rate < v:
                    countL[i] += 1
            for i, v in enumerate(HighV):
                if rate > (100-v):
                    countH[i] += 1
            count += 1

print("Patches:", PH, PW)
print("Full_Ground: ", end="")
for v, c in zip(LowV, countL):
    print(f"{(c/count*100):.2f}%(<{v}) , ", end="")
print()
print("Full_Water: ", end="")
for v, c in zip(HighV, countH):
    print(f"{(c/count*100):.2f}%(>{100-v}) , ", end="")
print()

'''




!!! old:


Patches: 8, 8
Train:
Full_Ground: 53.94%(<0.1) , 56.79%(<0.5) , 59.24%(<1)
Full_Water: 2.13%(>99.9) , 2.20%(>99.5) , 2.35%(>99)
Val:
Full_Ground: 58.68%(<0.1) , 60.80%(<0.5) , 62.57%(<1)
Full_Water: 1.91%(>99.9) , 2.05%(>99.5) , 2.12%(>99)



Patches: 16, 16
Train:
Full_Ground: 67.78%(<0.1) , 69.07%(<0.5) , 70.27%(<1)
Full_Water: 3.38%(>99.9) , 3.51%(>99.5) , 3.59%(>99)
Val:
Full_Ground: 70.40%(<0.1) , 71.44%(<0.5) , 72.42%(<1)
Full_Water: 3.45%(>99.9) , 3.65%(>99.5) , 3.79%(>99)


Patches: 4, 4
Train:
Full_Ground: 37.08%(<0.1) , 42.02%(<0.5) , 46.85%(<1)
Full_Water: 1.07%(>99.9) , 1.37%(>99.5) , 1.43%(>99)
Val:
Full_Ground: 45.00%(<0.1) , 48.47%(<0.5) , 51.67%(<1)
Full_Water: 0.69%(>99.9) , 0.83%(>99.5) , 0.97%(>99)
'''