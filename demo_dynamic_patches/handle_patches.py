from dynamic_patches import *
# from jscv.hr_models.DPRNet.dynamic_patches import *

import os
import tqdm
import re
import shutil
from PIL import Image
import cv2

directory = '0-Test/DPR_pvtb1_efficient-b3_SACI_v2-e40/outputs_val'
out_dir = '0-Test/DPR_pvtb1_efficient-b3_SACI_v2-e40/outputs_val(dynamic-test)'

''' args '''
pdtH, pdtW = PDT_size = (16,16)
max_patches_num = 36
threshold = 0.02
threshold_rate_1x1 = 0.7

score_compute_classtype = PNScoreComputer
score_compute_kargs = dict(
    fx_points=[(2, 0.6), (36, 0.25)],
    gx_point=(4, 0.5)
)

min_pathes_nums = int(max_patches_num*0.1)
min_thick_rate = 0.05
max_thick_rate = 0.15


def args_to_string():
    params = {
        'PDT_size': PDT_size,
        'max_patches_num': max_patches_num,
        'threshold': threshold,
        'threshold_1x1': threshold_rate_1x1*threshold,
        'score_compute_classtype': score_compute_classtype,
        'score_compute_kargs': score_compute_kargs,
        'min_pathes_nums': min_pathes_nums,
        'min_thick_rate': min_thick_rate,
        'max_thick_rate': max_thick_rate
    }
    # 计算最大键长度，用于对齐
    max_key_len = max(len(key) for key in params.keys())
    # 打印参数和值
    result = ""
    for param, value in params.items():
        result += f"{param:<{max_key_len}} : {value}\n"
    return result



copy_list = [
    # '1_1_label',
    # '3_4_c_div_(simple_div)',
    # '3_5_cw_div_(simple_div)',
    '3_6_c_wrong_div_(simple_div)',
    '4_2_PDT_div_(simple_div)',
]

paint_list = [
    # ('0_1_image', '0_3_image_div_(dynamic)'),
    # ('1_1_label', '1_3_label_div_(dynamic)'),
    ('3_3_coarse_wrong', '3_6_1_c_wrong_div_(dynamic)'),
    ('4_1_PDT', '4_2_1_PDT_div_(dynamic)'),
]

PDT_filename = 'PDT(16x16).tensor'

subdir_name = f"{pdtH}x{pdtW}-N_{min_pathes_nums}_{max_patches_num}\
-T{threshold}_{round(threshold_rate_1x1*threshold,3)}\
-y_{score_compute_kargs['fx_points'][0][1]}\
_{score_compute_kargs['fx_points'][1][1]}\
_{score_compute_kargs['gx_point'][1]}"

def generate_unique_filename(directory, base_name="name"):
    """
    在指定目录中生成一个不与现有文件冲突的唯一文件名，文件名格式为 (id)name。

    参数：
    directory (str): 目标文件夹路径。
    base_name (str): 文件名的基础部分，默认为 "name"。

    返回：
    str: 新生成的文件名，形式为 (n+1)name。
    """
    # 提取现有文件中的 id
    existing_ids = []
    for filename in os.listdir(directory):
        # 文件名格式为 (id)name，例如 "(1)file", "(2)file"
        parts = filename.split(')')
        if len(parts) > 1 and parts[0].startswith('(') and parts[1]:
            try:
                # 提取文件中的 id 并转换为整数
                file_id = int(parts[0][1:])
                existing_ids.append(file_id)
            except ValueError:
                pass  # 如果文件名不符合预期格式，则跳过

    # 计算下一个 id (n+1)
    next_id = max(existing_ids, default=0) + 1

    # 构建新的文件名
    new_filename = f'({next_id}){base_name}'

    return new_filename


# 排序函数：根据文件名中的数字部分进行排序
def extract_numbers(filename):
    # 使用正则表达式提取文件名中的所有数字
    numbers = re.findall(r'\d+', filename)
    # 转换为整数并返回
    return [int(num) for num in numbers]


def demo_paint_images():
    global out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir = out_dir + '/' + generate_unique_filename(out_dir, subdir_name)
    os.makedirs(out_dir)
    info_dir = out_dir + '/0_info'
    os.makedirs(info_dir)

    names = sorted(os.listdir(directory))
    names = sorted(names, key=extract_numbers)

    dynamicPatchesGrouping = DynamicPatchesGrouping(
        max_patches_num, 
        PDT_size=PDT_size,
        threshold_rate_1x1=threshold_rate_1x1,
        score_compute_classtype=score_compute_classtype,
        score_compute_kargs=score_compute_kargs,
        min_pathes_nums=min_pathes_nums,
        kernel_dict=get_kernel_dict(), toCUDA=True)
    
    dynamicPatchesGrouping.analyze_init()

    for idname in tqdm.tqdm(names):
        outdir_i = out_dir + '/' + idname
        os.makedirs(outdir_i)
        dir_i = directory + '/' + idname

        PDT = torch.load(dir_i + '/' + PDT_filename)
        assert isinstance(PDT, torch.Tensor)
        assert PDT.shape == (pdtH, pdtW)

        allocated_groups = dynamicPatchesGrouping.scan_patches_difficulty_table(PDT, threshold)
        
        info = dynamicPatchesGrouping.analyze_once(allocated_groups, PDT, True)
        
        with open(info_dir + '/result.txt', 'w') as info_file:
            #TODO
            print(args_to_string(), '\n', file=info_file)
            print(dynamicPatchesGrouping.analyze_all(), file=info_file, flush=True)


        with open(outdir_i + '/result.txt', 'w') as f:
            print(info, file=f)

        for f in os.listdir(dir_i):
            for copy_one in copy_list:
                if copy_one in f:
                    shutil.copy(dir_i+'/'+f, outdir_i+'/'+f)
                    break

            for fromf, tof in paint_list:
                if fromf in f:
                    map1 = torch.tensor(np.array(Image.open(dir_i+'/'+f)))
                    map1 = paint_on_image(map1, allocated_groups, dynamicPatchesGrouping, 
                                          min_thick_rate, max_thick_rate)
                    tof = f.replace(fromf, tof)
                    cv2.imwrite(outdir_i+'/'+tof, cv2.cvtColor(map1.cpu().numpy(), cv2.COLOR_BGR2RGB))
                    break
    # print(file=info_file)
    # info_file.close()

def demo_threshold_learner():
    import time
    learner = ThresholdLearner(
        select_rate_target=0.4,  # 期望的选择率
        threshold_init=0.005,  # 初始阈值
        base_factor=0.5,  # 初始步长
        adjustment_decay=0.95,  # 步长衰减因子
        window_size=3,  # 平滑窗口大小
        max_adjustment=0.1,  # 最大调整步长
        positive_relation=False,  # 是否为正相关（选择率随阈值增大而增大）
        base_factor_min=0.00001,  # 设置 base_factor 的最小值
        base_factor_max=0.1,  # 设置 base_factor 的最大值
        convergence_threshold=0.002,  # 设置收敛阈值
    )
    dynamicPatchesGrouping = DynamicPatchesGrouping(
        max_patches_num, 
        PDT_size=PDT_size,
        threshold_rate_1x1=threshold_rate_1x1,
        score_compute_classtype=score_compute_classtype,
        score_compute_kargs=score_compute_kargs,
        min_pathes_nums=min_pathes_nums,
        kernel_dict=get_kernel_dict(), toCUDA=True)

    names = sorted(os.listdir(directory))
    names = sorted(names, key=extract_numbers)

    for iteration in range(100):
        time0 = time.time()
        time_load = 0
        dynamicPatchesGrouping.analyze_init()
        for idname in names:
            dir_i = directory + '/' + idname
            time1 = time.time()
            PDT = torch.load(dir_i + '/' + PDT_filename)
            time_load += time.time() - time1
            assert isinstance(PDT, torch.Tensor)
            assert PDT.shape == (pdtH, pdtW)
            allocated_groups = dynamicPatchesGrouping.scan_patches_difficulty_table(PDT, learner.threshold)
            dynamicPatchesGrouping.analyze_once(allocated_groups, PDT)

        time0 = time.time() - time0
        info = dynamicPatchesGrouping.analyze_all()
        select_rate = info['select_rate']
        new_threshold, diff = learner.adjust_threshold(select_rate)
        print('DynamicPatchesGrouping Spend:', time0-time_load)
        print(info)
        # 输出当前阈值调整后的状态
        print(f"Iteration {iteration+1}: Threshold={new_threshold:.4f}, "
              f"Select Rate={select_rate:.4f}, Target={learner.select_rate_target:.4f}, "
              f"Base Factor={learner.base_factor:.4f}, Diff={diff:.4f}")
        print()
        # 检查是否已收敛
        if abs(diff) < learner.convergence_threshold:
            print(f"Convergence reached at iteration {iteration+1}.")
            break

if __name__ == '__main__':
    # demo_paint_images()
    demo_threshold_learner()