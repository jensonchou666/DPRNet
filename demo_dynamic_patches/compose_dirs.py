import os
import shutil

if __name__ == '__main__':

    directory = '0-Test/DPR_pvtb1_efficient-b3_SACI_v2-e40/outputs_val(dynamic-test)'

    out_dir = '0-Test/DPR_pvtb1_efficient-b3_SACI_v2-e40/outputs_val(dynamic-test-compose-1)'

    add_idx = 7

    remove_it = True
    remove_it = False

    if remove_it:
        for f in os.listdir(out_dir + '/0_info'):
            if f'[{add_idx}]' in f:
                os.remove(out_dir + '/0_info/' + f)
        for subdir in os.listdir(out_dir):
            for f in os.listdir(out_dir + '/' + subdir):
                if '(dynamic)' in f and f'[{add_idx}]' in f:
                    os.remove(os.path.join(out_dir + '/' + subdir, f))
        exit()


    dir_i = None

    for Dir in os.listdir(directory):
        j = Dir.find(')')
        idx = int(Dir[1:j])
        if idx == add_idx:
            dir_i = os.path.join(directory, Dir)
            break
    if not dir_i:
        exit('No directory')

    print(dir_i)

    os.makedirs(out_dir + '/0_info', exist_ok=True)

    for f in os.listdir(dir_i + '/0_info'):
        name, ext = os.path.splitext(f)
        dst = os.path.join(out_dir + '/0_info', name + f'[{add_idx}]' + ext)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(dir_i + '/0_info', f), dst)


    for subdir in os.listdir(dir_i):
        outsubdir = out_dir + '/' + subdir
        os.makedirs(outsubdir, exist_ok=True)
        for f in os.listdir(dir_i + '/' + subdir):
            if '(dynamic)' in f:
                name, ext = os.path.splitext(f)
                dst = os.path.join(outsubdir, name + f'[{add_idx}]' + ext)
                if not os.path.exists(dst):
                    shutil.copy(os.path.join(dir_i, subdir, f), dst)
            else:
                dst = os.path.join(outsubdir, f)
                if not os.path.exists(dst):
                    shutil.copy(os.path.join(dir_i, subdir, f), dst)