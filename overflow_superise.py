import subprocess
import time
import argparse
import os, sys


out_path = "work_dir/STD_LOG/overflow.log"

if os.path.exists(out_path):
    os.remove(out_path)

def get_args():
    parser = argparse.ArgumentParser(description='Monitor GPU memory usage')
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument('--interval', type=int, default=30, help='Interval in seconds for monitoring')
    parser.add_argument("--threshold", type=int, default=1000)
    return parser.parse_args()



def get_gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE,
        text=True
    )
    output = result.stdout.strip()
    memory_info = []
    for line in output.split('\n'):
        used, total = map(int, line.split(','))
        memory_info.append({'used': used, 'total': total})
    return memory_info



def run():
    args = get_args()
    id = args.id
    interval = args.interval
    threshold = args.threshold
    while(True):
        memory_info = get_gpu_memory()
        # print (memory_info)
        used = memory_info[id]['used']
        if used < threshold:
            print(f'GPU {id} memory usage is below the threshold of {threshold}MB. Current usage: {used}MB')
            with open(out_path, 'a') as file:
                file.write(f"GPU {id} memory usage: {used}MB\n")


        time.sleep(interval)
    
    
if __name__ == '__main__':
    run()