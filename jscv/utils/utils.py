import torch, os, sys, re
import numpy as np


from itertools import repeat
import collections.abc
# From PyTorch internals


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def model_speed_test(model, epoach=100, input_shape=[2, 3, 1024, 1024]):
    warmup(20)
    print("...\n")

    x = torch.rand(input_shape).cuda()

    time_counter = TimeCounter(True)
    time_counter.begin()

    for i in range(epoach):
        model(x)

    time_counter.record_time("total", last=True)
    print(time_counter.str_once(epoach))


def wrap_text(text, max_length=80):
    """
    将字符串按指定长度换行。

    参数:
        text (str): 输入的长字符串。
        max_length (int): 每行的最大字符数，默认为80。

    返回:
        str: 包含换行符的格式化字符串。
    """
    lines = []
    while len(text) > max_length:
        # 找到最后一个空格的位置
        break_pos = text.rfind(' ', 0, max_length + 1)
        if break_pos == -1:  # 如果没有空格，直接硬切分
            break_pos = max_length
        lines.append(text[:break_pos])
        text = text[break_pos:].lstrip()  # 去掉开头的空格
    lines.append(text)  # 添加最后一部分
    return '\n'.join(lines)

def warmup(epoachs=10):
    print('warm up ...\n')
    with torch.no_grad():
        n = torch.nn.Conv2d(100, 100, 3, padding=1).cuda()
        x = torch.randn(2, 100, 500, 500).cuda()
        for i in range(epoachs):
            x = n(x)
    torch.cuda.synchronize()

class TimeCounter:
    
    def __init__(self, DO_DEBUG=False):
        self.NUM_T_DEBUG = 1111111
        self.TimeNames = []
        self.TimeList = None
        self.TimeListOnce = []
        self.startEvent = torch.cuda.Event(enable_timing=True)
        self.endEvent = torch.cuda.Event(enable_timing=True)
        self.DO_DEBUG = DO_DEBUG

    def __str__(self) -> str:
        s = ""
        for n, t in zip(self.TimeNames, self.TimeListOnce):
            s += f"{n}: {t:.2f}, "
        return s[:-2]

    def str_once(self, epoachs=1) -> str:
        s = ""
        for n, t in zip(self.TimeNames, self.TimeListOnce):
            s += f"{n}: {t/epoachs:.2f}, "
        return s[:-2]

    def str_total(self, epoachs=1) -> str:
        s = ""
        for n, t in zip(self.TimeNames, self.TimeList):
            s += f"{n}: {t/epoachs}, "
        return s[:-2]

    def str_total_porp(self) -> str:
        s = ""
        sm = sum(self.TimeList)
        for n, t in zip(self.TimeNames, self.TimeList):
            s += f"{n}: {(t/sm):.2%}, "
        return s[:-2]

    def str_porp(self) -> str:
        s = ""
        sm = sum(self.TimeListOnce)
        for n, t in zip(self.TimeNames, self.TimeListOnce):
            s += f"{n}: {(t/sm):.2%}, "
        return s[:-2]

    def begin(self):
        return self.record_time(first=True)

    def last(self, name=None):
        return self.record_time(name, last=True)


    def _record_time_(self, name=None, first=False, last=False,
                      pause=False, resume=False):

        if first:
            self.TimeListOnce = []
            torch.cuda.synchronize()
            self.startEvent.record()
        elif resume:
            self.startEvent.record()
        else:
            self.endEvent.record()
            torch.cuda.synchronize()
            t = self.startEvent.elapsed_time(self.endEvent)
            self.TimeListOnce.append(t)
            if len(self.TimeNames) < self.NUM_T_DEBUG:
                self.TimeNames.append(name)
            if last:
                self.NUM_T_DEBUG = len(self.TimeListOnce)
                if self.TimeList is None:
                    self.TimeList = self.TimeListOnce.copy()
                else:
                    for i, t in enumerate(self.TimeListOnce):
                        #todo
                        # if i not in self.TimeList:
                        #     self.TimeList.insert(i, t)
                        self.TimeList[i] += t
                return
            elif pause:
                return
            self.startEvent.record()

    def record_time(self, name=None, first=False, last=False,
                    pause=False, resume=False):
        if self.DO_DEBUG:
            self._record_time_(name, first, last, pause, resume)




def test_model_latency(model, gpu_id=3, B=2,
                       epoachs_512=100, epoachs_1024=25, notes=True):

    s1 = ""
    if notes:
        s1 = "    # "

    torch.cuda.set_device(gpu_id)
    model = model.cuda().eval()
    warmup()
    ct = TimeCounter(True)
    with torch.no_grad():
        x = torch.randn(B, 3, 512, 512).cuda()
        ct.record_time(first=True)
        for i in range(epoachs_512):
            model(x)
        ct.record_time(last=True)
        t_one = ct.TimeListOnce[0] / epoachs_512
        print(f"{s1}{str(list(x.shape)):<22}  {t_one} ms")

        x = torch.randn(B, 3, 1024, 1024).cuda()
        ct.record_time(first=True)
        for i in range(epoachs_1024):
            model(x)
        ct.record_time(last=True)
        t_one = ct.TimeListOnce[0] / epoachs_1024
        print(f"{s1}{str(list(x.shape)):<22}  {t_one} ms")


# conv_op = None
def edge_detect(input: torch.Tensor, out_channel=1):
    # 用nn.Conv2d定义卷积操作
    # global conv_op
    # if conv_op is None:

    in_channel = input.shape[1]

    conv_op = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, in_channel, axis=0)
    sobel_kernel = np.repeat(sobel_kernel, out_channel, axis=1)

    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op.to(input.device)(input)
    # print(torch.max(edge_detect))
    # 将输出转换为图片格式
    #edge_detect = edge_detect.squeeze()
    return edge_detect



def edge_detect_binary(input: torch.Tensor, out_channel=1, threshold=0.01):
    e = edge_detect(input, out_channel)
    e = e / torch.max(e)
    e[e > threshold] = 1
    e[e < threshold] = 0
    return e

def edge_detect_target(input: torch.Tensor, out_channel=1):
    e = edge_detect(input, out_channel)
    # e = e / torch.max(e)
    e[e > 0.1] = 1
    e[e < 0.1] = 0
    return e





def seek_line(file, line):
    file.seek(0)
    line += '\n'
    while True:
        s = file.readline()
        if s == line or s == '':
            break
    file.seek(file.tell())
    if s == '':
        return False
    return True

def seek_line_match(file, pattern):
    file.seek(0)
    while True:
        s = file.readline()
        if s == '' or re.match(pattern, s[:-1]):
            break
    file.seek(file.tell())
    if s == '':
        return False
    return True


def eta_total(spend, batchs, total):
    return float(spend) * total / batchs

def set_default(cfg, default_dict: dict):
    if isinstance(cfg, dict):
        for k, v in default_dict.items():
            if k not in cfg:
                cfg[k] = v
    else:
        for k, v in default_dict.items():
            if not hasattr(cfg, k):
                setattr(cfg, k, v)


def format_if_in_dict(s, _dict):
    L1 = []
    aa = [-1, -1]
    for i in range(len(s)):
        if s[i] == '{':
            aa[0] = i

        elif s[i] == '}':
            aa[1] = i
            if aa[0] != -1:
                L1.append(aa.copy())
            aa[0] = -1
    k0 = 0
    s2 = ''
    for i, j in L1:
        if i != 0:
            s2 += s[k0:i]
        key = s[i + 1:j]
        if ':' in key:
            i1 = key.index(':')
            k = key[:i1]
            if k in _dict:
                _d0 = {k: _dict[k]}
                s2 += s[i:j + 1].format(**_d0)
            else:
                s2 += s[i:j + 1]
        elif key in _dict:
            s2 += str(_dict[key])
        else:
            s2 += s[i:j + 1]
        k0 = j + 1
    if len(L1) > 0:
        s2 += s[k0:]
    return s2

def format_if_in(s, **kargs):
    return format_if_in_dict(s, kargs)

class GlobalDoOnce:
    pass
_globaldoonce_ = GlobalDoOnce()
def do_once(obj, id):
    if obj is None:
        obj = _globaldoonce_
    name = f'_do_once_{id}'
    if hasattr(obj, name):
        return False
    else:
        setattr(obj, name, True)
        return True

#!!!
def do_count(obj, id, count):
    name = f'_do_once_{id}'
    if hasattr(obj, name):
        return False
    else:
        setattr(obj, name, True)
        return True


class redirect:
    def __init__(self, file, std='stdout'):
        self.file = file
        self.std = std

    def __enter__(self):
        self.old = getattr(sys, self.std)
        setattr(sys, self.std, self.file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(sys, self.std, self.old)


def torch_load(path, **kargs):
    if os.path.islink(path):
        path = os.readlink(path)
    return torch.load(path, **kargs)


def availabe(obj, attr):
    return hasattr(obj, attr) and (obj.__getattr__(attr) is not None)


def format_left(v, indent=1):
    return '{{:<{}}}'.format(indent).format(v)


def format_right(v, indent=1):
    return '{{:>{}}}'.format(indent).format(v)


def format_float(v, p=4):
    return '{{:.{}f}}'.format(p).format(v)

def format_time(value, **kargs):
    if value is None:
        return 'None'
    if int(value / 60) > 0:
        return '{}m{:.2f}s'.format(int(value / 60), value % 60)
    else:
        return f'{value:.4f}s'

def unit_div(i, unit):
    if unit == 'M':
        return i / 1000**2
    elif unit == 'G':
        return i / 1000**3
    elif unit == 'T':
        return i / 1000**4
    else:
        return i

import os

def on_train():
    return os.environ.get('run_mode') == 'train'

class Version:

    def last_version(dir, prefix='version_'):
        files = os.listdir(dir)
        version = -1
        for f in files:
            if f.startswith(prefix):
                _v = f[len(prefix):]
                if _v.isdigit():
                    version = max(version, int(_v))
        return version

    def copyed_max_version(dir, filename_prefix, bracket='()'):
        files = os.listdir(dir)
        version = 0
        l, r = bracket[:-1], bracket[-1]
        for f in files:
            n_old = filename_prefix + l
            if f.startswith(n_old):
                a = f[len(n_old):].split(r)[0]
                if a.isdigit() and int(a) > 0:
                    version = max(version, int(a))
        return version


def color_tulpe_to_string(color_tulpe):
    r, g, b = color_tulpe
    return f"#{r:02X}{g:02X}{b:02X}"




    # def next_copy_name(dir, filename_prefix, bracket='()'):
    #     v = Version.copyed_max_version(dir, filename_prefix, bracket)
    #     return
