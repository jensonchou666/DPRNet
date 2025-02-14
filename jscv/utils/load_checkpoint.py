import torch
from collections import OrderedDict
import inspect, re, os
from typing import Optional, Dict, AnyStr, Union

#import torch.nn as nn

#常见的prefix:

# 是否输出加载信息
PRINT_INFO = True

# mmseg的格式
backbone = 'backbone'
decode_head = 'decode_head'
auxiliary_head = 'auxiliary_head'

decoder = 'decoder'

# pytorch_lighting
net_backbone = 'net.backbone'
net_decode_head = 'net.decode_head'


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def _prefix_in_model_(model, param_name):
    if model is not str:
        model = varname(model)
    return f'{model}.{param_name}'


def backbone_in(model, backbone_name='backbone'):
    return _prefix_in_model_(model, backbone_name)


def decode_head_in(model, decode_head_name='decode_head'):
    return _prefix_in_model_(model, decode_head_name)


def decoder_in(model, decoder_name='decoder'):
    return _prefix_in_model_(model, decoder_name)


#TODO state_dict所在位置
def _ckpt_to_state_dict_(ckpt: dict):
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'state_dict_ema' in ckpt:
        state_dict = ckpt['state_dict_ema']
    elif 'model' in ckpt:
        model = ckpt['model']
        if isinstance(model, dict):
            state_dict = model
        elif isinstance(model, torch.nn.Module):
            state_dict = model.state_dict()
        else:
            raise Exception('wrong type')
    else:
        state_dict = ckpt
    return state_dict


def browse_checkpoint_dict(checkpoint_dict: dict):
    # 预览checkpoint的各个参数，若只需加载其中一部分，可根据结果为load_checkpoint设置prefix
    state_dict = _ckpt_to_state_dict_(checkpoint_dict)
    for i, (k, v) in enumerate(state_dict.items()):
        print(f'#{i}:', k)


def browse_checkpoint(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(f'All of weight parameters in {checkpoint_path} :')
    browse_checkpoint_dict(ckpt)


def load_checkpoint(model: torch.nn.Module,
                    checkpoint: Optional[Union[AnyStr, Dict]],
                    prefix='',
                    operate_for_dict=None,
                    **kargs_operate):
    # 从dict加载
    if isinstance(checkpoint, str):
        if PRINT_INFO:
            print(
                f"Begin load {prefix} weights from checkpoint file:{checkpoint}"
            )
        if os.path.islink(checkpoint):
            checkpoint = os.readlink(checkpoint)
        checkpoint = torch.load(checkpoint, map_location='cpu')

    assert isinstance(checkpoint, dict)

    _state_dict = _ckpt_to_state_dict_(checkpoint).copy()
    
    # if operate_for_dict is not None:
    #     operate_for_dict(state_dict=_state_dict, **kargs_operate)

    m_state_dict = model.state_dict()
    count_omit = 0
    
    if prefix is None or prefix == '':
        state_dict = _state_dict
        poplist = []
        for k in state_dict.keys():
            if k not in m_state_dict:
                print("omit: ", k)
                poplist.append(k)
                count_omit += 1
        for k in poplist:
            state_dict.pop(k)
    else:
        state_dict = OrderedDict()
        if prefix[-1] != '.':
            prefix += '.'
        for k, v in _state_dict.items():
            if k.startswith(prefix):
                k2 = k[len(prefix):]
                if k2 in m_state_dict:
                    state_dict[k2] = v
                else:
                    print("omit: ", k)
                    count_omit += 1

    if operate_for_dict is not None:
        operate_for_dict(state_dict=state_dict, **kargs_operate)

    model.load_state_dict(state_dict, False)

    if not PRINT_INFO:
        return

    if prefix is None or prefix == '':
        print("Have loaded checkpoint")
    else:
        print(f"Have loaded checkpoint for {prefix}")

    if count_omit != 0:
        print('omit key numbers:', count_omit)
