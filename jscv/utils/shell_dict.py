#! /opt/conda/bin/python

import torch
import sys

# from pynput.keyboard import Key, Listener

show_state_dict = True


from typing import Iterable, Optional, Union, Dict

from torch import Tensor


def func_Tensor(t: Tensor):
    return f'{t.shape}'


def func_len(t):
    return f'{type(t)}({len(t)})'

def func_model(t: torch.nn.Module):
    return f'{t._get_name()}'



show_func = {Tensor: func_Tensor, dict: func_len, list: func_len, torch.nn.Module: func_model}

show_type = False


def _show_(v):
    global show_type
    if show_type:
        return type(v)
    return v


class Indexer():

    def __init__(self):
        self.ls_list = []
        self.size = 0
        self.isdict = True

    def get_index(self, s):
        if s.isdigit():
            d = int(s)
            if d > self.size:
                print('index over len(ls_list)')
                return None
            if self.isdict:
                k = self.ls_list[d]
                if k in self._n:
                    return k
                else:
                    print("Can't find key:", k)
                    return None
            else:
                return d
        else:
            if self.isdict:
                if s in self._n:
                    return s
                else:
                    print("Can't find key:", s)
                    return None
            else:
                print('List indices must be integer')
                return None


    def flush(self, n):
        if isinstance(n, dict):
            self.ls_list = []
            self.isdict = True
            for k in n.keys():
                self.ls_list.append(k)
            self.size = len(n)
            self._n = n
        elif isinstance(n, Iterable):
            self.ls_list = []
            self.isdict = False
            self.size = len(n)
        else:
            raise TypeError('!')


def ls(n):
    global show_func
    # print('type:', type(n))

    maxi_len = len(str(len(n) - 1))

    if isinstance(n, dict):
        maxlen = 0
        for k in n.keys():
            maxlen = max(maxlen, len(str(k)))
        for i, (k, v) in enumerate(n.items()):
            k_str = '{{:<{}}}'.format(maxlen + 1).format(k)
            i_str = '{{:<{}}}'.format(maxi_len).format(i)
            c = True
            for tp, func in show_func.items():
                if isinstance(v, tp):
                    print(f'#{i_str} {k_str}: ', func(v))
                    c = False
                    break
            if c:
                print(f'#{i_str} {k_str}: ', _show_(v))
    elif isinstance(n, Iterable):
        for i, d in enumerate(n):
            i_str = '{{:<{}}}'.format(maxi_len).format(i)
            c = True
            for tp, func in show_func.items():
                if isinstance(d, tp):
                    print(f'#{i_str}:', func(d))
                    c = False
                    break
            if c:
                print(f'#{i_str}:', _show_(d))
    else:
        print("Wrong")


def prompt(cwds):
    cwd = '/'
    for c in cwds:
        #cwd += '/' + str(c)
        cwd += str(c) + '/'
    return f'{cwd}$ '



#TODO 实现方向建 history
# cmd_history = []
# def on_press(key):
#     global cmd_history

#     if key == Key.up:

#     return False


def shell_dict(dict0: Optional[Union[Dict, Iterable]] = None, path=None):
    if dict0 is None:
        if path is None:
            return
        dict0 = torch.load(path, map_location='cpu')
    idr = Indexer()
    n_list = [dict0]
    cwds = []
    # n_list_pre
    idr.flush(n_list[-1])

    change_list = []

    while True:
        n = n_list[-1]

        # if len(cwds) > 0:
        #     cwd = ''

        c = input(prompt(cwds)).strip()

        cmds = c.split(' ')
        cmd0 = cmds[0]

        sz = len(cmds)
        if sz > 1:
            content = c[len(cmd0):].strip()

        if cmd0 == 'help':
            print('quit  ls  v  cd  rm...')
        elif cmd0 == 'ls':
            if sz == 1:
                ls(n)
            else:
                i = idr.get_index(content)
                if i is None:
                    continue
                ls(n[i])
        elif cmd0 == 'v' or cmd0 == 'value':
            i = idr.get_index(content)
            if i is None:
                continue
            print(n[i])
        elif cmd0 == 'cd':
            if len(cmds) < 2:
                continue
            if content == '..':
                if len(cwds) > 0:
                    n_list.pop()
                    cwds.pop()
            elif content == '/':
                n_list = n_list[0:1]
                cwds = []
            else:
                i = idr.get_index(content)
                if i is None:
                    continue
                if not isinstance(n[i], Iterable):
                    print("cant enter")
                    continue
                n_list.append(n[i])
                cwds.append(i)
            idr.flush(n_list[-1])
            
        elif cmd0 == 'quit':
            return

        elif cmd0 == 'T':
            global show_type
            show_type = not show_type

        elif cmd0 == 'rm':
            if sz == 1:
                continue
            i = idr.get_index(content)
            if i is None:
                continue
            if isinstance(n, dict) or isinstance(n, list):
                n.pop(i)
                change_list.append(f'removed: {i}')
                print(change_list[-1])
            idr.flush(n_list[-1])
        elif cmd0 == 'save':
            print('these items changed:')
            if len(change_list) == 0:
                print("None")
            for x in change_list:
                print(x)
            i = input('confirm save ? [y/n]')
            if i.lower() == 'y':
                if path is None:
                    print("path is None")
                else:
                    torch.save(dict0, path)
                    change_list = []
                    print("OK")

if __name__ == "__main__":
    print(sys.argv)
    shell_dict(path=sys.argv[1])
