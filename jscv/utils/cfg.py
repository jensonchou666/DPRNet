import pydoc
import sys, os
from importlib import import_module
from pathlib import Path
from typing import Any, Union, Iterable

from addict import Dict

def on_train():
    return os.environ.get('run_mode') == 'train'


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        else:
            return value
        raise ex

    # def __setattr__(self, name, value):


def py2dict(file_path: Union[str, Path]) -> dict:
    """Convert python file to dictionary.
    The main use - config parser.
    file:
    ```
    a = 1
    b = 3
    c = range(10)
    ```
    will be converted to
    {'a':1,
     'b':3,
     'c': range(10)
    }
    Args:
        file_path: path to the original python file.
    Returns: {key: value}, where key - all variables defined in the file and value is their value.
    """
    file_path = Path(file_path).absolute()

    if file_path.suffix != ".py":
        raise TypeError(
            f"Only Py file can be parsed, but got {file_path.name} instead.")

    if not file_path.exists():
        raise FileExistsError(f"There is no file at the path {file_path}")

    module_name = file_path.stem

    if "." in module_name:
        raise ValueError("Dots are not allowed in config file path.")

    config_dir = str(file_path.parent)

    sys.path.insert(0, config_dir)

    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items() if not name.startswith("__")
    }

    return cfg_dict


def py2cfg(file_path: Union[str, Path]) -> ConfigDict:
    cfg_dict = py2dict(file_path)

    return ConfigDict(cfg_dict)


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)



def get_train_dataset_aug(cfg):
    if hasattr(cfg, "train_aug_kwargs"):
        train_aug = cfg.train_aug(crop_size=cfg.train_crop_size, **cfg.train_aug_kwargs)
    else:
        train_aug = cfg.train_aug(crop_size=cfg.train_crop_size)

    if hasattr(cfg, "val_aug_kwargs"):
        val_aug = cfg.val_aug(crop_size=cfg.val_crop_size, **cfg.val_aug_kwargs)
    else:
        val_aug = cfg.val_aug(crop_size=cfg.val_crop_size)

    return train_aug, val_aug

def get_test_dataset_aug(cfg):
    if hasattr(cfg, "test_aug_kwargs"):
        return cfg.test_aug(crop_size=cfg.val_crop_size, **cfg.test_aug_kwargs)
    else:
        return cfg.test_aug(crop_size=cfg.val_crop_size)

def get_dataset_kwargs(cfg):
    train_kargs, val_kargs, test_kargs = {}, {}, {}
    
    if hasattr(cfg, "dataset_kwargs"):
        train_kargs.update(cfg.dataset_kwargs)
        val_kargs.update(cfg.dataset_kwargs)
        test_kargs.update(cfg.dataset_kwargs)

    if hasattr(cfg, "train_dataset_kwargs"):
        train_kargs.update(cfg.train_dataset_kwargs)
    if hasattr(cfg, "val_dataset_kwargs"):
        val_kargs.update(cfg.val_dataset_kwargs)
    if hasattr(cfg, "test_dataset_kwargs"):
        test_kargs.update(cfg.test_dataset_kwargs)

    return train_kargs, val_kargs, test_kargs




'''
Old

class Var:

    def __init__(self, name):
        self.name = name

    def replace(obj, args, kargs):
        for i, v in enumerate(args):
            if isinstance(v, Var):
                args[i] = getattr(obj, v.name)
        for k, v in kargs.items():
            if isinstance(v, Var):
                kargs[k] = getattr(obj, v.name)

    def replace_dict(_dict, args, kargs):
        for i, v in enumerate(args):
            if isinstance(v, Var):
                args[i] = _dict[v.name]
        for k, v in kargs.items():
            if isinstance(v, Var):
                kargs[k] = _dict[v.name]


class Pack:

    def __init__(self, func, *func_args, root=None, **func_kargs):
        #self.vars = vars_outputs
        self.func = func
        self.args = list(func_args)
        self.kargs = func_kargs
        self.root = None

    def achieve(self, obj=None):
        if self.root is not None:
            obj = self.root
        Var.replace(obj, self.args, self.kargs)
        ret_dict = self.func(*self.args, **self.kargs)
        assert isinstance(ret_dict, dict)
        for k, v in ret_dict.items():
            setattr(obj, k, v)

    def achieve_to_dict(self, _dict: dict = None):
        if self.root is not None:
            _dict = self.root
        Var.replace_dict(_dict, self.args, self.kargs)
        ret_dict = self.func(*self.args, **self.kargs)
        assert isinstance(ret_dict, dict)
        for k, v in ret_dict.items():
            _dict[k] = v


class DictFunc:

    def __init__(self, func, ret_keys):
        self.func = func
        self.ret_keys = ret_keys
        if isinstance(ret_keys, str):
            self.ret_keys = [ret_keys]

    def __call__(self, *args: Any, **kwds: Any):
        ret = self.func(*args, **kwds)

        if len(self.ret_keys) == 1:
            return {self.ret_keys[0]: ret}
        else:
            d2 = {}
            for k, v in zip(self.ret_keys, ret):
                if k is not None:
                    d2[k] = v
            return d2


import os


def create_datasets_pack(packs: dict,
                         create_func,
                         train_crop_size=None,
                         val_crop_size=None,
                         **kargs):
    if os.environ.get('run_mode') == 'train':
        packs["train_dataset"] = Pack(DictFunc(create_func, "train_dataset"),
                                      crop_size=train_crop_size,
                                      stage="train",
                                      **kargs)
        packs["val_dataset"] = Pack(DictFunc(create_func, "val_dataset"),
                                    crop_size=val_crop_size,
                                    stage="val",
                                    **kargs)
    else:
        packs["test_dataset"] = Pack(DictFunc(create_func, "test_dataset"),
                                     crop_size=val_crop_size,
                                     stage="test",
                                     **kargs)


# 常用 vars:
class cfg_vars:
    datasets = ["train_dataset", "val_dataset", "test_dataset"]
    model = ["model", "optimizer", "lr_scheduler"]

    backbone_ckpt = ["backbone_ckpt_path", "backbone_prefix"]


# def register_Pack()

#

'''