import sys, os
import thop
import torch

from collections import OrderedDict, Iterable

from .utils import *




def profile_quiet(model, inputs):
    _stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    flops, params = thop.profile(model, inputs)
    sys.stdout = _stdout
    return flops, params


class TreeNode:

    def __init__(self, parents=None, seat=-1):
        self.parents = parents
        self.seat = seat
        self.childen = []

    def add_child(self, node):
        node.parents = self
        node.seat = len(self.childen)
        self.childen.append(node)

    def free(self):
        for c in self.childen:
            c.free()
        del self

    def forward(self, call, depth=0, index=0, seat=0):
        TreeNode._i_ = index
        self._forward(call, depth, seat)

    def backward(self, call, depth=0, index=0, seat=0):
        TreeNode._i_ = index
        self._backward(call, depth, seat)

    def _forward(self, call, depth=0, seat=0):
        TreeNode._i_ += 1
        call(self, depth=depth, index=TreeNode._i_ - 1, seat=seat)
        for i, c in enumerate(self.childen):
            c._forward(call, depth + 1, i)

    def _backward(self, call, depth=0, seat=0):
        TreeNode._i_ += 1
        for i, c in enumerate(self.childen):
            c._forward(call, depth + 1, i)
        call(self, depth=depth, index=TreeNode._i_ - 1, seat=seat)


class LayerNode(TreeNode):

    def __init__(self, key=None, parents=None, seat=-1):
        super().__init__(parents, seat)
        self.key = key
        self.columns = {}  # OrderedDict()

    def add_layer(self, key=None):
        layer = LayerNode(key)
        super().add_child(layer)
        return layer


class TraverseModel:

    def __init__(self):
        pass
        #self.opt = StatisticOperate

    def _enter_step_(self, model, inputs, **kargs):
        pass

    def _before_traverse_(self, model, inputs, **kargs):
        pass

    def _after_traverse_(self, model, inputs, **kargs):
        pass

    def _none_traverse_(self, model, inputs, **kargs):
        pass

    def _exit_step_(self, model, inputs, **kargs):
        pass

    def step(self, model: torch.nn.Module, inputs, **kargs):
        # define traverse()

        self._enter_step_(model, inputs, **kargs)

        if hasattr(model, 'traverse'):
            self._before_traverse_(model, inputs, **kargs)
            model.traverse(self, *inputs)
            self._after_traverse_(model, inputs, **kargs)
        else:
            self._none_traverse_(model, inputs, **kargs)

        return self._exit_step_(model, inputs, **kargs)


class StatisticModel(TraverseModel):

    indent_layer = 40

    def __init__(self, top_down=True, strline='=' * 80, indent=2):
        super().__init__()
        self.str_unitV = '|' + ' ' * indent
        self.strline = strline
        self.top_down = top_down

        self.root = LayerNode()
        self.node = self.root

        self.columns_info = OrderedDict()
        self.columns_info['key'] = {
            'title': 'Model Layer',
            'indent': StatisticModel.indent_layer,
        }

    def strprefix(self, depth):
        if depth == 0:
            return '──'
        if self.top_down:
            return self.str_unitV * (depth) + '└─'
        else:
            return self.str_unitV * (depth) + '┌─'

    def print_row(self, node: LayerNode, depth, index, seat):

        name = f'{index:<3} {self.strprefix(depth)}{str(node.key)}: {depth}-{seat}'
        row = '{{:<{}}}'.format(
            self.columns_info['key']['indent']).format(name)

        iter_col_value = iter(self.columns_info.items())
        next(iter_col_value)

        for k, v in iter_col_value:
            if k in node.columns:
                show = False
                if not isinstance(node.columns[k], dict):
                    value = node.columns[k]
                elif 'show' in node.columns[k]:
                    value = node.columns[k]['show']
                    show = True
                else:
                    value = node.columns[k]['value']
                if not show:
                    if 'precision' in v:
                        value = format_float(value, v['precision'])
                    #TODO 在这里添加Format的方式
                    elif 'custom_format' in v:
                        f = v['custom_format']
                        f(value,
                          node=node,
                          depth=depth,
                          index=index,
                          seat=seat)
                    else:
                        value = str(value)
            else:
                value = '--'
            row += '{{:<{}}}'.format(v['indent']).format(value)
        print(row)

    def print(self):
        print(self.strline)
        row = ''
        for info in self.columns_info.values():
            row += '{{:<{}}}'.format(info['indent']).format(info['title'])
        print(row)
        print(self.strline)

        for n in self.root.childen:
            assert type(n) is LayerNode
            if self.top_down:
                n.forward(self.print_row)
            else:
                n.backward(self.print_row)
        print(self.strline)

    def _enter_step_(self, model, inputs, **kargs):
        if 'model_name' in kargs:
            name = kargs['model_name']
        elif 'name' in kargs:
            name = kargs['name']
        else:
            name = model._get_name()

        self.node = self.node.add_layer(name)

        self.node.stat_self_alone = False

    def _before_traverse_(self, model, inputs, **kargs):
        pass

    def _after_traverse_(self, model, inputs, **kargs):
        pass

    def _none_traverse_(self, model, inputs, **kargs):
        self.node.stat_self_alone = True

    def _exit_step_(self, model, inputs, **kargs):
        if self.node.stat_self_alone:
            self.statistic_alone(model, inputs, **kargs)
        else:
            self.statistic(model, inputs, **kargs)

        self.node = self.node.parents

        return model(*inputs)

    def statistic(self, model, inputs, **kargs):
        pass

    def statistic_alone(self, model, inputs, **kargs):
        pass

    def statistic_self_alone(self):
        self.node.stat_self_alone = True

    def register(self, column_key, info_dict):
        self.columns_info[column_key] = info_dict

    def add_item(self, k, v):
        self.node.columns[k] = v


class StatisticScale(StatisticModel):

    def __init__(self,
                 form=['shape', 'flops', 'params'],
                 divide_rate=1,
                 strline='=' * 80,
                 indent=2):
        super().__init__(strline=strline, indent=indent)

        self.register_items(form)

        self.divide_rate = divide_rate

    def register_items(self, form):
        columns = self.columns_info

        if 'shape' in form:
            columns['shape'] = {
                'title': 'Input Shape(0)',
                'indent': 25,
            }

        if 'flops' in form:
            columns['flops'] = {
                'title': 'Flops',
                'indent': 14,
                'unit': 'G',
                'precision': 2,
            }
        if 'params' in form:
            columns['params'] = {
                'title': 'Params',
                'indent': 14,
                'unit': 'M',
                'precision': 2,
            }

    def add_float_item(self, key, value):
        unit = self.columns_info[key]['unit']
        p = self.columns_info[key]['precision']
        v = unit_div(value, unit)
        self.add_item(key, {'value': value, 'show': format_float(v, p) + unit})

    def _stat_common(self, model, inputs, **kargs):
        cinfo = self.columns_info
        if 'shape' in cinfo:
            shape = list(inputs[0].shape[1:])
            self.add_item('shape', shape)

    def statistic(self, model, inputs, **kargs):

        cinfo = self.columns_info

        self._stat_common(model, inputs, **kargs)

        if 'flops' in cinfo:
            flops = 0.0
            for n in self.node.childen:
                flops += n.columns['flops']['value']
            self.add_float_item('flops', flops)
        if 'params' in cinfo:
            params = 0.0
            for n in self.node.childen:
                params += n.columns['params']['value']
            self.add_float_item('params', params)

    def statistic_alone(self, model, inputs, **kargs):
        #TODO 统计空模块、 定制统计

        self._stat_common(model, inputs, **kargs)

        cinfo = self.columns_info
        if 'flops' in cinfo or 'params' in cinfo:
            flops, params = profile_quiet(model, inputs)

        if 'flops' in cinfo:
            flops = flops / self.divide_rate
            self.add_float_item('flops', flops)
        if 'params' in cinfo:
            params = params / self.divide_rate
            self.add_float_item('params', params)

    def stat(model: torch.nn.Module,
             input=[2, 3, 512, 512],
             form=['shape', 'flops', 'params'],
             model_name=None,
             indent: int = 2,
             device='gpu',
             normalize=False,  # result normalized to [2, 3, 512, 512]
             **kargs):
        if isinstance(input, torch.Tensor):
            input_shape = input.shape
        elif isinstance(input, Iterable):
            input_shape = input
            input = torch.randn(*input_shape)
        if model_name is None:
            model_name = model._get_name()

        if device == 'gpu':
            input = input.cuda()
            model = model.cuda()
        elif device == 'cpu':
            input = input.cpu()
            model = model.cpu()
        else:
            raise TypeError('not gpu or cpu')

        dr = 1
        dr_str = ''
        if normalize:
            dr_str = ', (normalized)'
            dr = float(input_shape[0])
            dr *= input_shape[1] / 3
            dr *= input_shape[2] / 512
            dr *= input_shape[3] / 512
        sc = StatisticScale(form=form, divide_rate=dr, strline='=' * 120, indent=indent)
        sc.step(model, (input, ), **kargs)
        print(f'\nStatistic of {model_name}, input:{input_shape}{dr_str}')
        sc.print()
        print('\n\n')


# StatisticModel.set_batch_size(1)