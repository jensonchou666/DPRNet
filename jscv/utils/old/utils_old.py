import re
import inspect
import sys, os
import thop

import torch


class GlobalEnv:
    on_training = False


env = GlobalEnv()


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)



class StatisticOperate:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.printlist = [strline, strfirst, strline]




    def profile_quiet(model, inputs):
        _stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        flops, params = thop.profile(model, inputs)
        sys.stdout = _stdout

    def do_profile_self(self):
        self._do_sum = False

    def dont_sum(self):
        self._do_sum = False

        #     self.total_flops = 0.0
        #   self.total_params = 0.0

        # self.strprefix = strprefix * (layer + 1) + '└─'



class TraverseModel:

    def __init__(self):
        self.depth = 0
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


    def step(self,
             model: torch.nn.Module,
             inputs,
             **kargs):

        self._enter_step_(model, inputs, **kargs)

        if hasattr(model, 'traverse'):
            self.depth += 1
            self._before_traverse_(model, inputs, **kargs)
            model.traverse(self, *inputs)
            self.depth -= 1
            self._after_traverse_(model, inputs, **kargs)
        else:
            self._none_traverse_(model, inputs, **kargs)

        return self._exit_step_(model, inputs, **kargs)

from collections import OrderedDict


class TreeNode:
    def __init__(self):
        self.childen = []

    def add_child(self, node):
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
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.columns = OrderedDict()

    def add_layer(self, key, ):
        self.childen.append(LayerNode(key))


class StatisticModel(TraverseModel):

    indent_layer = 40

    def __init__(self, top_down=True, strline='=' * 80, indent=2):
        super().__init__()
        self.str_unitV = '|' + ' ' * indent
        self.strline = strline

        #TODO self.root = LayerNode()
        self.top_down = top_down

        self.columns_info = OrderedDict()
        self.columns_info['key'] = {
            'title': 'Model Layer',
            'indent': StatisticModel.indent_layer,
        }
        # self.printlist = [strline, cl, strline]

    def strprefix(self, depth):
        if depth == 0:
            return '──'
        if self.root_top:
            return self.str_unitV * (depth + 1) + '└─'
        else:
            return self.str_unitV * (depth + 1) + '┌─'

    def print_row(self, node: LayerNode, depth, index, seat):
        row = ''
        

    def print(self, file=sys.__stdout__):
        print(self.strline)
        row = ''
        for info in self.columns_info.values():
            row += '{{:<{}}}'.format(info['indent']).format(info['title'])
        print(row)
        print(self.strline)
        c1 = next(iter(self.columns_info.values()))
        
        if self.top_down:
            self.root.forward(self.print_row)
        else:
            self.root.backward(self.print_row)

        for i, depth in enumerate(self.depths):
            row = ''
            value = c1['values'][i]
            value = self.strprefix(depth) + str(value)
            row += '{{:<{}}}'.format(c1['indent']).format(value)
            iter_clm = iter(self.columns_info.values())
            next(iter_clm)
            for cj in iter_clm:
                value = cj['values'][i]
                if 'precision' in cj:
                    value = '{{:.{}f}}'.format(cj['precision']).format(value)
                else:
                    value = str(value)
                row += '{{:<{}}}'.format(cj['indent']).format(value)
            print(row)
        print(self.strline)


    def new_row(self):
        self.depths.append(self.depth)
        self.levels.append({})

    def add_item(self, title, value):
        self.levels[-1][title] = value

    def add_items(self, model: torch.nn.Module, inputs, **kargs):
        if 'model_name' in kargs:
            name = kargs['model_name']
        elif 'name' in kargs:
            name = kargs['name']
        else:
            name = model._get_name()
        self.add_item('layer', name)

    def _enter_step_(self, model, inputs, **kargs):
        self.levels.append({})

    def _before_traverse_(self, model, inputs, **kargs):
        pass

    def _after_traverse_(self, model, inputs, **kargs):
        pass

    def _none_traverse_(self, model, inputs, **kargs):
        pass

    def _exit_step_(self, model, inputs, **kargs):
        a = self.levels.pop()



class StatisticScale(StatisticModel):
    
    indent_layer = 40

    indent_flops = 12
    precision_flops = 4

    indent_params = 12
    precision_params = 4


    def register_items(self, form):
        columns = self.columns_info
        if 'flops' in form:
            columns['flops'] = {
                'title': 'FLOPS',
                'indent': StatisticScale.indent_flops,
                'precision': StatisticScale.precision_flops,
            }
        if 'params' in form:
            columns['params'] = {
                'title': 'PARAMS',
                'indent': StatisticScale.indent_params,
                'precision': StatisticScale.precision_params,
            }

    def __init__(self, form=['flops', 'params'], strline='=' * 80, indent=2):
        super().__init__(strline, indent)
        self.register_items(form)


    def _enter_step_(self, model, inputs, **kargs):
        super()._enter_step_(model, inputs, **kargs)
        #__add__

    def _before_traverse_(self, model, inputs, **kargs):
        pass

    def _after_traverse_(self, model, inputs, **kargs):
        pass

    def _none_traverse_(self, model, inputs, **kargs):
        pass

    def _exit_step_(self, model, inputs, **kargs):
        #__add__

        super()._exit_step_(model, inputs, **kargs)












        flops, params = StatisticModel.profile_quiet(model, inputs)

        flops = flops / 1000**3 / self.batch_size
        params = params / 1000**2 / self.batch_size
        self.total_flops += flops
        self.total_params += params

        if model_name is None:
            model_name = model._get_name()

        pre_name = self.strprefix + model_name

        self.printlist.append(f'{pre_name:<40} {flops:.2f}G, {params:.2f}M')
        return model(*inputs)

    def stat(model: torch.nn.Module, input_shape=(2, 3, 1024, 1024), model_name=None):
        input = torch.randn(*input_shape)
        if model_name is None:
            model_name = model._get_name()
        print(
                f'\nStatistics of {model_name}, input:{input.shape}\n'
        )
        #StatisticModel.set_batch_size()
        sm = StatisticModel(input_shape[0])
        sm.step(model, (input,))

    def print(self, file=sys.__stdout__, total_name='Total', sum_total=True):
        if sum_total:
            total = f'──{total_name:<40} FLOPs={self.total_flops:.2f}G, Params={self.total_params:.2f}M'
            print(total, file=file)
        for s in self.printlist:
            print(s, file=file)


# StatisticModel.set_batch_size(1)