import sys, os
from .profile import profile
import torch
import copy
from collections import OrderedDict, Iterable

from .utils import *
from .table import TreeTable

def profile_quiet(model, inputs, input_kargs):
    _stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # ?? 使用 thop.profile() 可以测model的flops和params
    # ?? 但会向model的state_dict添加一些变量
    # ?? 导致之后model加载state_dict权重失败
    # ?? 故在使用profile()之前需要对model深拷贝一份
    #flops, params = 0, 0
    flops, params = profile(model, inputs, input_kargs)

    sys.stdout = _stdout
    return flops, params


class TraverseModel:

    def __init__(self):
        pass
        #self.opt = StatisticOperate

    def _enter_step_(self, model, inputs, in_kargs, **kargs):
        pass

    # def _before_traverse_(self, model, inputs, in_kargs, **kargs):
    #     pass

    # def _after_traverse_(self, model, inputs, in_kargs, **kargs):
    #     pass

    def _none_traverse_(self, model, inputs, in_kargs, **kargs):
        pass

    def _exit_step_(self, model, inputs, in_kargs, **kargs):
        pass

    def void_step(self, name, **kargs):
        self._enter_step_(None, None, void_block=True, name=name, **kargs)
        return self._exit_step_(None, None, void_block=True, name=name, **kargs)

    def step(self, model: torch.nn.Module, inputs, in_kargs={}, **kargs):
        # define traverse()

        if not isinstance(inputs, tuple) and not isinstance(inputs, list):
            inputs = (inputs,)

        self._enter_step_(model, inputs, in_kargs, **kargs)

        do_t = True
        if 'do_traverse' in kargs:
            do_t = kargs['do_traverse']

        if hasattr(model, 'traverse') and do_t:
            #self._before_traverse_(model, inputs, in_kargs, **kargs)
            model.traverse(self, *inputs)
            #self._after_traverse_(model, inputs, in_kargs, **kargs)
        else:
            self._none_traverse_(model, inputs, in_kargs, **kargs)

        return self._exit_step_(model, inputs, in_kargs, **kargs)


class StatisticModel(TraverseModel, TreeTable):


    def __init__(self):
        TreeTable.__init__(self)

        self.columns_info['key'] = {
            'title': 'Model Layer',
            #'indent': 40,
            'interval': 5,
            'format_func': self.format_first_column
        }

    def _enter_step_(self, model, inputs, in_kargs, **kargs):

        self.forward_res = None

        if 'void_block' in kargs:
            if 'name' in kargs:
                name = kargs['name']
            else:
                name = 'void_block'
            self.node = self.node.add_layer()
            self.add_item('key', name)
            return

        if 'model_name' in kargs:
            name = kargs['model_name']
        elif 'name' in kargs:
            name = kargs['name']
        else:
            name = model._get_name()

        self.node = self.node.add_layer()
        self.add_item('key', name)
        self.node.stat_self_alone = False

    def _none_traverse_(self, model, inputs, in_kargs, **kargs):
        self.node.stat_self_alone = True

    def _exit_step_(self, model, inputs, in_kargs, **kargs):

        if 'void_block' in kargs:
            self.node = self.node.parents
            return

        self.statistic_common(model, inputs, in_kargs, **kargs)

        if self.node.stat_self_alone:
            self.statistic_alone(model, inputs, in_kargs, **kargs)
        else:
            self.statistic_sum(model, inputs, in_kargs, **kargs)

        self.node = self.node.parents

        if self.forward_res is not None:
            return self.forward_res
        return model(*inputs, **in_kargs)

    def statistic_common(self, model, inputs, in_kargs, **kargs):
        pass

    def statistic_sum(self, model, inputs, in_kargs, **kargs):
        pass

    def statistic_alone(self, model, inputs, in_kargs, **kargs):
        pass

    def statistic_self_alone(self):
        self.node.stat_self_alone = True


def tensor_shape_str(t):
    if isinstance(t, torch.Tensor):
        return str(list(t.shape[1:]))
    else:
        return 'not-tensor'

class StatisticScale(StatisticModel):

    def __init__(self,
                 form=['shape', 'out_shape', 'flops', 'params'],
                 divide_rate=1):
        super().__init__()

        self.register_items(form)

        self.divide_rate = divide_rate

    def register_items(self, form):
        columns = self.columns_info

        if 'shape' in form:
            columns['shape'] = {
                'title': 'Input Shape',
                'interval': 5,
            }

        if 'out_shape' in form:
            columns['out_shape'] = {
                'title': 'Output Shape',
                'interval': 5,
            }

        if 'flops' in form:
            columns['flops'] = {
                'title': 'Flops(G)',
                'interval': 5,
                'unit': 'G',
                'unit_show': False,
                'precision': 4,
            }
        if 'params' in form:
            columns['params'] = {
                'title': 'Params(M)',
                'interval': 5,
                'unit': 'M',
                'unit_show': False,
                'precision': 4,
            }



    def statistic_common(self, model, inputs, in_kargs, **kargs):
        cinfo = self.columns_info
        if 'shape' in cinfo:
            i0 = inputs[0]
            if isinstance(i0, list) or isinstance(i0, tuple):
                i0 = i0[0]
            shape = list(i0.shape[1:])
            s = tensor_shape_str(inputs[0])
            if len(inputs) > 1:
                s = f'{s}(0)/({len(inputs)})'
            self.add_item('shape', s)

        if 'out_shape' in cinfo:
            res = model(*inputs, **in_kargs)
            self.forward_res = res
            if isinstance(res, list) or isinstance(res, tuple):
                s = f'{tensor_shape_str(res[0])}(0)/({len(res)})'
            else:
                s = tensor_shape_str(res)
            self.add_item('out_shape', s)

    def statistic_sum(self, model, inputs, in_kargs, **kargs):

        cinfo = self.columns_info

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

    def statistic_alone(self, model, inputs, in_kargs, **kargs):
        #TODO 统计空模块、 定制统计

        cinfo = self.columns_info

        if 'flops' in cinfo or 'params' in cinfo:
            flops, params = profile_quiet(model, inputs, in_kargs)

        if 'flops' in cinfo:
            flops = flops / self.divide_rate
            self.add_float_item('flops', flops)
        if 'params' in cinfo:
            params = params / self.divide_rate
            self.add_float_item('params', params)



    def stat(model: torch.nn.Module,
             input=[2, 3, 512, 512],
             form=['flops', 'params', 'shape', 'out_shape'],
             show_style: dict = StatisticModel.default_show_syle(),
             model_name=None,
             device='gpu',
             normalize=False,  # result normalized to [2, 3, 512, 512]
             deepcopy_model=True,
             no_grad=True,
             **kargs):
        if isinstance(input, torch.Tensor):
            input_shape = input.shape
        elif isinstance(input, Iterable):
            input_shape = input
            input = torch.randn(*input_shape)
        if model_name is None:
            model_name = model._get_name()

        if deepcopy_model:
            # model = copy.deepcopy(model)
            import dill
            obj_bytes = dill.dumps(model)
            model = dill.loads(obj_bytes)

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
        sc = StatisticScale(form=form, divide_rate=dr)
        sc.show_syle = show_style
        sc.show_syle['skip_root'] = True

        sc.set_columns_order('key', *form)
        #TODO torch.no_grad()
        if no_grad:
            with torch.no_grad():
                sc.step(model, (input, ), **kargs)
        else:
            sc.step(model, (input, ), **kargs)

        if deepcopy_model:
            del model
        del input

        torch.cuda.empty_cache()
        

        print(f'\nStatistic of {model_name}, input:{input_shape}{dr_str}')
        sc.show()


"""

    def traverse(self, stat: StatisticModel, x):
        stat.statistic_self_alone()

"""