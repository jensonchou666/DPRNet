from collections import OrderedDict
import sys, os

from .utils import *


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


#region 之前出的问题
# class LayerNode(TreeNode):

#     def __init__(self, key=None, colmns={}, parents=None, seat=-1):
#         super().__init__(parents, seat)
#         self.key = key
#         self.columns = colmns  # OrderedDict()


#     def add_layer(self, key=None, colmns={}):
#         layer = LayerNode(key, colmns)
#         super().add_child(layer)
#         return layer
#endregion
class LayerNode(TreeNode):

    def __init__(self, parents=None, seat=-1):
        super().__init__(parents, seat)
        #        self.key = key
        self.columns = {}  # OrderedDict()

    def add_layer(self):
        layer = LayerNode()
        self.add_child(layer)
        return layer





class Table():
    default_show_syle = {
        'index': True,
        'outer_line': '=' * 80,  # set None -> unvisible
        "inner_line": '-' * 80,
        'split_v': '',  # '|'
        'separator': ''  # ','
    }

    def __init__(self):
        super().__init__()
        self.columns_info = OrderedDict()
        self.show_syle = Table.default_show_syle

    def aligned(col_info, value):
        s0 = '{{:<{}}}'
        if 'align' in col_info:
            if col_info['align'] == 'left':
                s0 = '{{:<{}}}'
            elif col_info['align'] == 'right':
                s0 = '{{:>{}}}'
            elif col_info['align'] == 'center':
                s0 = '{{:^{}}}'
        return s0.format(col_info['indent']).format(value) + ' '

    def print_line(self, key="outer_line"):
        if isinstance(self.show_syle[key], str):
            print(self.show_syle[key])

    def show(self):
        #if self.show_syle['auto_expand_indent']:

        self._auto_indent = True
        _stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        for info in self.columns_info.values():
            if 'indent' not in info:
                info['indent'] = 1
                if 'interval' not in info:
                    info['interval'] = 1
            if 'interval' in info:
                info['indent'] = max(
                    info['indent'],
                    len(str(info['title'])) + info['interval'])

        self.show_every_row()

        sys.stdout = _stdout
        self._auto_indent = False

        self._show()

    def show_every_row(self):
        pass

    def _show(self):
        self.print_line()

        row = ''
        for info in self.columns_info.values():
            row += Table.aligned(info, info['title'])
        if row != '':
            row = row[:-1]
        print(row)

        self.print_line("inner_line")

        self.show_every_row()

        self.print_line()


class MatrixTable(Table):

    default_show_syle = {
        'index': True,
        'outer_line': '=' * 80,  # set None -> unvisible
        "inner_line": '-' * 80,
        'split_v': '',  # '|'
        'separator': ''  # ','
    }

    def __init__(self):
        super().__init__()
        self.values = []
        self.show_syle.update(MatrixTable.default_show_syle)

    def show(self):
        pass


class TreeTable(Table):

    default_show_syle = {
        'index': True,
        'outer_line': '=' * 80,  # set None -> unvisible
        "inner_line": '-' * 80,
        'split_v': '',  # '|'
        'separator': '',  # ','

        'before_key': ['├─', '└─', '┌─'],
        'pre_unit': '|' + ' ' * 2,
        'top_down': True,
        'print_pos': True,

        # 'auto_expand_indent': True
    }

    def __init__(self):
        super().__init__()

        self.root = LayerNode()
        self.node = self.root

        self.show_syle.update(TreeTable.default_show_syle)

    def format_first_column(self, value, col_info, node, depth, index, seat):
        name = ''
        if self.show_syle['index']:
            name = f'{index:<4}'
        if value is None:
            value = '(None-Name)'
        name += f'{self.strprefix(depth)}{value}'
        if self.show_syle['print_pos']:
            name += f": {depth}-{seat}"
        return name

    def strprefix(self, depth):
        bk = self.show_syle['before_key']
        if depth == 0:
            return bk[0]
        if self.top_down:
            return self.str_pre_unit * (depth) + bk[1]
        else:
            return self.str_pre_unit * (depth) + bk[2]

    def print_row(self, node: LayerNode, depth, index, seat):
        row = ''
        for k, v in self.columns_info.items():
            show = False
            if k not in node.columns:
                value = None
            elif not isinstance(node.columns[k], dict):
                value = node.columns[k]
            elif 'show' in node.columns[k]:
                value = node.columns[k]['show']
                show = True
            else:
                value = node.columns[k]['value']

            if not show:
                if 'format_func' in v:
                    f = v['format_func']
                    value = f(value,
                              col_info=(k, v),
                              node=node,
                              depth=depth,
                              index=index,
                              seat=seat)
                elif 'format' in v:
                    value = v['format'].format(v)
                elif value is None:
                    value = '--'
                elif 'precision' in v:
                    value = format_float(value, v['precision'])
                #TODO 在这里添加Format的方式
                else:
                    value = str(value)
            # if 'align' in v:
            #     if v['align'] == 'left':
            #         s0 = '{{:<{}}}'
            #     elif v['align'] == 'right':
            #         s0 = '{{:>{}}}'
            #     elif v['align'] == 'center':
            #         s0 = '{{:^{}}}'
            else:
                s0 = '{{:<{}}}'
            if 'interval' in v:
                v['indent'] = max(v['indent'], len(str(value) + v['interval']))
            row += s0.format(
                v['indent']).format(value) + self.show_syle['split_v'] + ' '
        if row != '':
            row = row[:-1]
        print(row)

    def show(self):
        #if self.show_syle['auto_expand_indent']:

        self._auto_indent = True
        _stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        for info in self.columns_info.values():
            if 'indent' not in info:
                info['indent'] = 1
                if 'interval' not in info:
                    info['interval'] = 1
            if 'interval' in info:
                info['indent'] = max(
                    info['indent'],
                    len(str(info['title'])) + info['interval'])

        for n in self.root.childen:
            n.forward(self.print_row)
        sys.stdout = _stdout
        self._auto_indent = False

        self._show()

    def _show(self):

        self.print_line()

        row = ''
        for info in self.columns_info.values():
            row += '{{:<{}}}'.format(info['indent']).format(
                info['title']) + ' '
        if row != '':
            row = row[:-1]
        print(row)

        self.print_line("inner_line")

        for n in self.root.childen:
            assert type(n) is LayerNode
            if self.top_down:
                n.forward(self.print_row)
            else:
                n.backward(self.print_row)

        self.print_line()

    def set_column_order(self, *key_list):
        col_new = OrderedDict()
        for k in key_list:
            assert k in self.columns_info
            col_new[k] = self.columns_info[k]
        self.columns_info = col_new

    def register(self, column_key, info_dict):
        self.columns_info[column_key] = info_dict

    def add_item(self, k, v):
        self.node.columns[k] = v

    def add_float_item(self, key, value):
        if 'unit' in self.columns_info[key]:
            unit = self.columns_info[key]['unit']
            v = unit_div(value, unit)
        else:
            unit = ''
            v = value
        if 'precision' in self.columns_info[key]:
            v = format_float(v, self.columns_info[key]['precision'])
        self.add_item(key, {'value': value, 'show': str(v) + unit})