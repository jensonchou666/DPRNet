from collections import OrderedDict
import sys, os
from typing import Optional, Dict, Iterable, Union
# from .utils import *


def format_float(v, p=4):
    return '{{:.{}f}}'.format(p).format(v)
def format_number(v, bits=6):
    return '{{:.{}}}'.format(bits).format(v)

def unit_div(i, unit: str):
    if unit.upper() == 'M':
        return i / 1000**2
    elif unit.upper() == 'G':
        return i / 1000**3
    elif unit.upper() == 'T':
        return i / 1000**4
    else:
        return i


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

    def default_show_syle():
        return {
            'row_number': True,
            'outer_line': '=',  # set None -> unvisible
            "inner_line": '-',
            'outer_v_line': '|',
            'line_len_extra': 2,
            'split_v': '',  # '|'
            'separator': '',  # ','
            'align': None,
            'default_interval': 2,
            'none_show': '--',
            'title': None,
        }

    # To complete it !
    def print_every_row(self):
        pass

    # To complete it !
    def add_item(self, k, v):
        pass

    def __init__(self):
        super().__init__()
        self.columns_info = OrderedDict()
        self.show_syle = Table.default_show_syle()
        
        self.max_row_num = 0
        self.__indent_row_num = 3
        self._auto_indent = False

    def aligned(col_info, value):
        s0 = '{{:<{}}}'
        if 'align' in col_info:
            if col_info['align'] == 'left':
                s0 = '{{:<{}}}'
            elif col_info['align'] == 'right':
                s0 = '{{:>{}}}'
            elif col_info['align'] == 'center':
                s0 = '{{:^{}}}'
        return s0.format(col_info['indent']).format(value)

    def print_line(self, key="outer_line"):
        indent = self.show_syle['line_len_extra']
        for cinfo in self.columns_info.values():
            indent += cinfo['indent']
        if isinstance(self.show_syle[key], str):
            print(self.show_syle[key] * indent)

    def print_row(self, columns, row_num, **kwargs):
        if columns is None:
            return
        self.max_row_num = max(row_num, self.max_row_num)
        outer_v_line = self.show_syle['outer_v_line']
        row = outer_v_line
        i_last_col = len(self.columns_info) - 1
        for i, (k, cinfo) in enumerate(self.columns_info.items()):
            show = False
            if k not in columns:
                value = None
            elif not isinstance(columns[k], dict):
                value = columns[k]
            elif 'show' in columns[k]:
                value = columns[k]['show']
                show = True
            elif 'value' in columns[k]:
                value = columns[k]['value']
            else:
                raise ValueError("can't find value in columns[k]")

            if not show:
                if 'format_func' in cinfo:
                    value = cinfo['format_func'](value,
                                                 columns=columns,
                                                 col_info=(k, cinfo),
                                                 row_num=row_num,
                                                 **kwargs)
                elif value is None:
                    value = self.show_syle['none_show']

                elif 'format' in cinfo:
                    #TODO 这里只能format一个value
                    value = cinfo['format'].format(value)

                elif 'precision' in cinfo:
                    value = format_float(value, cinfo['precision'])
                #TODO 在这里添加Format的方式
                else:
                    value = str(value)
            if i == i_last_col:
                sep = ''
                split_v = ''
            else:
                sep = self.show_syle['separator']
                split_v = self.show_syle['split_v']

            if i == 0 and self.show_syle['row_number']:
                str_row_num = "{{:<{}}}".format(
                    self.__indent_row_num).format(row_num)
                value = f'{str_row_num}{value}'

            value = f'{value}{sep}'
            if self._auto_indent and 'interval' in cinfo:
                cinfo['indent'] = max(cinfo['indent'],
                                      len(str(value)) + cinfo['interval'])
            row += Table.aligned(cinfo, value) + split_v
        print(row + outer_v_line)


    def show(self, print_tile=True):
        #if self.show_syle['auto_expand_indent']:

        align = self.show_syle['align']
        if align is not None:
            if align in ['left', 'right', 'center']:
                self.set_align(align)

        self._auto_indent = True
        _stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        for info in self.columns_info.values():
            if 'indent' not in info:
                info['indent'] = 1
                if 'interval' not in info:
                    info['interval'] = self.show_syle['default_interval']
            if 'interval' in info:
                info['indent'] = len(str(info['title'])) + info['interval']
                # info['indent'] = max(
                #     info['indent'],
                #     len(str(info['title'])) + info['interval'])

        self.max_row_num = 0
        self.__indent_row_num = 1
        self.print_every_row()
        self.__indent_row_num = len(str(self.max_row_num)) + 1
        sys.stdout = _stdout
        self._auto_indent = False

        if print_tile:
            self.print_title()
        self.print_every_row()
        if print_tile:
            self.print_line()


    def print_title(self):
        outer_v_line = self.show_syle['outer_v_line']
        row = outer_v_line
        for info in self.columns_info.values():
            row += Table.aligned(info,
                                 info['title']) + self.show_syle['split_v']
        Lv = max(len(self.show_syle['split_v']), 1)
        # if row != '':
        #     row = row[:-Lv]

        titlerow = row + outer_v_line

        if self.show_syle['title'] is not None:
            ttl = self.show_syle['title']
            l1 = int(len(titlerow) - len(ttl)) / 2 - 1
            l1 = int(l1)
            print("-" * l1, ttl, "-" * l1)
        self.print_line()
        print(titlerow)
        self.print_line("inner_line")

    def set_columns_order(self, *key_list):
        col_new = OrderedDict()
        for k in key_list:
            assert k in self.columns_info
            col_new[k] = self.columns_info[k]
        self.columns_info = col_new

    def set_align(self, _align='left'):
        for v in self.columns_info.values():
            v['align'] = _align

    def register_columns(self, columns_info: Optional[Union[Dict, Iterable]]):
        if isinstance(columns_info, dict):
            for k, v in columns_info.items():
                self.register(k, v)
        else:
            for k, v in columns_info:
                self.register(k, v)

    def register_columns_simple(self, form: Iterable, unified_info: Dict):
        for i in form:
            info = {'title': str(i)}
            info.update(unified_info)
            self.register(i, info)

    def register(self, column_key, info_dict):
        self.columns_info[column_key] = info_dict

    def add_items(self, items: Optional[Union[Dict, Iterable]]):
        if isinstance(items, dict):
            for k, v in items.items():
                self.add_item(k, v)
        else:
            for k, v in items:
                self.add_item(k, v)

    def add_float_item(self, key, value):
        if value is None:
            self.add_item(key, None)
            return
        info = self.columns_info[key]
        if 'unit' in info:
            unit = info['unit']
            v = unit_div(value, unit)
        else:
            unit = ''
            v = value
        if 'unit_show' in info and not info['unit_show']:
            unit = ''
        if 'numbers' in info:
            v = format_number(v, info['numbers'])
        if 'precision' in info:
            v = format_float(v, info['precision'])
        self.add_item(key, {'value': value, 'show': str(v) + unit})



class MatrixTable(Table):

    def __init__(self, values=None):
        super().__init__()
        if values is None:
            self.values = []
        else:
            self.values = values
        self.show_syle = MatrixTable.default_show_syle()

    def print_every_row(self):
        for i, columns in enumerate(self.values):
            self.print_row(columns, row_num=i)

    def add_item(self, k, v):
        self.values[-1][k] = v

    def new_row(self):
        self.values.append({})

    def show(self, values=None, print_tile=True):
        if values is not None:
            v = self.values
            self.values = values
            super().show(print_tile)
            self.values = v

        else:
            return super().show(print_tile)

    def sort(self, forms, values=None):
        # forms: [('a', 'max'), ('b', 'min'), ('c', 'min', 'value')]
        if values is None:
            values = self.values
        if len(values) == 0:
            return

        def swap(i, j):
            values[i], values[j] = values[j], values[i]

        def f_max(i, j, clm):
            if values[j][clm] > values[i][clm]:
                swap(i, j)

        def f_min(i, j, clm):
            if values[j][clm] < values[i][clm]:
                swap(i, j)

        value_key = None

        def f_max_key(i, j, clm):
            if values[j][clm][value_key] > values[i][clm][value_key]:
                swap(i, j)

        def f_min_key(i, j, clm):
            if values[j][clm][value_key] < values[i][clm][value_key]:
                swap(i, j)

        for item in forms:
            clm, mode = item[:2]
            if len(item) > 2:
                value_key = item[2]
                if mode == 'max':
                    func = f_max_key
                else:
                    func = f_min_key
            else:
                if mode == 'max':
                    func = f_max
                else:
                    func = f_min

            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    if clm not in values[j]:
                        continue
                    elif clm not in values[i]:
                        swap(i, j)
                    else:
                        func(i, j, clm)

class TreeTable(Table):

    def default_show_syle():
        a = Table.default_show_syle()
        a.update({
            'before_key': ['├─', '└─', '┌─'],
            'pre_unit': '|' + ' ' * 2,
            'top_down': True,
            'print_pos': True,
            'skip_root': False
            # 'auto_expand_indent': True
        })
        return a

    def __init__(self, show_syle=None):
        super(TreeTable, self).__init__()

        self.root = LayerNode()
        self.node = self.root
        if show_syle is None:
            show_syle = TreeTable.default_show_syle()
        self.show_syle = show_syle

    def format_first_column(self, value, columns, col_info, row_num, node,
                            depth, seat):
        name = ''
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
        if self.show_syle['top_down']:
            return self.show_syle['pre_unit'] * (depth) + bk[1]
        else:
            return self.show_syle['pre_unit'] * (depth) + bk[2]

    def print_row(self, node: LayerNode, depth, index, seat):
        super().print_row(columns=node.columns,
                          row_num=index,
                          node=node,
                          depth=depth,
                          seat=seat)

    def print_every_row(self):
        top_down = self.show_syle['top_down']

        if self.show_syle['skip_root']:
            for n in self.root.childen:
                if top_down:
                    n.forward(self.print_row)
                else:
                    n.backward(self.print_row)
        else:
            if top_down:
                self.root.forward(self.print_row)
            else:
                self.root.backward(self.print_row)

    def add_item(self, k, v):
        self.node.columns[k] = v

    def register(self, column_key, info_dict):
        super().register(column_key, info_dict)

        if len(self.columns_info) == 1:
            self.columns_info[column_key][
                'format_func'] = self.format_first_column

    def set_align(self, _align='left'):
        for i, v in enumerate(self.columns_info.values()):
            if i == 0:
                continue
            v['align'] = _align


if __name__ == "__main__":
    tt = TreeTable()

    # tt.register_columns([
    #     ('name', {'title': 'NAME', 'interval': 5}),
    #     ('a', {'title': 'param-a', 'interval': 4}),
    #     ('b', {'title': 'param-b', 'interval': 4})
    # ])
    tt.register_columns({
        'name': {
            'title': 'NAME',
            'interval': 5
        },
        'a': {
            'title': 'param-a',
            'interval': 4,
            'format': '{:.2f}'
        },
        'b': {
            'title': 'param-b',
            'interval': 4
        }
    })
    # tt.set_columns_order('name', 'b', 'a')
    tt.add_item('a', 4.22121)
    tt.add_item('name', 'Layer1')
    tt.add_item('b', {'show': '5.5M'})

    n1 = tt.node.add_layer()
    n2 = tt.node.add_layer()

    tt.node = n1
    tt.add_item('name', 'Layer22222222222')
    tt.add_item('a', 3.2223)
    #tt.add_item('b', '2.5M')

    tt.node = n2
    tt.add_item('name', 'Layer3')
    tt.add_item('a', 2.111)
    tt.add_item('b', '3.0M')

    tt.show()
    print()

    tt.show_syle['outer_line'] = None
    tt.show_syle['separator'] = ','

    tt.show()
    print()

    tt.set_align('center')
    tt.show_syle['index'] = False
    tt.show_syle['separator'] = ''
    tt.show_syle['split_v'] = '|'
    tt.show_syle['print_pos'] = False
    tt.show()