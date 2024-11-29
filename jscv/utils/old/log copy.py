from typing import Any, Dict, Optional, Union
import sys, os
from .table import MatrixTable

import shutil


class Logger(MatrixTable):

    v_prefix = 'version_'

    def last_version(save_dir):
        files = os.listdir(save_dir)
        version = -1
        for f in files:
            if f.startswith(Logger.v_prefix):
                _v = f[len(Logger.v_prefix):]
                if _v.isdigit():
                    version = max(version, int(_v))
        return version

    def __init__(
            self,
            save_dir: str,
            log_name: str = 'log',
            version: Optional[Union[int, str]] = None,
            #resume_epoch='last',
            use_stdout=False,
            suffix='.txt',
            save_old_logfile=True  # if resume, save old logfile
    ):
        super().__init__()
        self.use_stdout = use_stdout

        if self.use_stdout:
            self.filename = 'sys.stdout'
            self.file = sys.stdout
        else:
            if version is None:
                version = Logger.last_version(save_dir) + 1
                self.do_resume = False
            else:
                self.do_resume = True
                #self.resume_epoch = resume_epoch
            if isinstance(version, int):
                version = f'{Logger.v_prefix}{version}'
            self.save_dir = save_dir
            self.log_name = log_name
            self.suffix = suffix
            self.filename = log_name + suffix
            self.version_name = version
            self.logdir = os.path.join(save_dir, version)
            self.save_old_logfile = save_old_logfile

        self.show_syle['row_number'] = False
        # self.show_syle['default_interval'] = 4

        self._do_concate = False

    def max_old_log_version(self):
        files = os.listdir(self.logdir)
        version = 0
        for f in files:
            n_old = self.log_name + '('
            if f.startswith(n_old):
                a = f[len(n_old):].split(')')[0]
                if a.isdigit() and int(a) > 0:
                    version = max(version, int(a))
        return version

    def begin_log(self):
        self.new_row()
        self.init_log_file()
        if not self._do_concate:
            self.display_title()

    def display_title(self):
        _std = sys.stdout
        sys.stdout = self.file
        self.print_line()
        self.print_title()
        self.print_line('inner_line')
        sys.stdout = _std
        self.file.flush()

    def init_log_file(self):
        if not self.use_stdout:
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)

            self.filepath = os.path.join(self.logdir, self.filename)

            if self.do_resume and self.save_old_logfile:
                if os.path.exists(self.filepath):
                    v = self.max_old_log_version() + 1
                    old = self.log_name + f'({v})' + self.suffix
                    shutil.copyfile(self.filepath, os.path.join(old))
                    self._do_concate = True

            self.file = open(self.filepath, 'w+')

    def log(self, flush=True, **kwargs):
        if self._do_concate and not self.use_stdout:
            self._do_concate = False

            assert 'epoch' in self.columns_info, "log file doesn't have epoch column, can't concate"
            assert 'epoch' in self.values[-1]
            title = self.columns_info['epoch']['title']
            epoch = self.values[-1]['epoch']
            lines = self.file.readlines()
            line_pre = []
            log_begin = False
            for line in lines:
                s = line.strip().split(' ')
                if len(s) == 0:
                    line_pre.append(line)
                    continue
                s = s[0]
                if not log_begin and s == title:
                    log_begin = True
                if log_begin:
                    if s.isdigit():
                        if int(s) < epoch:
                            line_pre.append(line)
                    else:
                        line_pre.append(line)

            if len(line_pre) > 0:
                self.file.writelines(line_pre)
            else:
                self.display_title()

        _std = sys.stdout
        sys.stdout = self.file
        self.print_row(self.values[-1], len(self.values))
        sys.stdout = _std

        if flush:
            self.file.flush()

        self.new_row()