#!/usr/bin/env python3
# Relocate qt.conf after conda-unpack for Windows (see SIFT.bat)
# example qt.conf file before relocation:
#
# [Paths]
# Prefix = C:/tools/miniconda3/envs/test/Library
# Binaries = C:/tools/miniconda3/envs/test/Library/bin
# Libraries = C:/tools/miniconda3/envs/test/Library/lib
# Headers = C:/tools/miniconda3/envs/test/Library/include/qt
# TargetSpec = win32-msvc
# HostSpec = win32-msvc

import sys
import os
import re


def main(base):
    fn_qt_conf = os.path.join(base, 'qt.conf')
    with open(fn_qt_conf, 'rt') as fob:
        txt = fob.read()
        fob.close()
    src, = re.findall(r'Prefix\s*=\s*(.*?)Library', txt, re.MULTILINE)
    dst = base.replace('\\', '/')
    if not dst.endswith('/'):
        dst += '/'
    print(f"Relocating qt.conf from {src} to {dst}")
    new_txt = txt.replace(src, dst)
    with open(fn_qt_conf, 'wt') as fob:
        fob.write(new_txt)
        fob.close()


if __name__ == '__main__':
    main(*sys.argv[1:])
