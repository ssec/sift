# -*- mode: python -*-

import sys
from PyInstaller.compat import is_win, is_darwin, is_linux
from PyInstaller.utils.hooks import collect_submodules
import vispy.glsl
import vispy.io
import satpy

block_cipher = None
exe_name = "SIFT"
main_script_pathname = os.path.join("sift", "__main__.py")
_script_base = os.path.dirname(os.path.realpath(sys.argv[0]))

data_files = [
    (os.path.dirname(vispy.glsl.__file__), os.path.join("vispy", "glsl")),
    (os.path.join(os.path.dirname(vispy.io.__file__), "_data"), os.path.join("vispy", "io", "_data")),
    (os.path.join(os.path.dirname(satpy.__file__), "etc"), os.path.join('satpy', 'etc')),
]

for shape_dir in ["ne_50m_admin_0_countries", "ne_110m_admin_0_countries", "ne_50m_admin_1_states_provinces_lakes", "fonts", "colormaps"]:
    data_files.append((os.path.join("sift", "data", shape_dir), os.path.join("sift_data", shape_dir)))

hidden_imports = [
    "vispy.ext._bundled.six",
    "vispy.app.backends._pyqt4",
    "sqlalchemy.ext.baked",
    "satpy",
    "skimage",
    "skimage.measure",
] + collect_submodules("rasterio") + collect_submodules('satpy')
if is_win:
    hidden_imports += collect_submodules("encodings")
# PyGrib hidden import
if not is_win:
    hidden_imports += ['ncepgrib2']


# Add missing shared libraries
binaries = []
if is_linux:
    lib_dir = sys.executable.replace(os.path.join("bin", "python"), "lib")
    binaries += [(os.path.join(lib_dir, 'libfontconfig*.so'), '.')]

a = Analysis([main_script_pathname],
             pathex=[_script_base],
             binaries=binaries,
             datas=data_files,
             hiddenimports=hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=["tkinter"],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
# FIXME: Remove the console when all diagnostics are properly shown in the GUI

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name=exe_name,
          debug=False,
          strip=False,
          upx=True,
          console=True )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name=exe_name)

if is_darwin:
    app = BUNDLE(coll,
                 name=exe_name + '.app',
                 icon=None,
                 bundle_identifier=None,
                 info_plist={
                     'LSBackgroundOnly': 'false',
                 })
