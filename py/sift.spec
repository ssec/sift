# -*- mode: python -*-

block_cipher = None

from llvmlite.binding.ffi import _lib_dir, _lib_name
import vispy.glsl
import vispy.io
data_files = [
    (os.path.join(_lib_dir, _lib_name), '.'),
    (os.path.join(_lib_dir, "MSVCP120.dll"), '.'),
    (os.path.join(_lib_dir, "MSVCR120.dll"), '.'),
    (os.path.dirname(vispy.glsl.__file__), os.path.join("vispy", "glsl")),
    (os.path.join(os.path.dirname(vispy.io.__file__), "_data"), os.path.join("vispy", "io", "_data")),
]
for shape_dir in ["ne_50m_admin_0_countries", "ne_110m_admin_0_countries"]:
    data_files.append((os.path.join("cspov", "data", shape_dir), os.path.join("cspov", "data", shape_dir)))

a = Analysis(['cspov\\__main__.py'],
             pathex=['Z:\\repos\\repos\\git\\CSPOV\\py'],
             binaries=None,
             datas=data_files,
             hiddenimports=["vispy.app.backends._pyqt4", "PyQt4.QtNetwork", "scipy.linalg", "scipy.linalg.cython_blas", "scipy.linalg.cython_lapack", "scipy.integrate"],
             hookspath=None,
             runtime_hooks=None,
             excludes=["tkinter"],
             win_no_prefer_redirects=None,
             win_private_assemblies=None,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
# FIXME: Remove the console when all diagnostics are properly shown in the GUI
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='SIFT',
          debug=False,
          strip=None,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='SIFT')
