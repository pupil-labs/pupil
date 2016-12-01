# -*- mode: python -*-

import platform, sys, os, os.path, numpy, ntpath,glob

av_hidden_imports = ['av.format','av.packet','av.buffer','av.bytesource','av.frame','av.stream','av.descriptor','av.plane','av.audio.plane','av.container.streams','av.dictionary', 'av.audio.stream','av.subtitles','av.subtitles.stream','av.subtitles.subtitle','av.video.reformatter','av.video.plane','av.option']
pyglui_hidden_imports = ['pyglui.pyfontstash.fontstash','pyglui.cygl.shader','pyglui.cygl.utils']

block_cipher = None
import sys, os, os.path

np_path = os.path.dirname(numpy.__file__)


np_dlls = glob.glob(np_path + '/core/*.dll')

np_dll_list = [] 

for dll_path in np_dlls:
        dll_p, dll_f = ntpath.split(dll_path)
        np_dll_list += [(dll_f, dll_path, 'BINARY')]


system_path = os.path.join(os.environ['windir'], 'system32')

python_path = None
package_path = None
for path in sys.path:
        if path.endswith("scripts"):
                python_path = os.path.abspath(os.path.join(path, os.path.pardir))
        elif path.endswith("site-packages"):
                lib_dir = os.path.abspath(os.path.join(path, os.path.pardir))
                python_path = os.path.abspath(os.path.join(lib_dir, os.path.pardir))
                package_path = path

# a = Analysis(['../../pupil_src/capture/main.py'],
#                    pathex=['../../pupil_src/shared_modules/'],
#                    hiddenimports=['pyglui.cygl.shader']+scipy_imports+av_hidden_imports,
#                    hookspath=None,
#                    runtime_hooks=None,
#               excludes=['pyx_compiler','matplotlib'])

scipy_imports = ['scipy.integrate']
scipy_imports += ['scipy.integrate._ode', 'scipy.integrate.quadrature', 'scipy.integrate.odepack', 'scipy.integrate._odepack', 'scipy.integrate.quadpack', 'scipy.integrate._quadpack']
scipy_imports += ['scipy.integrate.vode', 'scipy.integrate.lsoda', 'scipy.integrate._dop', 'scipy.special._ufuncs', 'scipy.special._ufuncs_cxx']
a = Analysis(['main.py'],
             pathex=['../shared_modules/'],
             binaries=None,
             datas=None,
            hiddenimports=pyglui_hidden_imports+scipy_imports+av_hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='pupil_capture',
          resources=['pupil-capture.ico,ICON'],
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               [('glfw3.dll', 'glfw3.dll','BINARY')],
               [('OpenSans-Regular.ttf', os.path.join(package_path, 'pyglui/OpenSans-Regular.ttf'),'DATA')],
               [('Roboto-Regular.ttf', os.path.join(package_path, 'pyglui/Roboto-Regular.ttf'),'DATA')],
               [('fontawesome-webfont.ttf', os.path.join(package_path, 'pyglui/fontawesome-webfont.ttf'),'DATA')],
               np_dll_list,
               strip=False,
               upx=True,
               name='main')
