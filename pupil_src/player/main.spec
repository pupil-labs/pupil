# -*- mode: python -*-

import platform, sys, os, os.path, zmq, glob, ntpath

av_hidden_imports = ['av.format','av.packet','av.buffer','av.bytesource','av.frame','av.stream','av.descriptor','av.plane','av.audio.plane','av.container.streams','av.dictionary', 'av.audio.stream','av.subtitles','av.subtitles.stream','av.subtitles.subtitle','av.video.reformatter','av.video.plane','av.option']
pyglui_hidden_imports = ['pyglui.pyfontstash.fontstash','pyglui.cygl.shader','pyglui.cygl.utils']
pyndsi_hidden_imports = ['pyre']

block_cipher = None

system_path = os.path.join(os.environ['windir'], 'system32')

python_path = None
package_path = None

zmq_path = os.path.dirname(zmq.__file__)
print("ZMQ path = ", zmq_path)

zmq_p, zmq_lib = ntpath.split(glob.glob(zmq_path +  '/libzmq.*.pyd')[0])
print("ZMQ lib = ", zmq_lib)
for path in sys.path:
        if path.endswith("scripts"):
                python_path = os.path.abspath(os.path.join(path, os.path.pardir))
        elif path.endswith("site-packages"):
                lib_dir = os.path.abspath(os.path.join(path, os.path.pardir))
                python_path = os.path.abspath(os.path.join(lib_dir, os.path.pardir))
                package_path = path

print("Roboto path ", glob.glob(package_path+'/pyglui*/pyglui/OpenSans-Regular.ttf'))

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
            hiddenimports=pyglui_hidden_imports+scipy_imports+av_hidden_imports + pyndsi_hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=['pyx_compiler','matplotlib'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='pupil_player.exe',
          icon='pupil-player.ico',
          debug=False,
          strip=False,
          upx=True,
          resources=['pupil-player.ico,ICON'],
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               [('glfw3.dll', 'glfw3.dll','BINARY')],
               [('OpenSans-Regular.ttf', package_path+'/pyglui/OpenSans-Regular.ttf','DATA')],
               [('Roboto-Regular.ttf',  package_path+'/pyglui/Roboto-Regular.ttf','DATA')],
               [('fontawesome-webfont.ttf', package_path+'/pyglui/fontawesome-webfont.ttf','DATA')],
               [(zmq_lib,os.path.join(zmq_path, zmq_lib),'DATA') ],
               strip=False,
               upx=True,
               name='main')
