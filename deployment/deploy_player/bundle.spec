# -*- mode: python -*-


import platform, sys, os, os.path, zmq, glob, ntpath, numpy

av_hidden_imports = ['av.format','av.packet','av.buffer','av.bytesource','av.frame','av.stream','av.descriptor','av.plane','av.audio.plane','av.container.streams','av.dictionary', 'av.audio.stream','av.subtitles','av.subtitles.stream','av.subtitles.subtitle','av.video.reformatter','av.video.plane','av.option']
pyglui_hidden_imports = ['pyglui.pyfontstash.fontstash','pyglui.cygl.shader','pyglui.cygl.utils']
pyndsi_hidden_imports = ['pyre']

from pyglui import ui

if platform.system() == 'Darwin':
    sys.path.append('.')
    from version import dpkg_deb_version
    del sys.path[-1]

    a = Analysis(['../../pupil_src/main.py'],
                 pathex=['../../pupil_src/shared_modules/'],
                 hiddenimports=[]+av_hidden_imports+pyglui_hidden_imports+pyndsi_hidden_imports,
                 hookspath=None,
                 runtime_hooks=['../rthook_multiprocessing.py'],
                 excludes=['matplotlib'])

    pyz = PYZ(a.pure)
    exe = EXE(pyz,
              a.scripts,
              exclude_binaries=True,
              name='pupil_player',
              debug=False,
              strip=None,
              upx=False,
              console=True)

    #exclude system lib.
    libSystem = [bn for bn in a.binaries if 'libSystem.dylib' in bn]
    coll = COLLECT(exe,
                   a.binaries - libSystem,
                   a.zipfiles,
                   a.datas,
                   [('libglfw.dylib', '/usr/local/lib/libglfw.dylib','BINARY')],
                   [('OpenSans-Regular.ttf',ui.get_opensans_font_path(),'DATA')],
                   [('Roboto-Regular.ttf',ui.get_roboto_font_path(),'DATA')],
                   [('fontawesome-webfont.ttf',ui.get_fontawesome_font_path(),'DATA')],
                   strip=None,
                   upx=True,
                   name='Pupil Player')

    app = BUNDLE(coll,
                 name='Pupil Player.app',
                 icon='pupil-player.icns',
                 version = str(dpkg_deb_version()),
                 info_plist={
                          'NSHighResolutionCapable': 'True'
                            },
                )


elif platform.system() == 'Linux':
    a = Analysis(['../../pupil_src/main.py'],
                 pathex=['../../pupil_src/shared_modules/'],
                 hiddenimports=[]+av_hidden_imports+pyglui_hidden_imports+pyndsi_hidden_imports,
                 hookspath=None,
                 runtime_hooks=None,
                 excludes=['matplotlib'])

    pyz = PYZ(a.pure)
    exe = EXE(pyz,
              a.scripts,
              exclude_binaries=True,
              name='pupil_player',
              debug=False,
              strip=None,
              upx=True,
              console=True)


    # libc is also not meant to travel with the bundle. Otherwise pyre.helpers with segfault.
    binaries = [b for b in a.binaries if not "libc.so" in b[0]]

    # libstdc++ is also not meant to travel with the bundle. Otherwise nvideo opengl drivers will fail to load.
    binaries = [b for b in binaries if not "libstdc++.so" in b[0]]

    # required for 14.04 16.04 interoperability.
    binaries = [b for b in binaries if not "libgomp.so.1" in b[0]]

    coll = COLLECT(exe,
                   binaries,
                   a.zipfiles,
                   a.datas,
                   [('libglfw.so', '/usr/local/lib/libglfw.so','BINARY')],
                   [('libGLEW.so', '/usr/lib/x86_64-linux-gnu/libGLEW.so','BINARY')],
                   [('OpenSans-Regular.ttf',ui.get_opensans_font_path(),'DATA')],
                   [('Roboto-Regular.ttf',ui.get_roboto_font_path(),'DATA')],
                   [('fontawesome-webfont.ttf',ui.get_fontawesome_font_path(),'DATA')],
                   strip=True,
                   upx=True,
                   name='pupil_player')

elif platform.system() == 'Windows':
        import sys, os, os.path

        np_path = os.path.dirname(numpy.__file__)
        np_dlls = glob.glob(np_path + '/core/*.dll')
        np_dll_list = []

        for dll_path in np_dlls:
            dll_p, dll_f = ntpath.split(dll_path)
            np_dll_list += [(dll_f, dll_path, 'BINARY')]

        zmq_path = os.path.dirname(zmq.__file__)

        zmq_p, zmq_lib = ntpath.split(glob.glob(zmq_path +  '/libzmq.*.pyd')[0])

        system_path = os.path.join(os.environ['windir'], 'system32')

        print( "Using Environment:")
        python_path = None
        package_path = None
        for path in sys.path:
                print( " -- " + path)
                if path.endswith("scripts"):
                        python_path = os.path.abspath(os.path.join(path, os.path.pardir))
                elif path.endswith("site-packages"):
                        lib_dir = os.path.abspath(os.path.join(path, os.path.pardir))
                        python_path = os.path.abspath(os.path.join(lib_dir, os.path.pardir))
                        package_path = path
        if (python_path and package_path):
                print( "PYTHON PATH @ " + python_path)
                print( "PACKAGE PATH @ " + package_path)
        else:
                print( "could not find python_path or package_path. EXIT.")
                quit()
        scipy_imports = ['scipy.integrate']
        scipy_imports += ['scipy.integrate._ode', 'scipy.integrate.quadrature', 'scipy.integrate.odepack', 'scipy.integrate._odepack', 'scipy.integrate.quadpack', 'scipy.integrate._quadpack']
        scipy_imports += ['scipy.integrate.vode', 'scipy.integrate.lsoda', 'scipy.integrate._dop', 'scipy.special._ufuncs', 'scipy.special._ufuncs_cxx']

        a = Analysis(['../../pupil_src/main.py'],
                     pathex=['../../pupil_src/shared_modules/', '../../pupil_external'],
                     hiddenimports=['pyglui.cygl.shader']+scipy_imports+av_hidden_imports+pyndsi_hidden_imports,
                     hookspath=None,
                     runtime_hooks=None,
                 excludes=['matplotlib'])


        pyz = PYZ(a.pure)
        exe = EXE(pyz,
                  a.scripts,
                  exclude_binaries=True,
                  name='pupil_player.exe',
                  icon='pupil-player.ico',
                  debug=False,
                  strip=None,
                  upx=True,
                  console=True,
                  resources=['pupil-player.ico,ICON'])

        coll = COLLECT(exe,
                       a.binaries,
                       a.zipfiles,
                       a.datas,
                       [('glfw3.dll', '../../pupil_external/glfw3.dll','BINARY')],
                       [('OpenSans-Regular.ttf', os.path.join(package_path, 'pyglui/OpenSans-Regular.ttf'),'DATA')],
                       [('Roboto-Regular.ttf', os.path.join(package_path, 'pyglui/Roboto-Regular.ttf'),'DATA')],
                       [('fontawesome-webfont.ttf', os.path.join(package_path, 'pyglui/fontawesome-webfont.ttf'),'DATA')],
                       [(zmq_lib,os.path.join(zmq_path, zmq_lib),'DATA') ],
                       np_dll_list,
                       strip=None,
                       upx=True,
                       name='Pupil Player')
