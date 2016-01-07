# -*- mode: python -*-


import platform

av_hidden_imports = ['av.format','av.packet','av.frame','av.stream','av.plane','av.audio.plane','av.audio.stream','av.subtitles','av.subtitles.stream','av.subtitles.subtitle','av.video.reformatter','av.video.plane']
pyglui_hidden_imports = ['pyglui.pyfontstash.fontstash','pyglui.cygl.shader','pyglui.cygl.utils']

if platform.system() == 'Darwin':
    from version import dpkg_deb_version

    a = Analysis(['../pupil_src/capture/main.py'],
                 pathex=['../pupil_src/shared_modules/'],
                 hiddenimports=[]+av_hidden_imports+pyglui_hidden_imports,
                 hookspath=None,
                 runtime_hooks=None,
                 excludes=['pyx_compiler','matplotlib'])
    pyz = PYZ(a.pure)
    exe = EXE(pyz,
              a.scripts,
              exclude_binaries=True,
              name='pupil_capture',
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
                   [('libglfw3.dylib', '/usr/local/Cellar/glfw3/3.1.2/lib/libglfw3.dylib','BINARY')],
                   [('methods.so', '../pupil_src/shared_modules/c_methods/methods.so','BINARY')],
                   [('OpenSans-Regular.ttf','/usr/local/lib/python2.7/site-packages/pyglui/OpenSans-Regular.ttf','DATA')],
                   [('Roboto-Regular.ttf','/usr/local/lib/python2.7/site-packages/pyglui/Roboto-Regular.ttf','DATA')],
                   strip=None,
                   upx=True,
                   name='Pupil Capture')

    app = BUNDLE(coll,
                 name='Pupil Capture.app',
                 icon='pupil-capture.icns',
                 version = str(dpkg_deb_version()))


elif platform.system() == 'Linux':
    a = Analysis(['../pupil_src/capture/main.py'],
                 pathex=['../pupil_src/shared_modules/'],
                 hiddenimports=[]+av_hidden_imports+pyglui_hidden_imports,
                 hookspath=None,
                 runtime_hooks=None,
                 excludes=['pyx_compiler','matplotlib'])

    pyz = PYZ(a.pure)
    exe = EXE(pyz,
              a.scripts,
              exclude_binaries=True,
              name='pupil_capture',
              debug=False,
              strip=False,
              upx=True,
              console=True)


    # any libX file should be taken from distro else not protable between Ubuntu 12.04 and 14.04
    binaries = [b for b in a.binaries if not "libX" in b[0] and not "libxcb" in b[0]]
    # libc is also not meant to travel with the bundle. Otherwise pyre.helpers with segfault.
    binaries = [b for b in binaries if not "libc.so" in b[0]]

    # libstdc++ is also not meant to travel with the bundle. Otherwise nvideo opengl drivers will fail to load.
    binaries = [b for b in binaries if not "libstdc++.so" in b[0]]
    coll = COLLECT(exe,
                   binaries,
                   a.zipfiles,
                   a.datas,
                   [('methods.so', '../pupil_src/shared_modules/c_methods/methods.so','BINARY')],
                   [('libglfw.so', '/usr/local/lib/libglfw.so','BINARY')],
                   [('libGLEW.so', '/usr/lib/x86_64-linux-gnu/libGLEW.so','BINARY')],
                   [('OpenSans-Regular.ttf','/usr/local/lib/python2.7/dist-packages/pyglui/OpenSans-Regular.ttf','DATA')],
                   [('Roboto-Regular.ttf','/usr/local/lib/python2.7/dist-packages/pyglui/Roboto-Regular.ttf','DATA')],
                   strip=True,
                   upx=True,
                   name='pupil_capture')

elif platform.system() == 'Windows':
	import sys, os, os.path

	system_path = os.path.join(os.environ['windir'], 'system32')

	print "Using Environment:"
	python_path = None
	package_path = None
	for path in sys.path:
		print " -- " + path
		if path.endswith("scripts"):
			python_path = os.path.abspath(os.path.join(path, os.path.pardir))
		elif path.endswith("site-packages"):
			lib_dir = os.path.abspath(os.path.join(path, os.path.pardir))
			python_path = os.path.abspath(os.path.join(lib_dir, os.path.pardir))
			package_path = path
	if (python_path and package_path):	
		print "PYTHON PATH @ " + python_path
		print "PACKAGE PATH @ " + package_path
	else:
		print "could not find python_path or package_path. EXIT."
		quit()
	scipy_imports = ['scipy.integrate']
	scipy_imports += ['scipy.integrate._ode', 'scipy.integrate.quadrature', 'scipy.integrate.odepack', 'scipy.integrate._odepack', 'scipy.integrate.quadpack', 'scipy.integrate._quadpack']
	scipy_imports += ['scipy.integrate.vode', 'scipy.integrate.lsoda', 'scipy.integrate._dop', 'scipy.special._ufuncs', 'scipy.special._ufuncs_cxx']

	a = Analysis(['../pupil_src/capture/main.py'],
	             pathex=['../pupil_src/shared_modules/'],
	             hiddenimports=['pyglui.cygl.shader']+scipy_imports+av_hidden_imports,
	             hookspath=None,
	             runtime_hooks=None,
               excludes=['pyx_compiler','matplotlib'])


	pyz = PYZ(a.pure)
	exe = EXE(pyz,
	          a.scripts,
	          exclude_binaries=True,
	          name='pupil_capture.exe',
	          icon='pupil-capture.ico',
	          debug=False,
	          strip=None,
	          upx=True,
	          console=False,
	          resources=['pupil-capture.ico,ICON,GLFW_ICON'])
	coll = COLLECT(exe,
	               a.binaries,
	               a.zipfiles,
	               a.datas,
	               [('methods.so', '../pupil_src/shared_modules/c_methods/methods.so','BINARY')],
	               [('glfw3.dll', '../pupil_src/shared_modules/external/glfw3.dll','BINARY')],
	               [('glfw3.lib', '../pupil_src/shared_modules/external/glfw3.lib','BINARY')],
	               [('glfw3dll.lib', '../pupil_src/shared_modules/external/glfw3dll.lib','BINARY')],
	               [('opencv_ffmpeg2411.dll', os.path.join(python_path, 'opencv_ffmpeg2411.dll'),'BINARY')],
	               [('_videoInput.lib', os.path.join(python_path, '_videoInput.lib'),'BINARY')],
	               [('msvcp110.dll', os.path.join(system_path, 'msvcp110.dll'),'BINARY')],
	               [('msvcr110.dll', os.path.join(system_path, 'msvcr110.dll'),'BINARY')],
	               [('msvcr120.dll', os.path.join(system_path, 'msvcr120.dll'),'BINARY')],
	               [('OpenSans-Regular.ttf', os.path.join(package_path, 'pyglui/OpenSans-Regular.ttf'),'DATA')],
                   [('Roboto-Regular.ttf', os.path.join(package_path, 'pyglui/Roboto-Regular.ttf'),'DATA')],
	               strip=None,
	               upx=True,
	               name='Pupil Capture')


