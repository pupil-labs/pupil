# -*- mode: python -*-


import platform

if platform.system() == 'Darwin':
    from version import dpkg_deb_version

    a = Analysis(['../pupil_src/capture/main.py'],
                 pathex=['../pupil_src/shared_modules/'],
                 hiddenimports=['pyglui.pyfontstash.fontstash','pyglui.cygl.shader','pyglui.cygl.utils'],
                 hookspath=None,
                 runtime_hooks=None)
    pyz = PYZ(a.pure)
    exe = EXE(pyz,
              a.scripts,
              exclude_binaries=True,
              name='pupil_capture',
              debug=False,
              strip=None,
              upx=False,
              console=False)

    #exclude system lib.
    libSystem = [bn for bn in a.binaries if 'libSystem.dylib' in bn]
    coll = COLLECT(exe,
                   a.binaries - libSystem,
                   a.zipfiles,
                   a.datas,
                   [('methods.so', '../pupil_src/shared_modules/c_methods/methods.so','BINARY')],
                   [('uvcc.so', '../pupil_src/shared_modules/uvc_capture/mac_video/uvcc.so','BINARY')],
                   [('libglfw3.dylib', '/usr/local/Cellar/glfw3/3.0.2/lib/libglfw3.dylib','BINARY')],
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
                 hiddenimports=['pyglui.pyfontstash.fontstash','pyglui.cygl.shader','pyglui.cygl.utils'],
                 hookspath=None,
                 runtime_hooks=None)
    pyz = PYZ(a.pure)
    exe = EXE(pyz,
              a.scripts,
              exclude_binaries=True,
              name='pupil_capture',
              debug=False,
              strip=False,
              upx=True,
              console=True)

    coll = COLLECT(exe,
                   [b for b in a.binaries if not "libX" in b[0] and not "libxcb" in b[0]], # any libX file should be taken from distro else not protable between Ubuntu 12.04 and 14.04
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
	
	scipy_imports = []
	scipy_imports += ['scipy.integrate', 'scipy.integrate._ode', 'scipy.integrate.quadrature', 'scipy.integrate.odepack', 'scipy.integrate._odepack', 'scipy.integrate.quadpack', 'scipy.integrate._quadpack']
	scipy_imports += ['scipy.integrate.vode', 'scipy.integrate.lsoda', 'scipy.integrate._dop']

	a = Analysis(['../pupil_src/capture/main.py'],
	             pathex=['../pupil_src/shared_modules/'],
	             hiddenimports=['pyglui.cygl.shader', 'scipy.special._ufuncs_cxx']+scipy_imports,
	             hookspath=None,
	             runtime_hooks=None)
	#a.datas += [('GLFW_ICON','pupil-capture.ico','ICON')]
	pyz = PYZ(a.pure)
	exe = EXE(pyz,
	          a.scripts,
	          exclude_binaries=True,
	          name='pupil_capture.exe',
	          icon='pupil-capture.ico',
	          debug=False,
	          strip=None,
	          upx=True,
	          console=False)
	coll = COLLECT(exe,
	               a.binaries,
	               a.zipfiles,
	               a.datas,
	               [('methods.so', '../pupil_src/shared_modules/c_methods/methods.so','BINARY')],
	               [('_videoInput.lib', 'C:/Python27/_videoInput.lib','BINARY')],
	               [('glfw3.dll', '../pupil_src/shared_modules/external/glfw3.dll','BINARY')],
	               [('glfw3.lib', '../pupil_src/shared_modules/external/glfw3.lib','BINARY')],
	               [('glfw3dll.lib', '../pupil_src/shared_modules/external/glfw3dll.lib','BINARY')],
	               [('OpenSans-Regular.ttf','C:/Python27/Lib/site-packages/pyglui/OpenSans-Regular.ttf','DATA')],
                   [('Roboto-Regular.ttf','C:/Python27/Lib/site-packages/pyglui/Roboto-Regular.ttf','DATA')],
                   #[('GLFW_ICON', 'pupil-capture.ico', 'DATA')],
	               strip=None,
	               upx=True,
	               name='Pupil Capture')
		

