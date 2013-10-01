# -*- mode: python -*-

a = Analysis(['pupil_src/capture/main.py'],
             pathex=['pupil_src/shared_modules/'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='pupil_capture',
          debug=False,
          strip=None,
          upx=True,
          console=True )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               [('methods.so', 'pupil_src/shared_modules/c_methods/methods.so','BINARY')],
               [('capture.so', 'pupil_src/shared_modules/uvc_capture/linux_video/v4l2_capture/capture.so','BINARY')],
               [('libAntTweakBar.so', '/usr/lib/libAntTweakBar.so','BINARY')],
               [('libAntTweakBar.so.1', '/usr/lib/libAntTweakBar.so.1','BINARY')],
               [('libglfw.so.3.0', '/usr/local/lib/libglfw.so.3.0','BINARY')],
               [('libglfw.so.3', '/usr/local/lib/libglfw.so.3','BINARY')],
               [('libglfw.so', '/usr/local/lib/libglfw.so','BINARY')],
               strip=None,
               upx=True,
               name='pupil_capture')
