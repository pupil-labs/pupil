# -*- mode: python -*-

a = Analysis(['../pupil_src/player/main.py'],
             pathex=['../pupil_src/shared_modules/'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='pupil_player',
          debug=False,
          strip=None,
          upx=True,
          console=True)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               [('methods.so', '../pupil_src/shared_modules/c_methods/methods.so','BINARY')],
               [('capture.so', '../pupil_src/shared_modules/uvc_capture/linux_video/v4l2_capture/capture.so','BINARY')],
               [('libAntTweakBar.so', '/usr/lib/libAntTweakBar.so','BINARY')],
               [('libglfw.so', '/usr/local/lib/libglfw.so','BINARY')],
               [('v4l2-ctl', '/usr/bin/v4l2-ctl','BINARY')],
               [('icon.ico', 'linux_icon.ico','DATA')],
               strip=None,
               upx=True,
               name='pupil_player')
