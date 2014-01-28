
# -*- mode: python -*-
from git_version import get_tag_commit

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
          upx=False,
          console=False)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               [('methods.so', '../pupil_src/shared_modules/c_methods/methods.so','BINARY')],
               [('uvcc.so', '../pupil_src/shared_modules/uvc_capture/mac_video/uvcc.so','BINARY')],
               [('libAntTweakBar.dylib', '/usr/local/Cellar/anttweakbar/1.16/lib/libAntTweakBar.dylib','BINARY')],
               [('libglfw3.dylib', '/usr/local/Cellar/glfw3/3.0.2/lib/libglfw3.dylib','BINARY')],
               strip=None,
               upx=True,
               name='Pupil Player')

app = BUNDLE(coll,
             name='Pupil Player.app',
             icon='macos_icon.icns',
             version = str(get_tag_commit()))
