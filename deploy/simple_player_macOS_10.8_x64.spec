
# -*- mode: python -*-
from git_version import get_tag_commit

a = Analysis(['../pupil_src/simple_player/simple_circle.py'],
             # pathex=['../pupil_src/shared_modules/'],
             hiddenimports=['numpy'],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='pupil_simple_player',
          debug=False,
          strip=None,
          upx=False,
          console=False)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='Pupil Simple Player')

app = BUNDLE(coll,
             name='Pupil Simple Player.app',
             icon='macos_icon.icns',
             version = str(get_tag_commit()))
