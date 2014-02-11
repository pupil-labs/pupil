# -*- mode: python -*-

a = Analysis(['../pupil_src/capture/build_test.py'],
             pathex=[],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          [('v', None, 'OPTION')],
          exclude_binaries=True,
          name='build_test',
          debug=True,
          strip=None,
          upx=False,
          console=False )

coll = COLLECT(exe,
               a.binaries+[('cv2.so','/usr/local/lib/python2.7/site-packages/cv2.so','BINARY')],
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='build_test')
