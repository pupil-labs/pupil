# -*- mode: python -*-

a = Analysis(['../pupil_src/capture/build_test.py'],
             pathex=['../pupil_src/shared_modules/'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='pupil_capture',
          debug=True,
          strip=None,
          upx=False,
          console=True )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               [('methods.so', '../pupil_src/shared_modules/c_methods/methods.so','BINARY')],
               [('libAntTweakBar.dylib', '/usr/local/Cellar/anttweakbar/1.16/lib/libAntTweakBar.dylib','BINARY')],
               [('libglfw3.dylib', '/usr/local/Cellar/glfw3/3.0.2/lib/libglfw3.dylib','BINARY')],
               strip=None,
               upx=True,
               name='pupil_capture')

app = BUNDLE(coll,
             name='pupil_capture.app',
             icon='macos_icon.icns')
