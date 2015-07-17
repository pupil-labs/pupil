# -*- mode: python -*-


import platform

av_hidden_imports = ['av.format','av.packet','av.frame','av.stream','av.plane','av.audio.plane','av.audio.stream','av.subtitles','av.subtitles.stream','av.subtitles.subtitle','av.video.reformatter','av.video.plane']


if platform.system() == 'Darwin':
    from version import dpkg_deb_version

    a = Analysis(['../pupil_src/player/main.py'],
                 pathex=['../pupil_src/shared_modules/'],
                 hiddenimports=['pyglui.pyfontstash.fontstash','pyglui.cygl.shader','pyglui.cygl.utils']+av_hidden_imports,
                 hookspath=None,
                 runtime_hooks=None,
                 excludes=['pyx_compiler','matplotlib'])

    pyz = PYZ(a.pure)
    exe = EXE(pyz,
              a.scripts,
              exclude_binaries=True,
              name='pupil_player',
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
                   [('libglfw3.dylib', '/usr/local/Cellar/glfw3/3.1.1/lib/libglfw3.dylib','BINARY')],                   [('methods.so', '../pupil_src/shared_modules/c_methods/methods.so','BINARY')],
                   [('OpenSans-Regular.ttf','/usr/local/lib/python2.7/site-packages/pyglui/OpenSans-Regular.ttf','DATA')],
                   [('Roboto-Regular.ttf','/usr/local/lib/python2.7/site-packages/pyglui/Roboto-Regular.ttf','DATA')],
                   strip=None,
                   upx=True,
                   name='Pupil Player')

    app = BUNDLE(coll,
                 name='Pupil Player.app',
                 icon='pupil-player.icns',
                 version = str(dpkg_deb_version()))

elif platform.system() == 'Linux':
    a = Analysis(['../pupil_src/player/main.py'],
                 pathex=['../pupil_src/shared_modules/'],
                 hiddenimports=['pyglui.pyfontstash.fontstash','pyglui.cygl.shader','pyglui.cygl.utils']+av_hidden_imports,
                 hookspath=None,
                 runtime_hooks=None,
                 excludes=['pyx_compiler','matplotlib'])

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
                   [b for b in a.binaries if not "libX" in b[0] and not "libxcb" in b[0]], # any libX file should be taken from distro else not protable between Ubuntu 12.04 and 14.04
                   a.zipfiles,
                   a.datas,
                   [('methods.so', '../pupil_src/shared_modules/c_methods/methods.so','BINARY')],
                   [('libglfw.so', '/usr/local/lib/libglfw.so','BINARY')],
                   [('OpenSans-Regular.ttf','/usr/local/lib/python2.7/dist-packages/pyglui/OpenSans-Regular.ttf','DATA')],
                   [('Roboto-Regular.ttf','/usr/local/lib/python2.7/dist-packages/pyglui/Roboto-Regular.ttf','DATA')],
                   strip=True,
                   upx=True,
                   name='pupil_player')
