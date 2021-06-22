# -*- mode: python -*-


import glob
import os
import os.path
import pathlib
import platform
import sys

import numpy
import pkg_resources
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hidden_imports = []
hidden_imports += collect_submodules("av")

if platform.system() != "Windows":
    hidden_imports.append("cysignals")

hidden_imports += collect_submodules("pyglui")
hidden_imports += collect_submodules("sklearn")

import glfw
from pyglui import ui

glfw_name = glfw._glfw._name
glfw_path = pathlib.Path(glfw_name)
if not glfw_path.exists():
    glfw_path = pathlib.Path(pkg_resources.resource_filename("glfw", glfw_name))
glfw_binaries = [(glfw_path.name, str(glfw_path), "BINARY")]

data_files_pye3d = collect_data_files("pye3d")

if platform.system() == "Darwin":
    sys.path.append(".")
    from version import pupil_version

    del sys.path[-1]

    a = Analysis(
        ["../../pupil_src/main.py"],
        pathex=["../../pupil_src/shared_modules/"],
        hiddenimports=hidden_imports,
        hookspath=None,
        runtime_hooks=["../find_opengl_bigsur.py"],
        excludes=["matplotlib"],
        datas=data_files_pye3d,
    )
    pyz = PYZ(a.pure)
    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        name="pupil_service",
        debug=False,
        strip=None,
        upx=False,
        console=True,
    )

    # exclude system lib.
    libSystem = [bn for bn in a.binaries if "libSystem.dylib" in bn]
    coll = COLLECT(
        exe,
        a.binaries - libSystem,
        a.zipfiles,
        a.datas,
        [("pyglui/OpenSans-Regular.ttf", ui.get_opensans_font_path(), "DATA")],
        [("pyglui/Roboto-Regular.ttf", ui.get_roboto_font_path(), "DATA")],
        [("pyglui/pupil_icons.ttf", ui.get_pupil_icons_font_path(), "DATA")],
        glfw_binaries,
        strip=None,
        upx=True,
        name="Pupil Service",
    )

    app = BUNDLE(
        coll,
        name="Pupil Service.app",
        icon="pupil-service.icns",
        version=str(pupil_version()),
        info_plist={"NSHighResolutionCapable": "True"},
    )


elif platform.system() == "Linux":
    a = Analysis(
        ["../../pupil_src/main.py"],
        pathex=["../../pupil_src/shared_modules/"],
        hiddenimports=hidden_imports,
        hookspath=None,
        runtime_hooks=None,
        excludes=["matplotlib"],
        datas=data_files_pye3d,
    )

    pyz = PYZ(a.pure)
    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        name="pupil_service",
        debug=False,
        strip=False,
        upx=True,
        console=True,
    )

    # libc is also not meant to travel with the bundle. Otherwise pyre.helpers with segfault.
    binaries = [b for b in a.binaries if not "libc.so" in b[0]]

    # libstdc++ is also not meant to travel with the bundle. Otherwise nvideo opengl drivers will fail to load.
    binaries = [b for b in binaries if not "libstdc++.so" in b[0]]

    # required for 14.04 16.04 interoperability.
    binaries = [b for b in binaries if not "libgomp.so.1" in b[0]]

    # required for 17.10 interoperability.
    binaries = [b for b in binaries if not "libdrm.so.2" in b[0]]

    coll = COLLECT(
        exe,
        binaries,
        a.zipfiles,
        a.datas,
        [("libGLEW.so", "/usr/lib/x86_64-linux-gnu/libGLEW.so", "BINARY")],
        [("pyglui/OpenSans-Regular.ttf", ui.get_opensans_font_path(), "DATA")],
        [("pyglui/Roboto-Regular.ttf", ui.get_roboto_font_path(), "DATA")],
        [("pyglui/pupil_icons.ttf", ui.get_pupil_icons_font_path(), "DATA")],
        glfw_binaries,
        strip=True,
        upx=True,
        name="pupil_service",
    )

elif platform.system() == "Windows":
    import os
    import os.path
    import sys

    np_path = os.path.dirname(numpy.__file__)
    np_dlls = glob.glob(np_path + "/core/*.dll")
    np_dll_list = []

    for dll_path in np_dlls:
        dll_p, dll_f = os.path.split(dll_path)
        np_dll_list += [(dll_f, dll_path, "BINARY")]

    hidden_imports += collect_submodules("scipy")

    external_libs_path = pathlib.Path("../../pupil_external")

    a = Analysis(
        ["../../pupil_src/main.py"],
        pathex=["../../pupil_src/shared_modules/", str(external_libs_path)],
        binaries=None,
        datas=data_files_pye3d,
        hiddenimports=hidden_imports,
        hookspath=None,
        runtime_hooks=None,
        win_no_prefer_redirects=False,
        win_private_assemblies=False,
        excludes=["matplotlib"],
    )

    pyz = PYZ(a.pure)
    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        name="pupil_service.exe",
        icon="pupil-service.ico",
        debug=False,
        strip=None,
        upx=True,
        console=False,
        resources=["pupil-service.ico,ICON"],
    )

    vc_redist_path = external_libs_path / "vc_redist"
    vc_redist_libs = [
        (lib.name, str(lib), "BINARY") for lib in vc_redist_path.glob("*.dll")
    ]

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        [("PupilDrvInst.exe", "../../pupil_external/PupilDrvInst.exe", "BINARY")],
        [("pyglui/OpenSans-Regular.ttf", ui.get_opensans_font_path(), "DATA")],
        [("pyglui/Roboto-Regular.ttf", ui.get_roboto_font_path(), "DATA")],
        [("pyglui/pupil_icons.ttf", ui.get_pupil_icons_font_path(), "DATA")],
        glfw_binaries,
        np_dll_list,
        vc_redist_libs,
        strip=False,
        upx=True,
        name="Pupil Service",
    )
