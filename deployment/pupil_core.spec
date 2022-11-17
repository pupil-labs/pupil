# -*- mode: python -*-

import enum
import logging
import os
import pathlib
import platform
import sys
from importlib.resources import files

import PyInstaller
from PyInstaller.utils.hooks import collect_all
from rich.logging import RichHandler
from rich.traceback import install

install(suppress=[PyInstaller])
logging.getLogger().handlers = [RichHandler()]

sys.path.append(os.path.join("../", "pupil_src", "shared_modules"))
from version_utils import get_tag_commit, pupil_version, write_version_file


def main():
    cwd = SPECPATH
    current_platform = SupportedPlatform(platform.system())
    deployment_root = pathlib.Path(cwd)

    all_datas = []
    all_binaries = []
    all_hidden_imports = []
    for name in ("zmq", "pyre", "pyglui", "ndsi", "pupil_apriltags", "pye3d"):
        # dll_dir = None
        # if current_platform == SupportedPlatform.windows:
        #     site_path: pathlib.Path = files(name)
        #     libs_path = site_path.with_name(site_path.name + ".libs")
        #     if libs_path.is_dir():
        #         dll_dir = os.add_dll_directory(str(libs_path))

        datas, binaries, hiddenimports = collect_all(
            name, exclude_datas=["**/__pycache__"]
        )
        logging.info(f"Collected {name} {datas=}")
        logging.info(f"Collected {name} {binaries=}")
        logging.info(f"Collected {name} {hiddenimports=}")
        all_datas.extend(datas)
        all_binaries.extend(binaries)
        all_hidden_imports.extend(hiddenimports)
        # if dll_dir:
        #     dll_dir.close()

    a = Analysis(
        ["../pupil_src/main.py"],
        pathex=["../pupil_src/shared_modules/"],
        datas=all_datas,
        binaries=all_binaries,
        hiddenimports=all_hidden_imports,
        runtime_hooks=["pupil_core_hooks.py"],
    )
    pyz = PYZ(a.pure)

    for name in ("capture", "player", "service"):
        icon_name = "pupil-" + name + icon_ext[current_platform]
        icon_path = (deployment_root / "icons" / icon_name).resolve()

        exe = EXE(
            pyz,
            a.scripts,
            exclude_binaries=True,
            name=f"pupil_{name}",
            debug=False,
            strip=False,
            upx=True,
            console=True,
            target_arch="x86",
            icon=str(icon_path),
            resources=[f"{icon_path},ICON"],
        )

        extras = []
        if current_platform == SupportedPlatform.windows:
            # Add binaries that are not being collected automatically
            glfw_bin = files("glfw") / "glfw3.dll"
            extras.append((glfw_bin.name, str(glfw_bin), "BINARY"))
            apriltags: pathlib.Path = files("pupil_apriltags")
            apriltags = apriltags.with_name(apriltags.name + ".libs")
            for dll in apriltags.glob("*.dll"):
                extras.append((dll.name, str(dll), "BINARY"))

            extras.append(
                ("PupilDrvInst.exe", "../pupil_external/PupilDrvInst.exe", "BINARY")
            )

        app_name = f"Pupil {name.capitalize()}"
        collection = COLLECT(
            exe,
            a.binaries,
            a.zipfiles,
            a.datas,
            extras,
            strip=False,
            upx=True,
            name=app_name,
        )

        BUNDLE(
            collection,
            name=f"{app_name}.app",
            icon=icon_path,
            version=str(pupil_version()),
            info_plist={"NSHighResolutionCapable": "True"},
        )

        if current_platform == SupportedPlatform.windows:
            write_version_file(os.path.join(DISTPATH, app_name))


class SupportedPlatform(enum.Enum):
    macos = "Darwin"
    linux = "Linux"
    windows = "Windows"


icon_ext = {
    SupportedPlatform.macos: ".icns",
    SupportedPlatform.linux: ".svg",
    SupportedPlatform.windows: ".ico",
}
lib_ext = {
    SupportedPlatform.macos: ".dylib",
    SupportedPlatform.linux: ".so",
    SupportedPlatform.windows: ".dll",
}


def apriltag_relative_path(absolute_path):
    """Returns pupil_apriltags/lib/*"""
    return os.path.join(*absolute_path.parts[-3:])


main()
