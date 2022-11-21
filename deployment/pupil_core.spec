# -*- mode: python -*-

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

sys.path.append(os.path.join("..", "pupil_src", "shared_modules"))
sys.path.append(".")

from _packaging import (
    ICON_EXT,
    LIB_EXT,
    SupportedPlatform,
    linux,
    macos,
    pupil_version,
    windows,
    write_version_file,
)

CODESIGN_IDENTITY = (
    "Developer ID Application: Pupil Labs UG (haftungsbeschrankt) (R55K9ESN6B)"
)

SPECPATH: str
DISTPATH: str


def main():
    cwd = SPECPATH
    current_platform = SupportedPlatform(platform.system())
    deployment_root = pathlib.Path(cwd)

    all_datas = []
    all_binaries = []
    all_hidden_imports = []
    for name in (
        "zmq",
        "pyre",
        "pyglui",
        "ndsi",
        "pupil_apriltags",
        "pye3d",
        "OpenGL",
    ):
        datas, binaries, hiddenimports = collect_all(
            name, exclude_datas=["**/__pycache__"]
        )
        all_datas.extend(datas)
        all_binaries.extend(binaries)
        all_hidden_imports.extend(hiddenimports)

    a = Analysis(
        ["../pupil_src/main.py"],
        pathex=["../pupil_src/shared_modules/"],
        datas=all_datas,
        binaries=all_binaries,
        hiddenimports=all_hidden_imports,
        # runtime_hooks=["pupil_core_hooks.py"],
    )
    pyz = PYZ(a.pure)

    for name in ("capture", "player", "service"):
        icon_name = "pupil-" + name + ICON_EXT[current_platform]
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
            icon=str(icon_path),
            resources=[f"{icon_path},ICON"],
            target_arch="x86_64",
            codesign_identity=CODESIGN_IDENTITY,
            entitlements_file="entitlements.plist",
        )

        extras: list[tuple[str, str, str]] = []
        # Add binaries that are not being collected automatically
        glfw: pathlib.Path = files("glfw")
        extras.extend(
            (bin_path.name, str(bin_path), "BINARY")
            for bin_path in glfw.rglob("*" + LIB_EXT[current_platform])
        )

        apriltags: pathlib.Path = files("pupil_apriltags")
        apriltags = apriltags.with_name(apriltags.name + ".libs")
        extras.extend(
            (bin_path.name, str(bin_path), "BINARY")
            for bin_path in apriltags.rglob("*" + LIB_EXT[current_platform])
        )

        if current_platform == SupportedPlatform.windows:
            extras.append(
                ("PupilDrvInst.exe", "../pupil_external/PupilDrvInst.exe", "BINARY")
            )

        binaries = a.binaries
        if current_platform == SupportedPlatform.linux:
            # libc is also not meant to travel with the bundle. Otherwise pyre.helpers with segfault.
            binaries = (b for b in binaries if not "libc.so" in b[0])
            # libstdc++ is also not meant to travel with the bundle. Otherwise nvideo opengl drivers will fail to load.
            binaries = (b for b in binaries if not "libstdc++.so" in b[0])
            # required for 14.04 16.04 interoperability.
            binaries = (b for b in binaries if not "libgomp.so.1" in b[0])
            # required for 17.10 interoperability.
            binaries = (b for b in binaries if not "libdrm.so.2" in b[0])
            binaries = list(binaries)

        app_name = f"Pupil {name.capitalize()}"
        collection = COLLECT(
            exe,
            binaries,
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

        target_path_components: dict[SupportedPlatform, tuple[str, ...]] = {
            SupportedPlatform.windows: (DISTPATH, app_name),
            SupportedPlatform.linux: (DISTPATH, app_name),
            SupportedPlatform.macos: (DISTPATH, f"{app_name}.app", "Contents", "MacOS"),
        }
        version_file_target = os.path.join(*target_path_components[current_platform])
        logging.debug(f"Writing version file to {version_file_target}")
        write_version_file(version_file_target)

    bundle_postprocessing = {
        SupportedPlatform.windows: windows.create_compressed_msi,
        SupportedPlatform.macos: macos.package_bundles_as_dmg,
        SupportedPlatform.linux: linux.create_zipped_deb_packages,
    }
    bundle_postprocessing[current_platform](pathlib.Path(DISTPATH), pupil_version())


def apriltag_relative_path(absolute_path: pathlib.Path):
    """Returns pupil_apriltags/lib/*"""
    return os.path.join(*absolute_path.parts[-3:])


main()
