import logging
import os
import pathlib
import subprocess

from . import ParsedVersion


def package_bundles_as_dmg(base: pathlib.Path, version: ParsedVersion):
    logging.info(f"Creating dmg file for Pupil Core {version}")
    create_dmg(create_and_fill_dmg_srcfolder(base), base.name, version)


def create_and_fill_dmg_srcfolder(
    base: pathlib.Path, name: str = "bundles"
) -> pathlib.Path:
    bundle_dir = base / name
    bundle_dir.mkdir(exist_ok=True)
    for app in base.glob("*.app"):
        app.rename(bundle_dir / app.name)

    applications_target = pathlib.Path("/Applications")
    applications_symlink = bundle_dir / "Applications"
    if applications_symlink.exists():
        applications_symlink.unlink()
    applications_symlink.symlink_to(applications_target, target_is_directory=True)
    return bundle_dir


def create_dmg(bundle_dir: pathlib.Path, name: str, version: ParsedVersion):
    volumen_size = get_size(bundle_dir)
    dmg_cmd = (
        "hdiutil",
        "create",
        "-volname",
        f"Install Pupil Core {version}",
        "-srcfolder",
        bundle_dir,
        "-format",
        "ULMO",
        "-size",
        f"{volumen_size}b ",
        f"{name}.dmg",
    )
    subprocess.call(dmg_cmd)


def get_size(start_path: str | pathlib.Path = "."):
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size
