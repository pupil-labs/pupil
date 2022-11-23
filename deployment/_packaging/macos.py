import logging
import os
import pathlib
import subprocess

from . import ParsedVersion


def package_bundles_as_dmg(base: pathlib.Path, version: ParsedVersion):
    # for app in base.glob("*.app"):
    #     sign_app(app)
    logging.info(f"Creating dmg file for Pupil Core {version}")
    dmg_file = create_dmg(create_and_fill_dmg_srcfolder(base), base.name, version)
    sign_object(dmg_file)


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


def create_dmg(
    bundle_dir: pathlib.Path, name: str, version: ParsedVersion
) -> pathlib.Path:
    volumen_size = get_size(bundle_dir)
    dmg_name = f"{name}.dmg"
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
        dmg_name,
    )
    subprocess.check_call(dmg_cmd)
    return pathlib.Path(dmg_name)


def get_size(start_path: str | pathlib.Path = "."):
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def sign_app(path: pathlib.Path):
    for obj in path.rglob(".dylibs/*.dylib"):
        sign_object(obj)
    sign_object(path)


def sign_object(path: pathlib.Path):
    logging.info(f"Attempting to sign '{path}'")
    subprocess.check_call(
        [
            "codesign",
            "--all-architectures",
            "--force",
            "--strict=all",
            "--options",
            "runtime",
            "--entitlements",
            "entitlements.plist",
            "--continue",
            "--verify",
            "--verbose=4",
            "-s",
            os.environ["MACOS_CODESIGN_IDENTITY"],
            str(path),
        ]
    )
    logging.info(f"Successfully signed '{path}'")
