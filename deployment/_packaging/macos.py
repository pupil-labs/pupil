import json
import logging
import os
import pathlib
import subprocess

from . import ParsedVersion


def package_bundles_as_dmg(base: pathlib.Path, version: ParsedVersion):
    should_sign_and_notarize = (
        os.environ.get("MACOS_SHOULD_SIGN_AND_NOTARIZE", "false").strip() == "true"
    )
    if should_sign_and_notarize:
        for app in base.glob("*.app"):
            sign_app(app)
        logging.info(f"Creating dmg file for Pupil Core {version}")
        dmg_file = create_dmg(create_and_fill_dmg_srcfolder(base), base.name, version)
        sign_object(dmg_file)
        notarize_bundle(dmg_file)
    else:
        logging.info("Skipping signing, notarization, and creation of dmg file")
        for app in base.glob("*.app"):
            app.rename(app.name)


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
            "--timestamp",
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


def notarize_bundle(path: pathlib.Path):
    logging.info(f"Attempting to notarize '{path}'")
    auth_args = [
        "--apple-id",
        os.environ["MACOS_NOTARYTOOL_APPLE_ID"],
        "--team-id",
        os.environ["MACOS_NOTARYTOOL_TEAM_ID"],
        "--password",
        os.environ["MACOS_NOTARYTOOL_APPSPECIFIC_PASSWORD"],
    ]
    format_args = ["-f", "json"]
    submit_result = subprocess.check_output(
        ["xcrun", "notarytool", "submit", str(path), *auth_args, *format_args]
    )
    submit_result = json.loads(submit_result)
    logging.info(f"{submit_result['message']} (ID: {submit_result['id']})")

    try:
        wait_result = subprocess.check_output(
            [
                "xcrun",
                "notarytool",
                "wait",
                submit_result["id"],
                "--timeout",
                "1h",
                *auth_args,
                *format_args,
            ]
        )
        wait_result = json.loads(wait_result)
        logging.info(f"{wait_result['message']}. Status: {wait_result['status']}")
        if wait_result["status"] == "Accepted":
            staple_bundle_notarization(path)
            logging.info(f"Successfully notarized '{path}'")

    except subprocess.CalledProcessError:
        logging.exception("Issue during processing notarization:")

    log_result = subprocess.check_output(
        [
            "xcrun",
            "notarytool",
            "log",
            submit_result["id"],
            str(path.with_suffix(".json")),
            *auth_args,
            *format_args,
        ]
    )
    log_result = json.loads(log_result)
    logging.info(f"Notarization logs saved to {log_result['location']}")


def staple_bundle_notarization(path: pathlib.Path):
    subprocess.check_call(["xcrun", "stapler", "staple", "-v", str(path)])
