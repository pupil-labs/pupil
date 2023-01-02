import pathlib
import shutil
import subprocess

from . import ParsedVersion


def create_zipped_deb_packages(dist_root: pathlib.Path, app_version: ParsedVersion):
    deb_folder = dist_root / "debs"
    deb_folder.mkdir(exist_ok=True)
    for folder in dist_root.glob("Pupil */"):
        deb_pkg = create_deb_package(dist_root, folder.name, app_version)
        deb_pkg.rename(deb_folder / deb_pkg.name)

    shutil.make_archive(str(dist_root), "zip", deb_folder)


def create_deb_package(
    dist_root: pathlib.Path, app_name: str, app_version: ParsedVersion
) -> pathlib.Path:
    # lets build the structure for our deb package_name.

    package_name = app_name.lower().replace(" ", "_")
    deb_folder = f"{package_name}_v{app_version}"
    deb_root = (dist_root / deb_folder).resolve()
    if deb_root.exists():
        shutil.rmtree(str(deb_root))

    control = deb_root / "DEBIAN" / "control"
    desktop = deb_root / "usr" / "share" / "applications" / f"{package_name}.desktop"
    starter = deb_root / "usr" / "local" / "bin" / package_name
    opt_dir = deb_root / "opt"
    ico_dir = deb_root / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps"

    control.parent.mkdir(mode=0o755, exist_ok=True, parents=True)
    starter.parent.mkdir(mode=0o755, exist_ok=True, parents=True)
    desktop.parent.mkdir(mode=0o755, exist_ok=True, parents=True)
    ico_dir.mkdir(mode=0o755, exist_ok=True, parents=True)

    startup_WM_class = app_name
    if startup_WM_class == "Pupil Capture":
        startup_WM_class += " - World"

    # DEB control file
    with control.open("w") as f:
        dist_size = sum(f.stat().st_size for f in dist_root.rglob("*"))
        content = f"""\
Package: {package_name.replace("_", "-")}
Version: {app_version}
Architecture: amd64
Maintainer: Pupil Labs <info@pupil-labs.com>
Priority: optional
Description: {app_name} - Find more information on https://docs.pupil-labs.com/core/
Installed-Size: {round(dist_size / 1024)}
"""
        # See this link regarding the calculation of the Installed-Size field
        # https://www.debian.org/doc/debian-policy/ch-controlfields.html#installed-size
        f.write(content)
    control.chmod(0o644)

    # bin_starter script

    with starter.open("w") as f:
        content = f'''\
#!/bin/sh
exec /opt/{package_name}/{package_name} "$@"'''
        f.write(content)
    starter.chmod(0o755)

    # .desktop entry
    # ATTENTION: In order for the bundle icon to display correctly
    # two things are necessary:
    # 1. Icon needs to be the icon's base name/stem
    # 2. The window title must be equivalent to StartupWMClass
    with desktop.open("w") as f:
        content = f"""\
[Desktop Entry]
Version={app_version}
Type=Application
Name={app_name}
Comment=Preview Pupil Invisible data streams
Exec=/opt/{package_name}/{package_name}
Terminal=false
Icon={package_name.replace('_', '-')}
Categories=Application;
Name[en_US]={app_name}
Actions=Terminal;
StartupWMClass={startup_WM_class}
[Desktop Action Terminal]
Name=Open in Terminal
Exec=x-terminal-emulator -e {package_name}"""
        f.write(content)
    desktop.chmod(0o644)

    svg_file_name = f"{package_name.replace('_', '-')}.svg"
    src_path = pathlib.Path("icons", svg_file_name)
    dst_path = ico_dir / svg_file_name
    shutil.copy(str(src_path), str(dst_path))
    dst_path.chmod(0o755)

    # copy the actual application
    shutil.copytree(str(dist_root / app_name), str(opt_dir / package_name))
    for f in opt_dir.rglob("*"):
        if f.is_file():
            if f.name == package_name:
                f.chmod(0o755)
            else:
                f.chmod(0o644)
        elif f.is_dir():
            f.chmod(0o755)
    opt_dir.chmod(0o755)

    subprocess.call(["fakeroot", "dpkg-deb", "--build", deb_root])
    shutil.rmtree(str(deb_root))
    return deb_root.with_name(deb_root.name + ".deb")
