import enum

from version_utils import ParsedVersion, pupil_version, write_version_file

from . import linux, macos, windows


class SupportedPlatform(enum.Enum):
    macos = "Darwin"
    linux = "Linux"
    windows = "Windows"


ICON_EXT = {
    SupportedPlatform.macos: ".icns",
    SupportedPlatform.linux: ".svg",
    SupportedPlatform.windows: ".ico",
}
LIB_EXT = {
    SupportedPlatform.macos: ".dylib",
    SupportedPlatform.linux: ".so",
    SupportedPlatform.windows: ".dll",
}

__all__ = [
    "ParsedVersion",
    "SupportedPlatform",
    "ICON_EXT",
    "LIB_EXT",
    "linux",
    "macos",
    "pupil_version",
    "windows",
    "write_version_file",
]
