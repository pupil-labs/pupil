# Author: Douglas Creager <dcreager@dcreager.net>
# Changes, Additions: Moritz Kassner <moritz@pupil-labs.com>, Will Patera <will@pupil-labs.com>
# This file is placed into the public domain.

import logging
import os
import pathlib
import sys
import typing as T
from subprocess import STDOUT, CalledProcessError, check_output

import packaging.version

logger = logging.getLogger(__name__)


def get_tag_commit() -> T.Optional[str]:
    """
    returns string: 'tag'-'commits since tag'-'7 digit commit id'
    """
    try:
        desc_tag = check_output(
            ["git", "describe", "--tags", "--long"],
            stderr=STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        desc_tag = desc_tag.decode("utf-8")
        desc_tag = desc_tag.replace("\n", "")  # strip newlines
        return desc_tag
    except CalledProcessError as e:
        logger.error(f'Error calling git: "{e}" \n output: "{e.output}"')
        return None
    except OSError as e:
        logger.error(f'Could not call git, is it installed? error msg: "{e}"')
        return None


ParsedVersion = T.Union[packaging.version.LegacyVersion, packaging.version.Version]


def parse_version(vstring: str) -> ParsedVersion:
    return packaging.version.parse(vstring)


def pupil_version() -> ParsedVersion:
    return parse_version(pupil_version_string())


def pupil_version_string() -> str:
    """
    [major].[minor].[trailing-untagged-commits]
    """
    # NOTE: This returns the current version as read from the last git tag. Normally you
    # don't want to use this, but get_version() below, which also works in a bundled
    # version (without git).
    version = get_tag_commit()
    if version is None:
        raise ValueError("Version Error")

    try:
        parts_git_tag = version.split("-")
        version_parsed = packaging.version.Version(parts_git_tag[0])
        if version_parsed.is_prerelease:
            version = version_parsed.base_version
    except packaging.version.InvalidVersion:
        pass
    version = version.replace("v", "")  # strip version 'v'
    # print(version)
    if "-" in version:
        parts = version.split("-")
        version = ".".join(parts[:-1])
    if version_parsed.is_prerelease:
        version += "".join(map(str, version_parsed.pre))
    return version


def get_version():
    # get the current software version
    if getattr(sys, "frozen", False):
        version_file = os.path.join(sys._MEIPASS, "_version_string_")
        with open(version_file) as f:
            version_string = f.read()
    else:
        version_string = pupil_version_string()
    logger.debug(f"Running version: {version_string}")
    return parse_version(version_string)


def write_version_file(target_dir: str) -> pathlib.Path:
    version_string = pupil_version_string()
    version_file = os.path.join(target_dir, "_version_string_")
    logger.debug(f"Writing Pupil Core version '{version_string}' to {version_file}")
    with open(version_file, "w") as f:
        f.write(version_string)
    return pathlib.Path(version_file)


if __name__ == "__main__":
    print(f"{get_tag_commit()=}")
    print(f"{pupil_version_string()=}")
    print(f"{pupil_version()=}")
