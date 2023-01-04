"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2021 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import argparse
import fnmatch
import os
import re

license_txt = """\
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)\
"""


pattern = re.compile(
    "(\"{3}|'{3}|[/][*])\n\\([*]\\)~(.+?)~\\([*]\\)\n(\"{3}|'{3}|[*][/])(\r\n|\r|\n)*",
    re.DOTALL | re.MULTILINE,
)

# choose files types to include
# choose directories to exclude from search
includes = ["*.py", "*.c", "*.cpp", "*.hpp", "*.h", "*.pxd", "*.pyx", "*.pxi"]
excludes = [
    "recordings*",
    "shader.py",
    "singleeyefitter*",
    "vertex_buffer.py",
    "gprof2dot.py",
    "git_version.py",
    "transformations.py",
    "libuvcc*",
    ".gitignore",
    "version_utils.py",
    "update_license_header.py",
    ".venv*",
]

# transform glob patterns to regular expressions
includes = r"|".join([fnmatch.translate(x) for x in includes])
excludes = r"|".join([fnmatch.translate(x) for x in excludes]) or r"$."


def get_files(start_dir, includes, excludes):
    # use os.walk to recursively dig down into the Pupil directory
    match_files = []
    for root, dirs, files in os.walk(start_dir):
        if not re.search(excludes, root):
            files = [
                f
                for f in files
                if re.search(includes, f) and not re.search(excludes, f)
            ]
            files = [os.path.join(root, f) for f in files]
            match_files += files
        else:
            print("Excluding '%s'" % root)

    return match_files


def write_header(file_name, license_txt, delete_header=False, dry_run: bool = False):
    # find and replace license header
    # or add new header if not existing
    c_comment = ["/*\n", "\n*/\n"]
    py_comment = ['"""\n', '\n"""\n']
    file_type = os.path.splitext(file_name)[-1]

    if file_type in (".py", ".pxd", ".pyx", ".pxi"):
        license_txt = py_comment[0] + license_txt + py_comment[1]
    elif file_type in (".c", ".cpp", ".hpp", ".h"):
        license_txt = c_comment[0] + license_txt + c_comment[1]
    else:
        raise Exception("Dont know how to deal with this filetype")

    try:
        with open(file_name) as original:
            data = original.read()
    except UnicodeDecodeError:
        return

    if not dry_run:
        with open(file_name, "w") as modified:
            if re.findall(pattern, data):
                if delete_header:
                    license_txt = ""
                # if header already exists, then update, but dont add the last newline.
                modified.write(re.sub(pattern, license_txt, data))
            else:
                # else write the license header
                modified.write(license_txt + data)
    else:
        print(f"Would have modified {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default=".")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("-n", "--dry-run", action="store_true")
    args = parser.parse_args()

    # Add a license/docstring header to selected files
    match_files = get_files(args.directory, includes, excludes)
    print(f"Number of files to check: {len(match_files)}")

    for f in match_files:
        print(f"Checking {f}")
        write_header(f, license_txt, delete_header=args.delete, dry_run=args.dry_run)
