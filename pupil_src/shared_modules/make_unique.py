"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import re


def by_number_at_end(new_string, existing_strings):
    """
    Makes a string unique by adding a number at the end if necessary.
    Example: If new_string is "Hello", and existing_strings is ["Hello", "World"],
    then it will return "Hello 2". If "Hello 2" already exists, it returns
    "Hello 3" and so on.
    """
    if new_string not in existing_strings:
        return new_string
    # if there is a number at the end, remove it (e.g. "Hello 4" -> "Hello")
    stripped_string = re.sub(r" \d+$", "", new_string)
    if stripped_string not in existing_strings:
        return stripped_string
    lowest_unused_number = 2
    while True:
        new_string = f"{stripped_string} {lowest_unused_number}"
        if new_string not in existing_strings:
            return new_string
        lowest_unused_number += 1
