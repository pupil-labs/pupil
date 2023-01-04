"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
import sys

import numpy as np
from surface_tracker.surface_file_store import Surface_File_Store
from surface_tracker.surface_offline import Surface_Offline as Any_Surface

from .fixtures import (
    surface_definition_v00_dir,
    surface_definition_v01_after_update_dir,
    surface_definition_v01_before_update_dir,
)


def test_file_store_v00():
    _test_file_store_read_write(
        root_dir=surface_definition_v00_dir(),
    )


def test_file_store_v01_before_update():
    _test_file_store_read_write(
        root_dir=surface_definition_v01_before_update_dir(),
    )


def test_file_store_v01_after_update():
    _test_file_store_read_write(
        root_dir=surface_definition_v01_after_update_dir(),
    )


##### PRIVATE


def _test_file_store_read_write(root_dir: str, surface_class=Any_Surface):
    file_store = Surface_File_Store(parent_dir=root_dir)

    surfaces = file_store.read_surfaces_from_file(surface_class=surface_class)
    surfaces = list(surfaces)

    assert len(surfaces) > 0
    assert all(isinstance(surface, surface_class) for surface in surfaces)

    file_store.write_surfaces_to_file(surfaces=surfaces)

    assert os.path.isfile(file_store.file_path)
    assert os.path.dirname(file_store.file_path) == root_dir
