"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import contextlib
import logging
import os
import threading

import numpy as np

from .utils import SCAN_PATH_GAZE_DATUM_DTYPE, scan_path_zeros_numpy_array

logger = logging.getLogger(__name__)


class ScanPathStorage:

    version = 1

    def __init__(self, rec_dir, gaze_data=...):
        self.__lock = threading.RLock()
        self.rec_dir = rec_dir
        if gaze_data is ...:
            self.gaze_data = None
        else:
            self.gaze_data = gaze_data

    @property
    def gaze_data(self):
        with self._locked():
            return self._gaze_data

    @gaze_data.setter
    def gaze_data(self, gaze_data):
        with self._locked():
            if gaze_data is not None:
                self._validate_gaze_data(gaze_data)
                gaze_data = gaze_data.view(np.recarray)
            self._gaze_data = gaze_data

    @property
    def is_valid(self) -> bool:
        return self._gaze_data is not None

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    def mark_invalid(self):
        with self._locked():
            self._gaze_data = None
            self._is_complete = False
            self.__remove_from_disk()

    def mark_complete(self):
        with self._locked():
            self._is_complete = True
            self.__save_to_disk()

    def load_from_disk(self):
        with self._locked():
            self.__load_from_disk()

    @staticmethod
    def empty_gaze_data():
        gaze_data = scan_path_zeros_numpy_array()
        ScanPathStorage._validate_gaze_data(gaze_data)
        return gaze_data

    @staticmethod
    def _validate_gaze_data(gaze_data):
        assert isinstance(gaze_data, np.ndarray)
        assert gaze_data.dtype == SCAN_PATH_GAZE_DATUM_DTYPE
        assert len(gaze_data.shape) == 1

    @contextlib.contextmanager
    def _locked(self):
        self.__lock.acquire()
        try:
            yield
        finally:
            self.__lock.release()

    # Filesystem

    @property
    def __file_path(self) -> str:
        rec_dir = self.rec_dir
        filename = f"scan_path_cache_v{self.version}.npy"
        return os.path.join(rec_dir, "offline_data", filename)

    def __load_from_disk(self):
        try:
            gaze_data = np.load(self.__file_path)
        except OSError:
            return
        self.gaze_data = gaze_data
        # TODO: Figure out if gaze_data is complete
        self._is_complete = self.is_valid

    def __save_to_disk(self):
        if not self.is_valid:
            return
        np.save(self.__file_path, self._gaze_data)

    def __remove_from_disk(self):
        try:
            os.remove(self.__file_path)
        except FileNotFoundError:
            pass
