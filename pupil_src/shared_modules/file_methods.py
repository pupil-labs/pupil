'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import pickle
import msgpack
import os
import numpy as np
import traceback as tb
import logging
from glob import iglob
logger = logging.getLogger(__name__)
UnpicklingError = pickle.UnpicklingError


class Persistent_Dict(dict):
    """a dict class that uses pickle to save inself to file"""
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = os.path.expanduser(file_path)
        try:
            self.update(**load_object(self.file_path,allow_legacy=False))
        except IOError:
            logger.debug("Session settings file '{}' not found. Will make new one on exit.".format(self.file_path))
        except:  # KeyError, EOFError
            logger.warning("Session settings file '{}'could not be read. Will overwrite on exit.".format(self.file_path))
            logger.debug(tb.format_exc())

    def save(self):
        d = {}
        d.update(self)
        save_object(d, self.file_path)

    def close(self):
        self.save()


def _load_object_legacy(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'rb') as fh:
        data = pickle.load(fh, encoding='bytes')
    return data


def load_object(file_path,allow_legacy=True):
    import gc
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'rb') as fh:
        try:
            gc.disable()  # speeds deserialization up.
            data = msgpack.unpack(fh, encoding='utf-8')
        except Exception as e:
            if not allow_legacy:
                raise e
            else:
                logger.info('{} has a deprecated format: Will be updated on save'.format(file_path))
                data = _load_object_legacy(file_path)
        finally:
            gc.enable()
    return data


def save_object(object_, file_path):

    def ndarrray_to_list(o, _warned=[False]): # Use a mutlable default arg to hold a fn interal temp var.
        if isinstance(o, np.ndarray):
            if not _warned[0]:
                logger.warning("numpy array will be serialized as list. Invoked at:\n"+''.join(tb.format_stack()))
                _warned[0] = True
            return o.tolist()
        return o

    file_path = os.path.expanduser(file_path)
    with open(file_path, 'wb') as fh:
        msgpack.pack(object_, fh, use_bin_type=True,default=ndarrray_to_list)


def next_export_sub_dir(root_export_dir):
    # match any sub directories or files a three digit pattern
    pattern = os.path.join(root_export_dir, '[0-9][0-9][0-9]')
    existing_subs = sorted(iglob(pattern))
    try:
        latest = os.path.split(existing_subs[-1])[-1]
        next_sub_dir = '{:03d}'.format(int(latest) + 1)
    except IndexError:
        next_sub_dir = '000'

    return os.path.join(root_export_dir, next_sub_dir)
