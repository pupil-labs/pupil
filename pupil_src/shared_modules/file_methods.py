'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

try:
    import cPickle as pickle
except ImportError:
    import pickle

UnpicklingError = pickle.UnpicklingError
import os
import traceback as tb
import logging
logger = logging.getLogger(__name__)


class Persistent_Dict(dict):
    """a dict class that uses pickle to save inself to file"""
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = os.path.expanduser(file_path)
        try:
            self.update(load_object(self.file_path))
        except IOError:
            logger.debug("Session settings file '{}' not found. Will make new one on exit.".format(self.file_path))
        except:  # KeyError, EOFError
            logger.warning("Session settings file '{}'could not be read. Will overwrite on exit.".format(self.file_path))
            logger.debug(tb.format_exc())

    def save(self):
        d = {}
        d.update(self)
        try:
            save_object(d, self.file_path)
        except IOError:
            logger.warning("Could not save session settings to '{}'".format(self.file_path))

    def close(self):
        self.save()


def load_object(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'rb') as fh:
        data = pickle.load(fh, encoding='bytes')
    return data


def save_object(object_, file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'wb') as fh:
        pickle.dump(object_,fh, -1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # settings = Persistent_Dict("~/Desktop/test")
    # settings['f'] = "this is a test"
    # settings['list'] = ["list 1","list2"]
    # settings.close()

    # save_object("string",'test')
    # print load_object('test')
    # settings = Persistent_Dict('~/Desktop/pupil_settings/user_settings_eye')
    # print settings['roi']


    # example. Write out pupil data into csv file.
    from time import time
    t = time()
    l = load_object('/Users/mkassner/Downloads/data/pupil_data')
    print(l['notifications'])
    print(t-time())
    # t = time()
    # save_object(l,'/Users/mkassner/Downloads/data/pupil_data2')
    # print(t-time())
    import csv
    with open(os.path.join('/Users/mkassner/Pupil/pupil_code/pupil_src/capture/pupil_postions.csv'), 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(('timestamp',
                             'id',
                             'confidence',
                             'norm_pos_x',
                             'norm_pos_y',
                             'diameter',
                             'method',
                             'ellipse_center_x',
                             'ellipse_center_y',
                             'ellipse_axis_a',
                             'ellipse_axis_b',
                             'ellipse_angle'))
        for p in l['pupil_positions']:
            data_2d = [str(p['timestamp']),  # use str to be consitant with csv lib.
                       p['id'],
                       p['confidence'],
                       p['norm_pos'][0],
                       p['norm_pos'][1],
                       p['diameter'],
                       p['method']]
            try:
                ellipse_data = [p['ellipse']['center'][0],
                                p['ellipse']['center'][1],
                                p['ellipse']['axes'][0],
                                p['ellipse']['axes'][1],
                                p['ellipse']['angle']]
            except KeyError:
                ellipse_data = [None]*5

            row = data_2d + ellipse_data
            csv_writer.writerow(row)

