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
    
import os
import logging
logger = logging.getLogger(__name__)

class Persistent_Dict(dict):
    """a dict class that uses pickle to save inself to file"""
    def __init__(self, file_path):
        super(Persistent_Dict, self).__init__()
        self.file_path = os.path.expanduser(file_path)
        try:
            with open(self.file_path,'rb') as fh:
                try:
                    self.update(pickle.load(fh))
                except: #KeyError,EOFError
                    logger.warning("Session settings file '%s'could not be read. Will overwrite on exit."%self.file_path)
        except IOError:
            logger.debug("Session settings file '%s' not found. Will make new one on exit."%self.file_path)


    def save(self):
        d = {}
        d.update(self)
        try:
            with open(self.file_path,'wb') as fh:
                pickle.dump(d,fh,-1)
        except IOError:
            logger.warning("Could not save session settings to '%s'"%self.file_path)


    def close(self):
        self.save()


def load_object(file_path):
    file_path = os.path.expanduser(file_path)
    #reading to string and loads is 2.5x faster that using the file handle and load.
    with open(file_path,'rb') as fh:
        data = fh.read()
    # encoding='latin1' enables us to import python2 data directly 
    # but is a workaround - ideally we import bytes and then set encoding on all k,v pairs
    return pickle.loads(data,encoding='latin1')

def save_object(object,file_path):
    file_path = os.path.expanduser(file_path)
    data = pickle.dumps(object,-1)
    with open(file_path,'wb') as fh:
        data = fh.write(data)


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
    l = load_object('/Users/mkassner/Pupil/pupil_code/pupil_src/capture/pupil_data')
    import csv
    with open(os.path.join('/Users/mkassner/Pupil/pupil_code/pupil_src/capture/pupil_postions.csv'),'wb') as csvfile:
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
        for p in l:
            data_2d = [ '%s'%p['timestamp'],  #use str to be consitant with csv lib.
                        p['id'],
                        p['confidence'],
                        p['norm_pos'][0],
                        p['norm_pos'][1],
                        p['diameter'],
                        p['method'] ]
            try:
                ellipse_data = [p['ellipse']['center'][0],
                                p['ellipse']['center'][1],
                                p['ellipse']['axes'][0],
                                p['ellipse']['axes'][1],
                                p['ellipse']['angle'] ]
            except KeyError:
                ellipse_data = [None,]*5

            row = data_2d + ellipse_data
            csv_writer.writerow(row)

