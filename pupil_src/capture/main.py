'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os,platform
from time import sleep
from ctypes import c_bool, c_int,c_double
if platform.system() == 'Darwin':
    from billiard import Process, Pipe, Event,Queue,forking_enable,freeze_support
    from billiard.sharedctypes import RawValue, Value, Array
else:
    from multiprocessing import Process, Pipe, Event, Queue
    forking_enable = lambda x: x #dummy fn
    from multiprocessing import freeze_support
    from multiprocessing.sharedctypes import RawValue, Value, Array

if getattr(sys, 'frozen', False):
    if platform.system() == 'Darwin':
        user_dir = os.path.expanduser('~/Desktop/pupil_settings')
        rec_dir = os.path.expanduser('~/Desktop/pupil_recordings')
        version_file = os.path.join(sys._MEIPASS,'_version_string_')
    else:
        # Specifiy user dirs.
        user_dir = os.path.join(sys._MEIPASS.rsplit(os.path.sep,1)[0],"settings")
        rec_dir = os.path.join(sys._MEIPASS.rsplit(os.path.sep,1)[0],"recordings")
        version_file = os.path.join(sys._MEIPASS,'_version_string_')


else:
    # We are running in a normal Python environment.
    # Make all pupil shared_modules available to this Python session.
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))
	# Specifiy user dirs.
    rec_dir = os.path.join(pupil_base_dir,'recordings')
    user_dir = os.path.join(pupil_base_dir,'settings')


# create folder for user settings, tmp data and a recordings folder
if not os.path.isdir(user_dir):
    os.mkdir(user_dir)
if not os.path.isdir(rec_dir):
    os.mkdir(rec_dir)


import logging
# Set up root logger for the main process before doing imports of logged modules.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(user_dir,'world.log'),mode='w')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
# create formatter and add it to the handlers
formatter = logging.Formatter('World Process: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
formatter = logging.Formatter('WORLD Process [%(levelname)s] %(name)s : %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
# mute OpenGL logger
logging.getLogger("OpenGL").propagate = False
logging.getLogger("OpenGL").addHandler(logging.NullHandler())


#if you pass any additional argument when calling this script. The profiler will be used.
if len(sys.argv) >=2:
    from eye import eye_profiled as eye
    from world import world_profiled as world
else:
    from eye import eye
    from world import world

from methods import Temp

#get the current software version
if getattr(sys, 'frozen', False):
    with open(version_file) as f:
        version = f.read()
else:
    from git_version import get_tag_commit
    version = get_tag_commit()


def main():
    # To assign camera by name: put string(s) in list
    eye_src = ["Microsoft", "6000","Integrated Camera"]
    world_src = ["Logitech Camera","(046d:081d)","C510","B525", "C525","C615","C920","C930e"]

    # to assign cameras directly, using integers as demonstrated below
    # eye_src = 1
    # world_src = 0

    # to use a pre-recorded video.
    # Use a string to specify the path to your video file as demonstrated below
    # eye_src = '/Users/mkassner/Pupil/datasets/p1-left/frames/test.avi'
    # world_src = "/Users/mkassner/Desktop/2014_01_21/000/world.avi"

    # Camera video size in pixels (width,height)
    eye_size = (640,360)
    world_size = (1280,720)


    # on MacOS we will not use os.fork, elsewhere this does nothing.
    forking_enable(0)

    # Create and initialize IPC
    g_pool = Temp()
    g_pool.pupil_queue = Queue()
    g_pool.eye_rx, g_pool.eye_tx = Pipe(False)
    g_pool.quit = RawValue(c_bool,0)
    # this value will be substracted form the capture timestamp
    g_pool.timebase = RawValue(c_double,0)
    # make some constants avaiable
    g_pool.user_dir = user_dir
    g_pool.rec_dir = rec_dir
    g_pool.version = version
    g_pool.app = 'capture'
    # set up subprocesses
    p_eye = Process(target=eye, args=(g_pool,eye_src,eye_size))

    # Spawn subprocess:
    p_eye.start()
    if platform.system() == 'Linux':
        # We need to give the camera driver some time before requesting another camera.
        sleep(0.5)

    world(g_pool,world_src,world_size)

    # Exit / clean-up
    p_eye.join()

if __name__ == '__main__':
    freeze_support()
    main()
