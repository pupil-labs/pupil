'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os, platform
from time import sleep
from ctypes import c_bool, c_double

if platform.system() == 'Darwin':
    from billiard import Process, Pipe, Queue, Value, freeze_support, forking_enable
else:
    from multiprocessing import Process, Pipe, Queue, Value, freeze_support
    forking_enable = lambda _: _ #dummy fn

if getattr(sys, 'frozen', False):
    # Specifiy user dirs.
    user_dir = os.path.expanduser("~/pupil_capture_settings")
    rec_dir = os.path.expanduser("~/pupil_recordings")
    version_file = os.path.join(sys._MEIPASS,'_version_string_')
else:
    # We are running in a normal Python environment.
    # Make all pupil shared_modules available to this Python session.
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))
	# Specifiy user dirs.
    rec_dir = os.path.join(pupil_base_dir,'recordings')
    user_dir = os.path.join(pupil_base_dir,'capture_settings')
    version_file = None


# create folder for user settings, tmp data and a recordings folder
if not os.path.isdir(user_dir):
    os.mkdir(user_dir)
if not os.path.isdir(rec_dir):
    os.mkdir(rec_dir)

from version_utils import get_version

import logging
# Set up root logger for the main process before doing imports of logged modules.
logger = logging.getLogger()
if 'debug' in sys.argv:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(user_dir,'world.log'),mode='w')
fh.setLevel(logger.level)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logger.level+10)
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


if 'binocular' in sys.argv:
    binocular = True
    logger.debug("Starting in binocular mode")
else:
    binocular = False
    logger.debug("Starting in single eye cam mode")


if 'profiled' in sys.argv:
    logger.debug("Capture processes will be profiled.")
    from eye import eye_profiled as eye
    from world import world_profiled as world
else:
    from eye import eye
    from world import world


class Global_Container(object):
    pass

def main():
    # To assign camera by name: put string(s) in list
    eye_cam_names = ["USB 2.0 Camera","Microsoft", "6000","Integrated Camera"]
    world_src = ["Logitech Camera","(046d:081d)","C510","B525", "C525","C615","C920","C930e"]
    eye_src = (eye_cam_names,0),(eye_cam_names,1) #first match for eye0 and second match for eye1

    # to assign cameras directly, using integers as demonstrated below
    # eye_src =  4 , 5 #second arg will be ignored for monocular eye trackers
    # world_src = 1

    # to use a pre-recorded video.
    # Use a string to specify the path to your video file as demonstrated below
    # eye_src = '/Users/mkassner/Downloads/eye.avi' , '/Users/mkassner/Downloads/eye.avi'
    # world_src = "/Users/mkassner/Desktop/2014_01_21/000/world.avi"

    # Camera video size in pixels (width,height)
    eye_size = (640,480)
    world_size = (1280,720)


    # on MacOS we will not use os.fork, elsewhere this does nothing.
    forking_enable(0)

    #g_pool holds variables. Only if added here they are shared across processes.
    g_pool = Global_Container()

    # Create and initialize IPC
    g_pool.pupil_queue = Queue()
    g_pool.quit = Value(c_bool,0)
    g_pool.timebase = Value(c_double,0)
    g_pool.eye_tx = []
    # make some constants avaiable
    g_pool.user_dir = user_dir
    g_pool.rec_dir = rec_dir
    g_pool.version = get_version(version_file)
    g_pool.app = 'capture'
    g_pool.binocular = binocular


    p_eye = []
    for eye_id in range(1+1*binocular):
        rx,tx = Pipe(False)
        p_eye += [Process(target=eye, args=(g_pool,eye_src[eye_id],eye_size,rx,eye_id))]
        g_pool.eye_tx += [tx]
        p_eye[-1].start()
        if platform.system() == 'Linux':
            # We need to give the camera driver some time before requesting another camera.
            sleep(0.5)

    world(g_pool,world_src,world_size)


    # Exit / clean-up
    for p in p_eye:
        p.join()

if __name__ == '__main__':
    freeze_support()
    main()
