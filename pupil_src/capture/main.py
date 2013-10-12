'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import sys, os
from time import sleep
from ctypes import c_bool, c_int
from multiprocessing import Process, Pipe, Event,Queue
from multiprocessing.sharedctypes import RawValue, Value, Array


if getattr(sys, 'frozen', False):
    user_dir = os.path.join(sys._MEIPASS.rsplit(os.path.sep,1)[0],"settings")
    rec_dir = os.path.join(sys._MEIPASS.rsplit(os.path.sep,1)[0],"recordings")
else:
    # We are running in a normal Python environment.
    # Make all pupil shared_modules available to this Python session.
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))
	# Specifiy user dirs.
    rec_dir = os.path.join(pupil_base_dir,'recordings')
    user_dir = os.path.join(pupil_base_dir,'settings')

from methods import Temp
from git_version import get_tag_commit


#if you pass any additional argument when calling this script. The profiler will be used.
if len(sys.argv) >=2:
    from eye import eye_profiled as eye
    from world import world_profiled as world
else:
    from eye import eye
    from world import world



def main():
    #get the current software version
    if getattr(sys, 'frozen', False):
        with open(os.path.join(sys._MEIPASS,'_version_string_')) as f:
            version = f.read()
    else:
        version = get_tag_commit()

    # create folder for user settings, tmp data and a recordings folder
    if not os.path.isdir(user_dir):
        os.mkdir(user_dir)
    if not os.path.isdir(rec_dir):
        os.mkdir(rec_dir)



    # To assign by name: put string(s) in list
    eye_src = ["Microsoft", "6000"]
    world_src = ["Logitech Camera","B525", "C525","C615","C920","C930e"]

    # to assign cameras directly, using integers as demonstrated below
    # eye_src = 1
    # world_src = 0

    # to use a pre-recorded video.
    # Use a string to specify the path to your video file as demonstrated below
    # eye_src = "/Users/mkassner/Downloads/wetransfer-fe724a/eye.avi"
    # world_src = "/Users/mkassner/Downloads/wetransfer-fe724a/world.avi"

    # Camera video size in pixels (width,height)
    eye_size = (640,360)
    world_size = (1280,720)

    # Create and initialize IPC
    g_pool = Temp()
    g_pool.pupil_queue = Queue()
    g_pool.eye_rx, g_pool.eye_tx = Pipe(False)
    g_pool.quit = RawValue(c_bool,0)
    # make some constants avaiable
    g_pool.user_dir = user_dir
    g_pool.rec_dir = rec_dir
    g_pool.version = version
    # set up subprocesses
    p_eye = Process(target=eye, args=(g_pool,eye_src,eye_size))

    # spawn subprocess
    p_eye.start()
    # On Linux, we need to give the camera driver some time before requesting another camera.
    sleep(0.5)
    # On MacOS cameras using MJPG compression (world camera) need to run in the main process.
    world(g_pool,world_src,world_size)

    # Exit / clean-up
    p_eye.join()

if __name__ == '__main__':
    main()
