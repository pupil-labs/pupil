'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os, sys, platform
import logging
logger = logging.getLogger()

if getattr(sys, 'frozen', False):
    # Specifiy user dirs.
    user_dir = os.path.expanduser(os.path.join('~','pupil_capture_settings'))
    version_file = os.path.join(sys._MEIPASS,'_version_string_')
else:
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))
    # Specifiy user dir.
    user_dir = os.path.join(pupil_base_dir,'capture_settings')
    version_file = None


sys.path.append('/usr/local/Cellar/opencv3/3.0.0/lib/python2.7/site-packages')
# create folder for user settings, tmp data
if not os.path.isdir(user_dir):
    os.mkdir(user_dir)

#app version
from version_utils import get_version
app_version = get_version(version_file)


if platform.system() == 'Darwin' and getattr(sys, 'frozen', False):
    from billiard import Process, Pipe, Queue, Value,freeze_support,active_children, forking_enable
    forking_enable(0)
else:
    from multiprocessing import Process, Pipe, Queue, Value,active_children, freeze_support

from ctypes import c_double,c_bool

if 'profiled' in sys.argv:
    from world import world_profiled as world
    from eye import eye_profiles as eye
    logger.warning("Profiling active.")
else:
    from world import world
    from eye import eye

# To assign camera by name: put string(s) in list
world_src = ["Pupil Cam1 ID2","Logitech Camera","(046d:081d)","C510","B525", "C525","C615","C920","C930e"]
eye0_src = ["Pupil Cam1 ID0","HD-6000","Integrated Camera","HD USB Camera","USB 2.0 Camera"]
eye1_src = ["Pupil Cam1 ID1","HD-6000","Integrated Camera"]

# to use a pre-recorded video.
# Use a string to specify the path to your video file as demonstrated below
# world_src = "/Users/mkassner/Downloads/000/world.mkv"
# eye0_src = '/Users/mkassner/Downloads/eye0.mkv'
# eye1_src =  '/Users/mkassner/Downloads/eye.avi'

video_sources = {'world':world_src,'eye0':eye0_src,'eye1':eye1_src}

def main():

    #IPC
    pupil_queue = Queue()
    timebase = Value(c_double,0)

    cmd_world_end,cmd_launcher_end = Pipe()
    com0 = Pipe(True)
    eyes_are_alive = Value(c_bool,0),Value(c_bool,0)
    com1 = Pipe(True)
    com_world_ends = com0[0],com1[0]
    com_eye_ends = com0[1],com1[1]


    p_world = Process(target=world,args=(pupil_queue,
                                        timebase,
                                        cmd_world_end,
                                        com_world_ends,
                                        eyes_are_alive,
                                        user_dir,
                                        app_version,
                                        video_sources['world'] ))
    p_world.start()

    while True:
        #block and listen for commands from world process.
        cmd = cmd_launcher_end.recv()
        if cmd == "Exit":
            break
        else:
            eye_id = cmd
            p_eye = Process(target=eye,args=(pupil_queue,
                                            timebase,
                                            com_eye_ends[eye_id],
                                            eyes_are_alive[eye_id],
                                            user_dir,
                                            app_version,
                                            eye_id,
                                            video_sources['eye%s'%eye_id] ))
            p_eye.start()

    for p in active_children(): p.join()
    logger.debug('Laucher exit')

if __name__ == '__main__':
    freeze_support()
    main()
