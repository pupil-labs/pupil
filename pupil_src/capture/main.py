'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os, sys, platform


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

# create folder for user settings, tmp data
if not os.path.isdir(user_dir):
    os.mkdir(user_dir)


if platform.system() in ('Darwin','Linux1'):
    from billiard import freeze_support
else:
    from multiprocessing import freeze_support

if 'profiled' in sys.argv:
    from world import world_profiled as world
else:
    from world import world


def main():

    # To assign camera by name: put string(s) in list
    default_world = ["Pupil Cam1 ID2","Logitech Camera","(046d:081d)","C510","B525", "C525","C615","C920","C930e"]
    default_eye0 = ["Pupil Cam1 ID0","HD-6000","Integrated Camera","HD USB Camera","USB 2.0 Camera"]
    default_eye1 = ["Pupil Cam1 ID1","HD-6000","Integrated Camera"]

    # to use a pre-recorded video.
    # Use a string to specify the path to your video file as demonstrated below
    # default_world_src = "/Users/mkassner/Downloads/000/world.mkv"
    # default_eye0 = '/Users/mkassner/Downloads/eye0.mkv'
    # default_eye1 =  '/Users/mkassner/Downloads/eye.avi'

    video_sources = {'world':default_world,'eye0':default_eye0,'eye1':default_eye1}


    #start the world fn
    world(user_dir,version_file,video_sources)

if __name__ == '__main__':
    freeze_support()
    main()
