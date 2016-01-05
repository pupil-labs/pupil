'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys

if 'profiled' in sys.argv:
    logger.debug("Capture processes will be profiled.")
    from world import world_profiled as world
else:
    from world import world

from world import freeze_support

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
    world(video_sources)

if __name__ == '__main__':
    freeze_support()
    main()
