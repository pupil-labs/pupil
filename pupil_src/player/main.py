'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

if __name__ == '__main__':
    # make shared modules available across pupil_src
    from sys import path as syspath
    from os import path as ospath
    loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
    syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
    del syspath, ospath


import os, sys
import shelve
from ctypes import  c_int,c_bool,c_float,create_string_buffer
import numpy as np

#display
from glfw import *
import atb

# helpers/utils
from methods import normalize, denormalize,Temp
from gl_utils import basic_gl_setup, adjust_gl_view, draw_gl_texture, clear_gl_screen, draw_gl_point_norm,draw_gl_texture
# Plug-ins
from display_gaze import Display_Gaze
from marker_detector import Marker_Detector

import logging
# Set up root logger for the main process before doing imports of logged modules.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('player.log'),mode='w')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
# create formatter and add it to the handlers
formatter = logging.Formatter('Player: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
formatter = logging.Formatter('Player [%(levelname)s] %(name)s : %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
# mute OpenGL logger
logging.getLogger("OpenGL").propagate = False
logging.getLogger("OpenGL").addHandler(logging.NullHandler())



data_folder = "/Users/mkassner/Pupil/pupil_code/recordings/2013_12_11/000"

video_path = data_folder + "/world.avi"
timestamps_path = data_folder + "/timestamps.npy"
gaze_positions_path = data_folder + "/gaze_positions.npy"
record_path = data_folder + "/world_viz.avi"


#deal with older recordings that use a different coodinate system.
with open(data_folder + "/info.csv") as info:
    data = dict( ((line.strip().split('\t')) for line in info.readlines() ) )
version = [v for k,v in data.iteritems() if "Capture Software Version" in k ][0]
version = int(filter(type(version).isdigit, version)[:3]) #(get major,minor,fix of version)

# glfw close global
quit = False



def main():



        # Callback functions
    def on_resize(window,w, h):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        adjust_gl_view(w,h)
        atb.TwWindowSize(w, h)
        glfwMakeContextCurrent(active_window)

    def on_key(window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,action):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    on_close(window)

    def on_char(window,char):
        if not atb.TwEventCharGLFW(char,1):
            pass

    def on_button(window,button, action, mods):
        if not atb.TwEventMouseButtonGLFW(button,action):
            pos = glfwGetCursorPos(window)
            pos = normalize(pos,glfwGetWindowSize(world_window))
            pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
            for p in g.plugins:
                p.on_click(pos,button,action)

    def on_pos(window,x, y):
        if atb.TwMouseMotion(int(x),int(y)):
            pass

    def on_scroll(window,x,y):
        if not atb.TwMouseWheel(int(x)):
            pass

    def on_close(window):
        quit = True
        logger.info('Process closing from window')







if __name__ == '__main__':
    main()