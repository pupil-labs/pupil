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
from time import time,sleep
from ctypes import  c_int,c_bool,c_float,create_string_buffer
import numpy as np

#display
from glfw import *
import atb


from uvc_capture import autoCreateCapture
# helpers/utils
from methods import normalize, denormalize,Temp
from gl_utils import basic_gl_setup, adjust_gl_view, draw_gl_texture, clear_gl_screen, draw_gl_point_norm,draw_gl_texture
# Plug-ins
from display_gaze import Display_Gaze
from seek_bar import Seek_Bar


import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('player.log',mode='w')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
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

logger = logging.getLogger(__name__)


version = "dev"

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
                    pass


    def on_char(window,char):
        if not atb.TwEventCharGLFW(char,1):
            pass

    def on_button(window,button, action, mods):
        if not atb.TwEventMouseButtonGLFW(button,action):
            pos = glfwGetCursorPos(window)
            pos = normalize(pos,glfwGetWindowSize(main_window))
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
        glfwSetWindowShouldClose(window)
        logger.info('Process closing from window')


    try:
        data_folder = sys.argv[1]
    except:
        logger.warning("You did not supply a datafolder when you called this script. \
               \nI will use the path hardcoded into the script instead.")
        data_folder = "/Users/mkassner/Pupil/pupil_code/recordings/2013_12_11/000"

    #parse and load data folder info
    video_path = data_folder + "/world.avi"
    timestamps_path = data_folder + "/timestamps.npy"
    gaze_positions_path = data_folder + "/gaze_positions.npy"
    record_path = data_folder + "/world_viz.avi"


    #parse info.csv file
    with open(data_folder + "/info.csv") as info:
        meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )
    rec_version = [v for k,v in meta_info.iteritems() if "Capture Software Version" in k ][0]
    rec_version_int = int(filter(type(rec_version).isdigit, rec_version)[:3]) #(get major,minor,fix of version)
    logger.debug("Recording version %s , %s"%(rec_version,rec_version_int))


    #load gaze information
    gaze_list = list(np.load(gaze_positions_path))
    timestamps = list(np.load(timestamps_path))
    # gaze_list: gaze x | gaze y | pupil x | pupil y | timestamp
    # timestamps timestamp

    # this takes the timestamps list and makes a list
    # with the length of the number of recorded frames.
    # Each slot conains a list that will have 0, 1 or more assosiated gaze postions.
    positions_by_frame = [[] for i in timestamps]
    frame_idx = 0
    data_point = gaze_list.pop(0)
    gaze_timestamp = data_point[4]

    while gaze_list:
        # if the current gaze point is before the mean of the current world frame timestamp and the next worldframe timestamp
        try:
            t_between_frames = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
        except IndexError:
            break
        if gaze_timestamp <= t_between_frames:
            positions_by_frame[frame_idx].append({'norm_gaze':(data_point[0],data_point[1]),'norm_pupil': (data_point[2],data_point[3]), 'timestamp':gaze_timestamp})
            data_point = gaze_list.pop(0)
            gaze_timestamp = data_point[4]
        else:
            frame_idx+=1



    # load session persistent settings
    session_settings = shelve.open("user_settings",protocol=2)
    def load(var_name,default):
        return session_settings.get(var_name,default)
    def save(var_name,var):
        session_settings[var_name] = var


    # Initialize capture, check if it works
    cap = autoCreateCapture(video_path,timestamps=timestamps_path)
    if cap is None:
        logger.error("Did not receive valid Capture")
        return
    width,height = cap.get_size()


    # Initialize glfw
    glfwInit()
    main_window = glfwCreateWindow(width, height, "Pupil Player: "+meta_info["Recording Name"], None, None)
    glfwMakeContextCurrent(main_window)

    # Register callbacks main_window
    glfwSetWindowSizeCallback(main_window,on_resize)
    glfwSetWindowCloseCallback(main_window,on_close)
    glfwSetKeyCallback(main_window,on_key)
    glfwSetCharCallback(main_window,on_char)
    glfwSetMouseButtonCallback(main_window,on_button)
    glfwSetCursorPosCallback(main_window,on_pos)
    glfwSetScrollCallback(main_window,on_scroll)

    # gl_state settings
    basic_gl_setup()


    # create container for globally scoped vars (within world)
    g = Temp()
    g.plugins = []
    g.play = False
    g.new_seek = True
    g.plugins.append(Display_Gaze(g))
    g.plugins.append(Seek_Bar(g,capture=cap))

    # helpers called by the main atb bar
    def update_fps():
        old_time, bar.timestamp = bar.timestamp, time()
        dt = bar.timestamp - old_time
        if dt:
            bar.fps.value += .05 * (1. / dt - bar.fps.value)

    def set_window_size(mode,data):
        width,height = cap.get_size()
        ratio = (1,.75,.5,.25)[mode]
        w,h = int(width*ratio),int(height*ratio)
        glfwSetWindowSize(main_window,w,h)
        data.value=mode # update the bar.value

    def get_from_data(data):
        """
        helper for atb getter and setter use
        """
        return data.value

    def get_play():
        return g.play

    def set_play(value):
        g.play = value

    atb.init()
    # add main controls ATB bar
    bar = atb.Bar(name = "Controls", label="Controls",
            help="Scene controls", color=(50, 50, 50), alpha=100,valueswidth=150,
            text='light', position=(10, 10),refresh=.1, size=(300, 200))
    bar.next_atb_pos = (10,220)
    bar.fps = c_float(0.0)
    bar.timestamp = time()
    bar.window_size = c_int(load("window_size",0))
    window_size_enum = atb.enum("Display Size",{"Full":0, "Medium":1,"Half":2,"Mini":3})
    bar.version = create_string_buffer(version,512)
    bar.recording_version = create_string_buffer(rec_version,512)
    bar.add_var("fps", bar.fps, step=1., readonly=True)
    bar._fps = c_float(cap.get_fps())
    bar.add_var("recoding fps",bar._fps,readonly=True)
    bar.add_var("display size", vtype=window_size_enum,setter=set_window_size,getter=get_from_data,data=bar.window_size)
    bar.add_var("play",vtype=c_bool,getter=get_play,setter=set_play,key="space")
    bar.add_var("next frame",getter=cap.get_frame_index)
    bar.add_var("version of recording",bar.recording_version, readonly=True, help="version of the capture software used to make this recording")
    bar.add_var("version of player",bar.version, readonly=True, help="version of the Pupil Player")
    bar.add_button("exit", on_close,data=main_window,key="esc")

    #set the last saved window size
    set_window_size(bar.window_size.value,bar.window_size)
    on_resize(main_window, *glfwGetFramebufferSize(main_window))
    glfwSetWindowPos(main_window,0,0)



    while not glfwWindowShouldClose(main_window):

        #grab new frame
        if g.play or g.new_seek:
            new_frame = cap.get_frame()
            g.new_seek = False
            #end of video logic: pause at last frame.
            if not new_frame:
                g.play=False
            else:
                frame = new_frame

        update_fps()


        #new positons and events
        current_pupil_positions = positions_by_frame[frame.index]
        events = None

        # allow each Plugin to do its work.
        for p in g.plugins:
            p.update(frame,current_pupil_positions,events)

        #check if a plugin need to be destroyed
        g.plugins = [p for p in g.plugins if p.alive]

        # render camera image
        glfwMakeContextCurrent(main_window)
        draw_gl_texture(frame.img)

        # render visual feedback from loaded plugins
        for p in g.plugins:
            p.gl_display()

        atb.draw()
        glfwSwapBuffers(main_window)
        glfwPollEvents()


    # de-init all running plugins
    for p in g.plugins:
        p.alive = False
        #reading p.alive actually runs plug-in cleanup
        _ = p.alive

    save('window_size',bar.window_size.value)
    session_settings.close()

    cap.close()
    bar.destroy()
    glfwDestroyWindow(main_window)
    glfwTerminate()
    logger.debug("Process done")



if __name__ == '__main__':
    if 1:
        main()
    else:
        import cProfile,subprocess,os
        cProfile.runctx("main()",{},locals(),"player.pstats")
        loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
        gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
        subprocess.call("python "+gprof2dot_loc+" -f pstats player.pstats | dot -Tpng -o player_cpu_time.png", shell=True)
        print "created cpu time graph for pupil player . Please check out the png next to the main.py file"
