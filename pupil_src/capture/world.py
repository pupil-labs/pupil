'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

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
from time import time
import shelve
import logging
from ctypes import  c_int,c_bool,c_float,create_string_buffer
import numpy as np

#display
from glfw import *
import atb

# helpers/utils
from methods import normalize, denormalize,Temp
from gl_utils import basic_gl_setup, adjust_gl_view, draw_gl_texture, clear_gl_screen, draw_gl_point_norm,draw_gl_texture
from uvc_capture import autoCreateCapture
import calibrate
# Plug-ins
import calibration_routines
import recorder
from show_calibration import Show_Calibration
from display_recent_gaze import Display_Recent_Gaze
from pupil_server import Pupil_Server
from marker_detector import Marker_Detector

# create logger for the context of this function
logger = logging.getLogger(__name__)


def world(g_pool,cap_src,cap_size):
    """world
    Creates a window, gl context.
    Grabs images from a capture.
    Receives Pupil coordinates from g_pool.pupil_queue
    Can run various plug-ins.
    """


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
        g_pool.quit.value = True
        logger.info('Process closing from window')



    # load session persistent settings
    session_settings = shelve.open(os.path.join(g_pool.user_dir,'user_settings_world'),protocol=2)
    def load(var_name,default):
        return session_settings.get(var_name,default)
    def save(var_name,var):
        session_settings[var_name] = var


    # Initialize capture, check if it works
    cap = autoCreateCapture(cap_src, cap_size, 24)
    if cap is None:
        logger.error("Did not receive valid Capture")
        return
    frame = cap.get_frame()
    if frame.img is None:
        logger.error("Could not retrieve image from capture")
        cap.close()
        return
    height,width = frame.img.shape[:2]


    # helpers called by the main atb bar
    def update_fps():
        old_time, bar.timestamp = bar.timestamp, time()
        dt = bar.timestamp - old_time
        if dt:
            bar.fps.value += .05 * (1. / dt - bar.fps.value)

    def set_window_size(mode,data):
        height,width = frame.img.shape[:2]
        ratio = (1,.75,.5,.25)[mode]
        w,h = int(width*ratio),int(height*ratio)
        glfwSetWindowSize(world_window,w,h)
        data.value=mode # update the bar.value

    def get_from_data(data):
        """
        helper for atb getter and setter use
        """
        return data.value

    def open_calibration(selection,data):
        # prepare destruction of current ref_detector... and remove it
        for p in g.plugins:
            if isinstance(p,calibration_routines.detector_by_index):
                p.alive = False
        g.plugins = [p for p in g.plugins if p.alive]

        new_ref_detector = calibration_routines.detector_by_index[selection](g_pool,atb_pos=bar.next_atb_pos)
        g.plugins.append(new_ref_detector)
        g.plugins.sort(key=lambda p: p.order)

        # save the value for atb bar
        data.value=selection

    def toggle_record_video():
        for p in g.plugins:
            if isinstance(p,recorder.Recorder):
                p.alive = False
                return
        # set up folder within recordings named by user input in atb
        if not bar.rec_name.value:
            bar.rec_name.value = recorder.get_auto_name()

        new_plugin = recorder.Recorder(g_pool,bar.rec_name.value, bar.fps.value, frame.img.shape, bar.record_eye.value, g_pool.eye_tx)
        g.plugins.append(new_plugin)
        g.plugins.sort(key=lambda p: p.order)

    def toggle_show_calib_result():
        for p in g.plugins:
            if isinstance(p,Show_Calibration):
                p.alive = False
                return

        new_plugin = Show_Calibration(g_pool,frame.img.shape)
        g.plugins.append(new_plugin)
        g.plugins.sort(key=lambda p: p.order)

    def toggle_server():
        for p in g.plugins:
            if isinstance(p,Pupil_Server):
                p.alive = False
                return

        new_plugin = Pupil_Server(g_pool,(10,300))
        g.plugins.append(new_plugin)
        g.plugins.sort(key=lambda p: p.order)


    def toggle_ar():
        for p in g.plugins:
            if isinstance(p,Marker_Detector):
                p.alive = False
                return

        new_plugin = Marker_Detector(g_pool,(10,400))
        g.plugins.append(new_plugin)
        g.plugins.sort(key=lambda p: p.order)

    atb.init()
    # add main controls ATB bar
    bar = atb.Bar(name = "World", label="Controls",
            help="Scene controls", color=(50, 50, 50), alpha=100,valueswidth=150,
            text='light', position=(10, 10),refresh=.3, size=(300, 200))
    bar.next_atb_pos = (10,220)
    bar.fps = c_float(0.0)
    bar.timestamp = time()
    bar.calibration_type = c_int(load("calibration_type",0))
    bar.record_eye = c_bool(load("record_eye",0))
    bar.window_size = c_int(load("window_size",0))
    window_size_enum = atb.enum("Display Size",{"Full":0, "Medium":1,"Half":2,"Mini":3})
    calibrate_type_enum = atb.enum("Calibration Method",calibration_routines.index_by_name)
    bar.rec_name = create_string_buffer(512)
    bar.version = create_string_buffer(g_pool.version,512)
    bar.rec_name.value = recorder.get_auto_name()
    bar.add_var("fps", bar.fps, step=1., readonly=True)
    bar.add_var("display size", vtype=window_size_enum,setter=set_window_size,getter=get_from_data,data=bar.window_size)
    bar.add_var("calibration method",setter=open_calibration,getter=get_from_data,data=bar.calibration_type, vtype=calibrate_type_enum,group="Calibration", help="Please choose your desired calibration method.")
    bar.add_button("show calibration result",toggle_show_calib_result, group="Calibration", help="Click to show calibration result.")
    bar.add_var("session name",bar.rec_name, group="Recording", help="creates folder Data_Name_XXX, where xxx is an increasing number")
    bar.add_button("record", toggle_record_video, key="r", group="Recording", help="Start/Stop Recording")
    bar.add_var("record eye", bar.record_eye, group="Recording", help="check to save raw video of eye")
    bar.add_button("start/stop marker tracking",toggle_ar,key="x",help="find markers in scene to map gaze onto referace surfaces")
    bar.add_button("start/stop server",toggle_server,key="s",help="the server broadcasts pupil and gaze positions locally or via network")
    bar.add_separator("Sep1")
    bar.add_var("version",bar.version, readonly=True)
    bar.add_var("exit", g_pool.quit)

    # add uvc camera controls ATB bar
    cap.create_atb_bar(pos=(320,10))

    # Initialize glfw
    glfwInit()
    world_window = glfwCreateWindow(width, height, "World", None, None)
    glfwMakeContextCurrent(world_window)

    # Register callbacks world_window
    glfwSetWindowSizeCallback(world_window,on_resize)
    glfwSetWindowCloseCallback(world_window,on_close)
    glfwSetKeyCallback(world_window,on_key)
    glfwSetCharCallback(world_window,on_char)
    glfwSetMouseButtonCallback(world_window,on_button)
    glfwSetCursorPosCallback(world_window,on_pos)
    glfwSetScrollCallback(world_window,on_scroll)

    #set the last saved window size
    set_window_size(bar.window_size.value,bar.window_size)
    on_resize(world_window, *glfwGetFramebufferSize(world_window))
    glfwSetWindowPos(world_window,0,0)

    # gl_state settings
    basic_gl_setup()

    # refresh speed settings
    glfwSwapInterval(0)

    # load last calibration data
    try:
        pt_cloud = np.load(os.path.join(g_pool.user_dir,'cal_pt_cloud.npy'))
        logger.info("Using calibration found in %s" %g_pool.user_dir)
        map_pupil = calibrate.get_map_from_cloud(pt_cloud,(width,height))
    except:
        logger.info("No calibration found.")
        def map_pupil(vector):
            """ 1 to 1 mapping
            """
            return vector

    # create container for globally scoped vars (within world)
    g = Temp()
    g.plugins = []
    g_pool.map_pupil = map_pupil

    #load calibration plugin
    open_calibration(bar.calibration_type.value,bar.calibration_type)

    #load gaze_display plugin
    g.plugins.append(Display_Recent_Gaze(g_pool))

    # Event loop
    while not g_pool.quit.value:

        # Get an image from the grabber
        frame = cap.get_frame()
        update_fps()


        #a container that allows plugins to post and read events
        events = []

        #receive and map pupil positions
        recent_pupil_positions = []
        while not g_pool.pupil_queue.empty():
            p = g_pool.pupil_queue.get()
            if p['norm_pupil'] is None:
                p['norm_gaze'] = None
            else:
                p['norm_gaze'] = g_pool.map_pupil(p['norm_pupil'])
            recent_pupil_positions.append(p)


        # allow each Plugin to do its work.
        for p in g.plugins:
            p.update(frame,recent_pupil_positions,events)

        #check if a plugin need to be destroyed
        g.plugins = [p for p in g.plugins if p.alive]

        # render camera image
        glfwMakeContextCurrent(world_window)
        draw_gl_texture(frame.img)

        # render visual feedback from loaded plugins
        for p in g.plugins:
            p.gl_display()

        atb.draw()
        glfwSwapBuffers(world_window)
        glfwPollEvents()


    # de-init all running plugins
    for p in g.plugins:
        p.alive = False
        #reading p.alive actually runs plug-in cleanup
        _ = p.alive

    save('window_size',bar.window_size.value)
    save('calibration_type',bar.calibration_type.value)
    save('record_eye',bar.record_eye.value)
    session_settings.close()

    cap.close()
    glfwDestroyWindow(world_window)
    glfwTerminate()
    logger.debug("Process done")

def world_profiled(g_pool,cap_src,cap_size):
    import cProfile,subprocess,os
    from world import world
    cProfile.runctx("world(g_pool,cap_src,cap_size)",{"g_pool":g_pool,'cap_src':cap_src,'cap_size':cap_size},locals(),"world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print "created cpu time graph for world process. Please check out the png next to the world.py file"