'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

# make shared modules available across pupil_src
if __name__ == '__main__':
    from sys import path as syspath
    from os import path as ospath
    loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
    syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
    del syspath, ospath

import os, sys
from time import time
import shelve
from ctypes import  c_int,c_bool,c_float,create_string_buffer
import numpy as np

#display
from glfw import *
import atb

# helpers/utils
from methods import normalize, denormalize,Temp
from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen, draw_gl_point_norm
from uvc_capture import autoCreateCapture
import calibrate
# Plugins
import reference_detectors
import recorder
from show_calibration import Show_Calibration


def world(g_pool):
    """world
    """


    # Callback functions
    def on_resize_world(window,w, h):
        glfwMakeContextCurrent(window)
        adjust_gl_view(w,h)
        atb.TwWindowSize(w, h)

    def on_resize_player(window,w, h):
        glfwMakeContextCurrent(window)
        adjust_gl_view(w,h)

    def on_key(window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    on_close(window)

    def on_char(window,char):
        if not atb.TwEventCharGLFW(char,1):
            pass

    def on_button(window,button, action, mods):
        if not atb.TwEventMouseButtonGLFW(button,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                pos = glfwGetCursorPos(window)
                pos = normalize(pos,glfwGetWindowSize(world_window))
                pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
                for p in g.plugins:
                    p.on_click(pos)

    def on_pos(window,x, y):
        if atb.TwMouseMotion(int(x),int(y)):
            pass

    def on_scroll(window,x,y):
        if not atb.TwMouseWheel(int(x)):
            pass

    def on_close(window):
        g_pool.quit.value = True
        print "WORLD Process closing from window"


    # load session persistent settings
    session_settings = shelve.open('user_settings_world',protocol=2)
    def load(var_name,default):
        try:
            return session_settings[var_name]
        except:
            return default
    def save(var_name,var):
        session_settings[var_name] = var



    # Initialize capture, check if it works
    cap = autoCreateCapture(g_pool.world_src, g_pool.world_size,24)
    if cap is None:
        print "WORLD: Error could not create Capture"
        return
    frame = cap.get_frame()
    if frame.img is None:
        print "WORLD: Error could not get image"
        return
    height,width = frame.img.shape[:2]


    # helpers called by the main atb bar
    def update_fps():
        old_time, bar.timestamp = bar.timestamp, time()
        dt = bar.timestamp - old_time
        if dt:
            bar.fps.value += .05 * (1 / dt - bar.fps.value)

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
        # prepare destruction of old ref_detector.
        if g.current_ref_detector:
            g.current_ref_detector.alive = False

        # remove old ref detector from list of plugins
        g.plugins = [p for p in g.plugins if p.alive]

        print "selected: ",reference_detectors.name_by_index[selection]
        g.current_ref_detector = reference_detectors.detector_by_index[selection](g_pool,atb_pos=bar.next_atb_pos)
        g.plugins.append(g.current_ref_detector)
        # save the value for atb bar
        data.value=selection

    def toggle_record_video():
        if any([True for p in g.plugins if isinstance(p,recorder.Recorder)]):
            for p in g.plugins:
                if isinstance(p,recorder.Recorder):
                    p.alive = False
        else:
            # set up folder within recordings named by user input in atb
            if not bar.rec_name.value:
                bar.rec_name.value = recorder.get_auto_name()
            recorder_instance = recorder.Recorder(bar.rec_name.value, bar.fps.value, frame.img.shape, bar.record_eye.value, g_pool.eye_tx)
            g.plugins.append(recorder_instance)

    def toggle_show_calib_result():
        if any([True for p in g.plugins if isinstance(p,Show_Calibration)]):
            for p in g.plugins:
                if isinstance(p,Show_Calibration):
                    p.alive = False
        else:
            calib = Show_Calibration(frame.img.shape)
            g.plugins.append(calib)

    def show_calib_result():
        # kill old if any
        if any([True for p in g.plugins if isinstance(p,Show_Calibration)]):
            for p in g.plugins:
                if isinstance(p,Show_Calibration):
                    p.alive = False
            g.plugins = [p for p in g.plugins if p.alive]
        # make new
        calib = Show_Calibration(frame.img.shape)
        g.plugins.append(calib)

    def hide_calib_result():
        if any([True for p in g.plugins if isinstance(p,Show_Calibration)]):
            for p in g.plugins:
                if isinstance(p,Show_Calibration):
                    p.alive = False

    # Initialize ant tweak bar
    atb.init()
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

    bar.calibrate_type_enum = atb.enum("Calibration Method",reference_detectors.index_by_name)
    bar.rec_name = create_string_buffer(512)
    bar.rec_name.value = recorder.get_auto_name()
    # play and record can be tied together via pointers to the objects
    bar.add_var("fps", bar.fps, step=1., readonly=True)
    bar.add_var("display size", vtype=window_size_enum,setter=set_window_size,getter=get_from_data,data=bar.window_size)
    bar.add_var("calibration method",setter=open_calibration,getter=get_from_data,data=bar.calibration_type, vtype=bar.calibrate_type_enum,group="Calibration", help="Please choose your desired calibration method.")
    bar.add_button("show calibration result",toggle_show_calib_result, group="Calibration", help="Click to show calibration result.")
    bar.add_var("session name",bar.rec_name, group="Recording", help="creates folder Data_Name_XXX, where xxx is an increasing number")
    bar.add_button("record", toggle_record_video, key="r", group="Recording", help="Start/Stop Recording")
    bar.add_var("record eye", bar.record_eye, group="Recording", help="check to save raw video of eye")
    bar.add_separator("Sep1")
    bar.add_var("exit", g_pool.quit)

    # add uvc camera controls to a seperate ATB bar
    cap.create_atb_bar(pos=(320,10))


    # create container for globally scoped vars (within world)
    g = Temp()
    g.plugins = []
    g.current_ref_detector = None


    try:
        pt_cloud = np.load('cal_pt_cloud.npy')
        map_pupil = calibrate.get_map_from_cloud(pt_cloud,(width,height))
    except:
        print "no calibration found"
        def map_pupil(vector):
            """ 1 to 1 mapping
            """
            return vector
    g_pool.map_pupil = map_pupil


    open_calibration(bar.calibration_type.value,bar.calibration_type)

    # Initialize glfw
    glfwInit()
    world_window = glfwCreateWindow(width, height, "World", None, None)
    glfwSetWindowPos(world_window,0,0)
    on_resize_world(world_window,width,height)
    #set the last saved window size
    set_window_size(bar.window_size.value,bar.window_size)

    player_window = glfwCreateWindow(640, 360, "Player", None, None)
    glfwSetWindowPos(player_window,20,20)
    on_resize_player(player_window,640,360)


    # Register callbacks world_window
    glfwSetWindowSizeCallback(world_window,on_resize_world)
    glfwSetWindowCloseCallback(world_window,on_close)
    glfwSetKeyCallback(world_window,on_key)
    glfwSetCharCallback(world_window,on_char)
    glfwSetMouseButtonCallback(world_window,on_button)
    glfwSetCursorPosCallback(world_window,on_pos)
    glfwSetScrollCallback(world_window,on_scroll)
    #Register cllbacks player_window
    glfwSetWindowSizeCallback(player_window,on_resize_player)
    glfwSetWindowCloseCallback(player_window,on_close)
    glfwSetKeyCallback(player_window,on_key)
    glfwSetCharCallback(player_window,on_char)

    # gl_state settings
    import OpenGL.GL as gl
    for context in (player_window,world_window):
        glfwMakeContextCurrent(context)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_BLEND)
        gl.glClearColor(1.,1.,1.,0.)

    del gl

    # Event loop
    while not glfwWindowShouldClose(world_window) and not glfwWindowShouldClose(player_window) and not g_pool.quit.value:

        # Get an image from the grabber
        frame = cap.get_frame()
        update_fps()

        recent_pupil_positions = []
        while not g_pool.pupil_queue.empty():
            p = g_pool.pupil_queue.get()
            if p['norm_pupil'] is None:
                p['norm_gaze'] = None
            else:
                p['norm_gaze'] = g_pool.map_pupil(p['norm_pupil'])
            recent_pupil_positions.append(p)

        for p in g.plugins:
            p.update(frame,recent_pupil_positions)

        g.plugins = [p for p in g.plugins if p.alive]



        glfwMakeContextCurrent(player_window)
        clear_gl_screen()

        # render visual feedback from loaded plugins in player window
        for p in g.plugins:
            p.gl_display_player_window()


        # render camera image in world_window
        glfwMakeContextCurrent(world_window)
        clear_gl_screen()
        draw_gl_texture(frame.img)

        # render visual feedback from loaded plugins in world window
        for p in g.plugins:
            p.gl_display()


        # update gaze point from shared variable pool and draw on world_window.
        for pt in recent_pupil_positions:
            if pt['norm_gaze'] is not None:
                draw_gl_point_norm(pt['norm_gaze'],color=(1.,0.,0.,0.5))

        atb.draw()
        glfwSwapBuffers(player_window)
        glfwSwapBuffers(world_window)
        glfwPollEvents()

    # end while running and clean-up

    # de-init all running plugins
    for p in g.plugins:
        p.alive = False
    g.plugins = [p for p in g.plugins if p.alive]

    save('window_size',bar.window_size.value)
    save('calibration_type',bar.calibration_type.value)
    save('record_eye',bar.record_eye.value)
    session_settings.close()

    cap.close()
    glfwDestroyWindow(player_window)
    glfwDestroyWindow(world_window)
    glfwTerminate()
    print "WORLD Process closed"

def world_profiled(g_pool):
    import cProfile,subprocess,os
    from world import world
    cProfile.runctx("world(g_pool,)",{"g_pool":g_pool},locals(),"world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print "created cpu time graph for world process. Please check out the png next to the world.py file"


