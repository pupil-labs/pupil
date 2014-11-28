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
from file_methods import Persistent_Dict
import logging
from ctypes import  c_int,c_bool,c_float,create_string_buffer
import numpy as np

#display
from glfw import *
from pyglui import ui,graph

#monitoring
import psutil

# helpers/utils
from methods import normalize, denormalize,Temp
from gl_utils import basic_gl_setup,adjust_gl_view, clear_gl_screen, draw_gl_point_norm,make_coord_system_pixel_based,make_coord_system_norm_based,create_named_texture,draw_named_texture
from uvc_capture import autoCreateCapture, FileCaptureError, EndofVideoFileError, CameraCaptureError, FakeCapture
from audio import Audio_Input_List
# Plug-ins
from calibration_routines import calibration_plugins, gaze_mapping_plugins
from recorder import Recorder
from show_calibration import Show_Calibration
from display_recent_gaze import Display_Recent_Gaze
from pupil_server import Pupil_Server
from pupil_remote import Pupil_Remote
from marker_detector import Marker_Detector

plugin_by_index =  [Recorder,Show_Calibration, Display_Recent_Gaze,Pupil_Server,Pupil_Remote,Marker_Detector]+calibration_plugins+gaze_mapping_plugins
name_by_index = [p.__name__ for p in plugin_by_index]
plugin_by_name = dict(zip(name_by_index,plugin_by_index))
default_plugins = [('Dummy_Gaze_Mapper',{}),('Display_Recent_Gaze',{}), ('Screen_Marker_Calibration',{}),('Recorder',{}),('Pupil_Server',{})]

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
        hdpi_factor = glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0]
        w,h = w*hdpi_factor, h*hdpi_factor
        g_pool.gui.update_window(w,h)
        graph.adjust_size(w,h)
        adjust_gl_view(w,h)
        for p in g_pool.plugins:
            p.on_window_resize(window,w,h)

        glfwMakeContextCurrent(active_window)


    def on_iconify(window,iconfied):
        if not isinstance(cap,FakeCapture):
            g_pool.update_textures = not iconfied

    def on_key(window, key, scancode, action, mods):
        g_pool.gui.update_key(key,scancode,action,mods)
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                on_close(window)

    def on_char(window,char):
        g_pool.gui.update_char(char)


    def on_button(window,button, action, mods):
        g_pool.gui.update_button(button,action,mods)
        pos = glfwGetCursorPos(window)
        pos = normalize(pos,glfwGetWindowSize(world_window))
        pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
        for p in g_pool.plugins:
            p.on_click(pos,button,action)

    def on_pos(window,x, y):
        hdpi_factor = float(glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0])
        x,y = x*hdpi_factor,y*hdpi_factor
        g_pool.gui.update_mouse(x,y)

    def on_scroll(window,x,y):
        g_pool.gui.update_scroll(x,y)


    def on_close(window):
        g_pool.quit.value = True
        logger.info('Process closing from window')



    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_world'))


    # Initialize capture
    cap = autoCreateCapture(cap_src, cap_size, 24, timebase=g_pool.timebase)

     # Get an image from the grabber
    try:
        frame = cap.get_frame()
    except CameraCaptureError:
        logger.error("Could not retrieve image from capture")
        cap.close()
        return


    # any object we attach to the g_pool object *from now on* will only be visible to this process!
    # vars should be declared here to make them visible to the code reader.
    g_pool.plugins = []
    g_pool.update_textures = True

    if isinstance(cap,FakeCapture):
        g_pool.update_textures = False
    g_pool.capture = cap
    g_pool.pupil_confidence_threshold = session_settings.get('pupil_confidence_threshold',.6)
    g_pool.window_size = session_settings.get('window_size',1.)



    #load Plugins
    #plugins that are loaded based on user settings
    for initializer in session_settings.get('loaded_plugins',default_plugins):
        name, args = initializer
        logger.debug("Loading plugin: %s with settings %s"%(name, args))
        try:
            p = plugin_by_name[name](g_pool,**args)
            g_pool.plugins.append(p)
        except IOError:
            logger.warning("Plugin '%s' failed to load from settings file." %name)


    #only need for the gui to show the loaded calibration type
    for p in g_pool.plugins:
        if p.base_class_name == 'Calibration_Plugin':
            g_pool.active_calibration_plugin =  p.__class__
            break




    #UI and other callback functions
    def set_window_size(size):
        hdpi_factor = glfwGetFramebufferSize(world_window)[0]/glfwGetWindowSize(world_window)[0]
        w,h = int(frame.width*size*hdpi_factor),int(frame.height*size*hdpi_factor)
        glfwSetWindowSize(world_window,w,h)


    def reset_timebase():
        #the last frame from worldcam will be t0
        g_pool.timebase.value = g_pool.capure.get_now()
        logger.info("New timebase set to %s all timestamps will count from here now."%g_pool.timebase.value)

    def set_calibration_plugin(new_calibration):
        g_pool.active_calibration_plugin = new_calibration
        # prepare destruction of current calibration plugin... and remove it
        for p in g_pool.plugins:
            if p.base_class_name == 'Calibration_Plugin':
                p.alive = False
        g_pool.plugins = [p for p in g_pool.plugins if p.alive]

        #add new plugin
        new = new_calibration(g_pool)
        new.init_gui()
        g_pool.plugins.append(new)
        g_pool.plugins.sort(key=lambda p: p.order)

    def set_scale(new_scale):
        g_pool.gui.scale = new_scale

    def get_scale():
        return g_pool.gui.scale



    # Initialize glfw
    glfwInit()
    world_window = glfwCreateWindow(frame.width, frame.height, "World", None, None)
    glfwMakeContextCurrent(world_window)

    # Register callbacks world_window
    glfwSetWindowSizeCallback(world_window,on_resize)
    glfwSetWindowCloseCallback(world_window,on_close)
    glfwSetWindowIconifyCallback(world_window,on_iconify)
    glfwSetKeyCallback(world_window,on_key)
    glfwSetCharCallback(world_window,on_char)
    glfwSetMouseButtonCallback(world_window,on_button)
    glfwSetCursorPosCallback(world_window,on_pos)
    glfwSetScrollCallback(world_window,on_scroll)

    # gl_state settings
    basic_gl_setup()
    g_pool.image_tex = create_named_texture(frame.img)
    # refresh speed settings
    glfwSwapInterval(0)

    glfwSetWindowPos(world_window,0,0)



    #setup GUI
    g_pool.gui = ui.UI()
    g_pool.gui.scale = session_settings.get('gui_scale',1)
    g_pool.sidebar = ui.Scrolling_Menu("Settings",pos=(-250,0),size=(0,0),header_pos='left')

    general_settings = ui.Growing_Menu('General')
    general_settings.append(ui.Slider('scale', setter= set_scale,getter=get_scale,step = .1,min=.5,max=2,label='Interface Size'))
    general_settings.append(ui.Switch('update_textures',g_pool,label="Update Display"))
    general_settings.append(ui.Selector('active_calibration_plugin',g_pool, selection = calibration_plugins,
                                        labels = [p.__name__ for p in calibration_plugins],
                                        setter=set_calibration_plugin,label='Calibration Type'))

    g_pool.sidebar.append(general_settings)

    g_pool.quickbar = ui.Stretching_Menu('Quick Bar',(0,100),(120,-100))

    g_pool.gui.append(g_pool.quickbar)
    g_pool.gui.append(g_pool.sidebar)


    #set the last saved window size
    set_window_size(g_pool.window_size)
    on_resize(world_window, *glfwGetWindowSize(world_window))



    # setup GUI for plugins.
    for p in g_pool.plugins:
        p.init_gui()


    #set up performace graphs:
    pid = os.getpid()
    ps = psutil.Process(pid)
    ts = time()

    cpu_g = graph.Graph()
    cpu_g.pos = (20,100)
    cpu_g.update_fn = ps.get_cpu_percent
    cpu_g.update_rate = 5
    cpu_g.label = 'CPU %0.1f'

    fps_g = graph.Graph()
    fps_g.pos = (140,100)
    fps_g.update_rate = 5
    fps_g.label = "%0.0f FPS"

    pupil_g = graph.Graph(max_val=1.2)
    pupil_g.pos = (260,100)
    pupil_g.update_rate = 5
    pupil_g.label = "Confidence: %0.2f"

    # Event loop
    while not g_pool.quit.value:

        # Get an image from the grabber
        try:
            frame = cap.get_frame()
        except CameraCaptureError:
            logger.error("Capture from Camera Failed. Stopping.")
            break
        except EndofVideoFileError:
            logger.warning("Video File is done. Stopping")
            break

        #update performace graphs
        t = time()
        dt,ts = t-ts,t
        fps_g.add(1./dt)
        cpu_g.update()


        #a dictionary that allows plugins to post and read events
        events = {}

        #receive and map pupil positions
        recent_pupil_positions = []
        while not g_pool.pupil_queue.empty():
            p = g_pool.pupil_queue.get()
            recent_pupil_positions.append(p)
            pupil_g.add(p['confidence'])

        # allow each Plugin to do its work.
        for p in g_pool.plugins:
            p.update(frame,recent_pupil_positions,events)

        #check if a plugin need to be destroyed
        g_pool.plugins = [p for p in g_pool.plugins if p.alive]

        # render camera image
        glfwMakeContextCurrent(world_window)

        make_coord_system_norm_based()
        if g_pool.update_textures:
            draw_named_texture(g_pool.image_tex,frame.img)
        else:
            draw_named_texture(g_pool.image_tex)

        make_coord_system_pixel_based((frame.height,frame.width,3))

        # render visual feedback from loaded plugins
        for p in g_pool.plugins:
            p.gl_display()

        fps_g.draw()
        cpu_g.draw()
        pupil_g.draw()
        g_pool.gui.update()
        glfwSwapBuffers(world_window)
        glfwPollEvents()


    loaded_plugins = []
    for p in g_pool.plugins:
        try:
            p_initializer = p.class_name,p.get_init_dict()
            loaded_plugins.append(p_initializer)
        except AttributeError:
            #not all plugins want to be savable, they will not have the init dict.
            # any object without a get_init_dict method will throw this exception.
            pass


    # de-init all running plugins
    for p in g_pool.plugins:
        p.alive = False
        #reading p.alive actually runs plug-in cleanup
        _ = p.alive

    session_settings['loaded_plugins'] = loaded_plugins
    session_settings['window_size'] = g_pool.window_size
    session_settings['pupil_confidence_threshold'] = g_pool.pupil_confidence_threshold
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