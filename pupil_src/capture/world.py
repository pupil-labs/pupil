'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

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


import os, sys,platform
import logging
import numpy as np

#display
from glfw import *
from pyglui import ui,graph,cygl
from pyglui.cygl.utils import Named_Texture
from gl_utils import basic_gl_setup,adjust_gl_view, clear_gl_screen,make_coord_system_pixel_based,make_coord_system_norm_based

#check versions for our own depedencies as they are fast-changing
from pyglui import __version__ as pyglui_version
assert pyglui_version >= '0.6'

#monitoring
import psutil

# helpers/utils
from file_methods import Persistent_Dict
from version_utils import VersionFormat
from methods import normalize, denormalize, delta_t
from video_capture import autoCreateCapture, FileCaptureError, EndofVideoFileError, CameraCaptureError


# Plug-ins
from plugin import Plugin_List,import_runtime_plugins
from calibration_routines import calibration_plugins, gaze_mapping_plugins
from recorder import Recorder
from show_calibration import Show_Calibration
from display_recent_gaze import Display_Recent_Gaze
from pupil_server import Pupil_Server
from pupil_sync import Pupil_Sync
from marker_detector import Marker_Detector
from log_display import Log_Display

# create logger for the context of this function
logger = logging.getLogger(__name__)



#UI Platform tweaks
if platform.system() == 'Linux':
    scroll_factor = 10.0
    window_position_default = (0,0)
elif platform.system() == 'Windows':
    scroll_factor = 1.0
    window_position_default = (8,31)
else:
    scroll_factor = 1.0
    window_position_default = (0,0)


def world(g_pool,cap_src,cap_size):
    """world
    Creates a window, gl context.
    Grabs images from a capture.
    Receives Pupil coordinates from eye process[es]
    Can run various plug-ins.
    """

    #manage plugins
    runtime_plugins = import_runtime_plugins(os.path.join(g_pool.user_dir,'plugins'))
    user_launchable_plugins = [Show_Calibration,Pupil_Server,Pupil_Sync,Marker_Detector]+runtime_plugins
    system_plugins  = [Log_Display,Display_Recent_Gaze,Recorder]
    plugin_by_index =  system_plugins+user_launchable_plugins+calibration_plugins+gaze_mapping_plugins
    name_by_index = [p.__name__ for p in plugin_by_index]
    plugin_by_name = dict(zip(name_by_index,plugin_by_index))
    default_plugins = [('Log_Display',{}),('Dummy_Gaze_Mapper',{}),('Display_Recent_Gaze',{}), ('Screen_Marker_Calibration',{}),('Recorder',{})]



    # Callback functions
    def on_resize(window,w, h):
        if not g_pool.iconified:
            g_pool.gui.update_window(w,h)
            g_pool.gui.collect_menus()
            graph.adjust_size(w,h)
            adjust_gl_view(w,h)
            for p in g_pool.plugins:
                p.on_window_resize(window,w,h)

    def on_iconify(window,iconified):
        g_pool.iconified = iconified

    def on_key(window, key, scancode, action, mods):
        g_pool.gui.update_key(key,scancode,action,mods)

    def on_char(window,char):
        g_pool.gui.update_char(char)

    def on_button(window,button, action, mods):
        g_pool.gui.update_button(button,action,mods)
        pos = glfwGetCursorPos(window)
        pos = normalize(pos,glfwGetWindowSize(main_window))
        pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
        for p in g_pool.plugins:
            p.on_click(pos,button,action)

    def on_pos(window,x, y):
        hdpi_factor = float(glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0])
        x,y = x*hdpi_factor,y*hdpi_factor
        g_pool.gui.update_mouse(x,y)

    def on_scroll(window,x,y):
        g_pool.gui.update_scroll(x,y*scroll_factor)


    def on_close(window):
        g_pool.quit.value = True
        logger.info('Process closing from window')


    tick = delta_t()
    def get_dt():
        return next(tick)

    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_world'))
    if session_settings.get("version",VersionFormat('0.0')) < g_pool.version:
        logger.info("Session setting are from older version of this app. I will not use those.")
        session_settings.clear()

    # Initialize capture
    cap = autoCreateCapture(cap_src, timebase=g_pool.timebase)
    default_settings = {'frame_size':cap_size,'frame_rate':24}
    previous_settings = session_settings.get('capture_settings',None)
    if previous_settings and previous_settings['name'] == cap.name:
        cap.settings = previous_settings
    else:
        cap.settings = default_settings

    # Test capture
    try:
        frame = cap.get_frame()
    except CameraCaptureError:
        logger.error("Could not retrieve image from capture")
        cap.close()
        return


    # any object we attach to the g_pool object *from now on* will only be visible to this process!
    # vars should be declared here to make them visible to the code reader.
    g_pool.iconified = False

    g_pool.capture = cap
    g_pool.pupil_confidence_threshold = session_settings.get('pupil_confidence_threshold',.6)
    g_pool.active_calibration_plugin = None


    def open_plugin(plugin):
        if plugin ==  "Select to load":
            return
        g_pool.plugins.add(plugin)

    def set_scale(new_scale):
        g_pool.gui.scale = new_scale
        g_pool.gui.collect_menus()


    #window and gl setup
    glfwInit()
    width,height = session_settings.get('window_size',(frame.width, frame.height))
    main_window = glfwCreateWindow(width,height, "World")
    window_pos = session_settings.get('window_position',window_position_default)
    glfwSetWindowPos(main_window,window_pos[0],window_pos[1])
    glfwMakeContextCurrent(main_window)
    cygl.utils.init()

    #setup GUI
    g_pool.gui = ui.UI()
    g_pool.gui.scale = session_settings.get('gui_scale',1)
    g_pool.sidebar = ui.Scrolling_Menu("Settings",pos=(-350,0),size=(0,0),header_pos='left')
    general_settings = ui.Growing_Menu('General')
    general_settings.append(ui.Slider('scale',g_pool.gui, setter=set_scale,step = .05,min=1.,max=2.5,label='Interface size'))
    general_settings.append(ui.Button('Reset window size',lambda: glfwSetWindowSize(main_window,frame.width,frame.height)) )
    general_settings.append(ui.Selector('Open plugin', selection = user_launchable_plugins,
                                        labels = [p.__name__.replace('_',' ') for p in user_launchable_plugins],
                                        setter= open_plugin, getter=lambda: "Select to load"))
    general_settings.append(ui.Slider('pupil_confidence_threshold', g_pool,step = .01,min=0.,max=1.,label='Minimum pupil confidence'))
    general_settings.append(ui.Info_Text('Capture Version: %s'%g_pool.version))
    g_pool.sidebar.append(general_settings)

    g_pool.calibration_menu = ui.Growing_Menu('Calibration')
    g_pool.sidebar.append(g_pool.calibration_menu)
    g_pool.gui.append(g_pool.sidebar)
    g_pool.quickbar = ui.Stretching_Menu('Quick Bar',(0,100),(120,-100))
    g_pool.gui.append(g_pool.quickbar)
    g_pool.gui.append(ui.Hot_Key("quit",setter=on_close,getter=lambda:True,label="X",hotkey=GLFW_KEY_ESCAPE))
    g_pool.capture.init_gui(g_pool.sidebar)

    #plugins that are loaded based on user settings from previous session
    g_pool.notifications = []
    g_pool.plugins = Plugin_List(g_pool,plugin_by_name,session_settings.get('loaded_plugins',default_plugins))

    #We add the calibration menu selector, after a calibration has been added:
    g_pool.calibration_menu.insert(0,ui.Selector('active_calibration_plugin',getter=lambda: g_pool.active_calibration_plugin.__class__, selection = calibration_plugins,
                                        labels = [p.__name__.replace('_',' ') for p in calibration_plugins],
                                        setter= open_plugin,label='Method'))

    # Register callbacks main_window
    glfwSetFramebufferSizeCallback(main_window,on_resize)
    glfwSetWindowCloseCallback(main_window,on_close)
    glfwSetWindowIconifyCallback(main_window,on_iconify)
    glfwSetKeyCallback(main_window,on_key)
    glfwSetCharCallback(main_window,on_char)
    glfwSetMouseButtonCallback(main_window,on_button)
    glfwSetCursorPosCallback(main_window,on_pos)
    glfwSetScrollCallback(main_window,on_scroll)

    # gl_state settings
    basic_gl_setup()
    g_pool.image_tex = Named_Texture()
    g_pool.image_tex.update_from_frame(frame)

    # refresh speed settings
    glfwSwapInterval(0)

    #trigger setup of window and gl sizes
    on_resize(main_window, *glfwGetFramebufferSize(main_window))

    #now the we have  aproper window we can load the last gui configuration
    g_pool.gui.configuration = session_settings.get('ui_config',{})



    #set up performace graphs:
    pid = os.getpid()
    ps = psutil.Process(pid)
    ts = frame.timestamp

    cpu_graph = graph.Bar_Graph()
    cpu_graph.pos = (20,130)
    cpu_graph.update_fn = ps.cpu_percent
    cpu_graph.update_rate = 5
    cpu_graph.label = 'CPU %0.1f'

    fps_graph = graph.Bar_Graph()
    fps_graph.pos = (140,130)
    fps_graph.update_rate = 5
    fps_graph.label = "%0.0f FPS"

    pupil_graph = graph.Bar_Graph(max_val=1.0)
    pupil_graph.pos = (260,130)
    pupil_graph.update_rate = 5
    pupil_graph.label = "Confidence: %0.2f"

    # Event loop
    while not g_pool.quit.value:

        # Get an image from the grabber
        try:
            frame = cap.get_frame()
        except CameraCaptureError:
            logger.error("Capture from camera failed. Stopping.")
            break
        except EndofVideoFileError:
            logger.warning("Video file is done. Stopping")
            break

        #update performace graphs
        t = frame.timestamp
        dt,ts = t-ts,t
        try:
            fps_graph.add(1./dt)
        except ZeroDivisionError:
            pass
        cpu_graph.update()


        #a dictionary that allows plugins to post and read events
        events = {}

        #report time between now and the last loop interation
        events['dt'] = get_dt()

        #receive and map pupil positions
        recent_pupil_positions = []
        while not g_pool.pupil_queue.empty():
            p = g_pool.pupil_queue.get()
            recent_pupil_positions.append(p)
            pupil_graph.add(p['confidence'])
        events['pupil_positions'] = recent_pupil_positions

        # notify each plugin if there are new notifications:
        while g_pool.notifications:
            n = g_pool.notifications.pop(0)
            for p in g_pool.plugins:
                p.on_notify(n)

        # allow each Plugin to do its work.
        for p in g_pool.plugins:
            p.update(frame,events)

        #check if a plugin need to be destroyed
        g_pool.plugins.clean()

        # render camera image
        glfwMakeContextCurrent(main_window)
        if g_pool.iconified:
            pass
        else:
            g_pool.image_tex.update_from_frame(frame)

        make_coord_system_norm_based()
        g_pool.image_tex.draw()
        make_coord_system_pixel_based((frame.height,frame.width,3))
        # render visual feedback from loaded plugins
        for p in g_pool.plugins:
            p.gl_display()

        if not g_pool.iconified:
            graph.push_view()
            fps_graph.draw()
            cpu_graph.draw()
            pupil_graph.draw()
            graph.pop_view()
            g_pool.gui.update()
            glfwSwapBuffers(main_window)
        glfwPollEvents()

    glfwRestoreWindow(main_window) #need to do this for windows os
    session_settings['loaded_plugins'] = g_pool.plugins.get_initializers()
    session_settings['pupil_confidence_threshold'] = g_pool.pupil_confidence_threshold
    session_settings['gui_scale'] = g_pool.gui.scale
    session_settings['ui_config'] = g_pool.gui.configuration
    session_settings['capture_settings'] = g_pool.capture.settings
    session_settings['window_size'] = glfwGetWindowSize(main_window)
    session_settings['window_position'] = glfwGetWindowPos(main_window)
    session_settings['version'] = g_pool.version
    session_settings.close()

    # de-init all running plugins
    for p in g_pool.plugins:
        p.alive = False
    g_pool.plugins.clean()
    g_pool.gui.terminate()
    glfwDestroyWindow(main_window)
    glfwTerminate()
    cap.close()

    logger.debug("Process done")

def world_profiled(g_pool,cap_src,cap_size):
    import cProfile,subprocess,os
    from world import world
    cProfile.runctx("world(g_pool,cap_src,cap_size)",{"g_pool":g_pool,'cap_src':cap_src,'cap_size':cap_size},locals(),"world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print "created cpu time graph for world process. Please check out the png next to the world.py file"
