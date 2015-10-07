'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os,platform
from glob import glob
from copy import deepcopy
try:
    from billiard import freeze_support
except:
    from multiprocessing import freeze_support

if getattr(sys, 'frozen', False):
    user_dir = os.path.expanduser(os.path.join('~','pupil_player_settings'))
    version_file = os.path.join(sys._MEIPASS,'_version_string_')
else:
    # We are running in a normal Python environment.
    # Make all pupil shared_modules available to this Python session.
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))
    # Specifiy user dirs.
    user_dir = os.path.join(pupil_base_dir,'player_settings')
    version_file = None
    if __name__ == '__main__':
        #compile all cython source files
        from pyx_compiler import build_extensions
        build_extensions()


# create folder for user settings, tmp data
if not os.path.isdir(user_dir):
    os.mkdir(user_dir)


import logging
#set up root logger before other imports
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
#since we are not using OS.fork on MacOS we need to do a few extra things to log our exports correctly.
if platform.system() == 'Darwin':
    if __name__ == '__main__': #clear log if main
        fh = logging.FileHandler(os.path.join(user_dir,'player.log'),mode='w')
    #we will use append mode since the exporter will stream into the same file when using os.span processes
    fh = logging.FileHandler(os.path.join(user_dir,'player.log'),mode='a')
else:
    fh = logging.FileHandler(os.path.join(user_dir,'player.log'),mode='w')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('Player: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
formatter = logging.Formatter('Player [%(levelname)s] %(name)s : %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

logging.getLogger("OpenGL").setLevel(logging.ERROR)
logging.getLogger("libav").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


#Linux scrolling is scaled differently
if platform.system() == 'Linux':
    y_scroll_factor = 10.0
else:
    y_scroll_factor = 1.0

#imports
from file_methods import Persistent_Dict,load_object
import numpy as np

#display
from glfw import *
# check versions for our own depedencies as they are fast-changing
from pyglui import __version__ as pyglui_version
assert pyglui_version >= '0.3'
from pyglui import ui,graph,cygl
from pyglui.cygl.utils import Named_Texture
from gl_utils import basic_gl_setup,adjust_gl_view, clear_gl_screen,make_coord_system_pixel_based,make_coord_system_norm_based
from OpenGL.GL import glClearColor
#capture
from video_capture import File_Capture,EndofVideoFileError,FileSeekError

# helpers/utils
from version_utils import VersionFormat, read_rec_version, get_version
from methods import normalize, denormalize, delta_t
from player_methods import correlate_data, is_pupil_rec_dir,update_recording_0v4_to_current,update_recording_0v3_to_current

#monitoring
import psutil

# Plug-ins
from plugin import Plugin_List,import_runtime_plugins
from vis_circle import Vis_Circle
from vis_cross import Vis_Cross
from vis_polyline import Vis_Polyline
from vis_light_points import Vis_Light_Points
from vis_watermark import Vis_Watermark
from seek_bar import Seek_Bar
from trim_marks import Trim_Marks
from export_launcher import Export_Launcher
from scan_path import Scan_Path
from offline_marker_detector import Offline_Marker_Detector
from marker_auto_trim_marks import Marker_Auto_Trim_Marks
from pupil_server import Pupil_Server
from fixation_detector import Dispersion_Duration_Fixation_Detector
from manual_gaze_correction import Manual_Gaze_Correction
from show_calibration import Show_Calibration
from batch_exporter import Batch_Exporter
from eye_video_overlay import Eye_Video_Overlay
from log_display import Log_Display

system_plugins = [Log_Display,Seek_Bar,Trim_Marks]
user_launchable_plugins = [Export_Launcher, Vis_Circle,Vis_Cross, Vis_Polyline, Vis_Light_Points,Scan_Path,Dispersion_Duration_Fixation_Detector,Vis_Watermark, Manual_Gaze_Correction, Show_Calibration, Offline_Marker_Detector,Pupil_Server,Batch_Exporter,Eye_Video_Overlay] #,Marker_Auto_Trim_Marks
user_launchable_plugins += import_runtime_plugins(os.path.join(user_dir,'plugins'))
available_plugins = system_plugins + user_launchable_plugins
name_by_index = [p.__name__ for p in available_plugins]
index_by_name = dict(zip(name_by_index,range(len(name_by_index))))
plugin_by_name = dict(zip(name_by_index,available_plugins))

class Global_Container(object):
    pass

def session(rec_dir):

    # Callback functions
    def on_resize(window,w, h):
        g_pool.gui.update_window(w,h)
        g_pool.gui.collect_menus()
        graph.adjust_size(w,h)
        adjust_gl_view(w,h)
        for p in g_pool.plugins:
            p.on_window_resize(window,w,h)

    def on_key(window, key, scancode, action, mods):
        g_pool.gui.update_key(key,scancode,action,mods)

    def on_char(window,char):
        g_pool.gui.update_char(char)

    def on_button(window,button, action, mods):
        g_pool.gui.update_button(button,action,mods)
        pos = glfwGetCursorPos(window)
        pos = normalize(pos,glfwGetWindowSize(window))
        pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
        for p in g_pool.plugins:
            p.on_click(pos,button,action)

    def on_pos(window,x, y):
        hdpi_factor = float(glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0])
        g_pool.gui.update_mouse(x*hdpi_factor,y*hdpi_factor)

    def on_scroll(window,x,y):
        g_pool.gui.update_scroll(x,y*y_scroll_factor)


    def on_drop(window,count,paths):
        for x in range(count):
            new_rec_dir =  paths[x]
            if is_pupil_rec_dir(new_rec_dir):
                logger.debug("Starting new session with '%s'"%new_rec_dir)
                global rec_dir
                rec_dir = new_rec_dir
                glfwSetWindowShouldClose(window,True)
            else:
                logger.error("'%s' is not a valid pupil recording"%new_rec_dir)




    tick = delta_t()
    def get_dt():
        return next(tick)


    video_path = [f for f in glob(os.path.join(rec_dir,"world.*")) if f[-3:] in ('mp4','mkv','avi')][0]
    timestamps_path = os.path.join(rec_dir, "world_timestamps.npy")
    pupil_data_path = os.path.join(rec_dir, "pupil_data")

    #parse info.csv file
    meta_info_path = os.path.join(rec_dir,"info.csv")
    with open(meta_info_path) as info:
        meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )

    rec_version = read_rec_version(meta_info)
    if rec_version >= VersionFormat('0.5'):
        pass
    elif rec_version >= VersionFormat('0.4'):
        update_recording_0v4_to_current(rec_dir)
    elif rec_version >= VersionFormat('0.3'):
        update_recording_0v3_to_current(rec_dir)
        timestamps_path = os.path.join(rec_dir, "timestamps.npy")

    else:
        logger.Error("This recording is to old. Sorry.")
        return

    timestamps = np.load(timestamps_path)
    # Initialize capture
    cap = File_Capture(video_path,timestamps=list(timestamps))

    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(user_dir,"user_settings"))
    if session_settings.get("version",VersionFormat('0.0')) < get_version(version_file):
        logger.info("Session setting are from older version of this app. I will not use those.")
        session_settings.clear()

    width,height = session_settings.get('window_size',cap.frame_size)
    window_pos = session_settings.get('window_position',(0,0))
    main_window = glfwCreateWindow(width, height, "Pupil Player: "+meta_info["Recording Name"]+" - "+ rec_dir.split(os.path.sep)[-1], None, None)
    glfwSetWindowPos(main_window,window_pos[0],window_pos[1])
    glfwMakeContextCurrent(main_window)
    cygl.utils.init()

    # load pupil_positions, gaze_positions
    pupil_data = load_object(pupil_data_path)
    pupil_list = pupil_data['pupil_positions']
    gaze_list = pupil_data['gaze_positions']

    # create container for globally scoped vars
    g_pool = Global_Container()
    g_pool.app = 'player'
    g_pool.version = get_version(version_file)
    g_pool.capture = cap
    g_pool.timestamps = timestamps
    g_pool.play = False
    g_pool.new_seek = True
    g_pool.user_dir = user_dir
    g_pool.rec_dir = rec_dir
    g_pool.rec_version = rec_version
    g_pool.meta_info = meta_info
    g_pool.pupil_positions_by_frame = correlate_data(pupil_list,g_pool.timestamps)
    g_pool.gaze_positions_by_frame = correlate_data(gaze_list,g_pool.timestamps)
    g_pool.fixations_by_frame = [[] for x in g_pool.timestamps] #populated by the fixation detector plugin

    def next_frame(_):
        try:
            cap.seek_to_frame(cap.get_frame_index())
        except FileSeekError:
            pass
        g_pool.new_seek = True

    def prev_frame(_):
        try:
            cap.seek_to_frame(cap.get_frame_index()-2)
        except FileSeekError:
            pass
        g_pool.new_seek = True

    def set_scale(new_scale):
        g_pool.gui.scale = new_scale
        g_pool.gui.collect_menus()

    def open_plugin(plugin):
        if plugin ==  "Select to load":
            return
        g_pool.plugins.add(plugin)

    def purge_plugins():
        for p in g_pool.plugins:
            if p.__class__ in user_launchable_plugins:
                p.alive = False
        g_pool.plugins.clean()

    g_pool.gui = ui.UI()
    g_pool.gui.scale = session_settings.get('gui_scale',1)
    g_pool.main_menu = ui.Growing_Menu("Settings",pos=(-350,20),size=(300,400))
    g_pool.main_menu.append(ui.Button("Close Pupil Player",lambda:glfwSetWindowShouldClose(main_window,True)))
    g_pool.main_menu.append(ui.Slider('scale',g_pool.gui, setter=set_scale,step = .05,min=0.75,max=2.5,label='Interface Size'))
    g_pool.main_menu.append(ui.Info_Text('Player Version: %s'%g_pool.version))
    g_pool.main_menu.append(ui.Info_Text('Recording Version: %s'%rec_version))
    g_pool.main_menu.append(ui.Selector('Open plugin', selection = user_launchable_plugins,
                                        labels = [p.__name__.replace('_',' ') for p in user_launchable_plugins],
                                        setter= open_plugin, getter = lambda: "Select to load"))
    g_pool.main_menu.append(ui.Button('Close all plugins',purge_plugins))
    g_pool.main_menu.append(ui.Button('Reset window size',lambda: glfwSetWindowSize(main_window,cap.frame_size[0],cap.frame_size[1])) )
    g_pool.quickbar = ui.Stretching_Menu('Quick Bar',(0,100),(120,-100))
    g_pool.play_button = ui.Thumb('play',g_pool,label='Play',hotkey=GLFW_KEY_SPACE)
    g_pool.play_button.on_color[:] = (0,1.,.0,.8)
    g_pool.forward_button = ui.Thumb('forward',getter = lambda: False,setter= next_frame, hotkey=GLFW_KEY_RIGHT)
    g_pool.backward_button = ui.Thumb('backward',getter = lambda: False, setter = prev_frame, hotkey=GLFW_KEY_LEFT)
    g_pool.quickbar.extend([g_pool.play_button,g_pool.forward_button,g_pool.backward_button])
    g_pool.gui.append(g_pool.quickbar)
    g_pool.gui.append(g_pool.main_menu)


    #we always load these plugins
    system_plugins = [('Trim_Marks',{}),('Seek_Bar',{})]
    default_plugins = [('Log_Display',{}),('Scan_Path',{}),('Vis_Polyline',{}),('Vis_Circle',{}),('Export_Launcher',{})]
    previous_plugins = session_settings.get('loaded_plugins',default_plugins)
    g_pool.notifications = []
    g_pool.plugins = Plugin_List(g_pool,plugin_by_name,system_plugins+previous_plugins)

    for p in g_pool.plugins:
        if p.class_name == 'Trim_Marks':
            g_pool.trim_marks = p
            break


    # Register callbacks main_window
    glfwSetFramebufferSizeCallback(main_window,on_resize)
    glfwSetKeyCallback(main_window,on_key)
    glfwSetCharCallback(main_window,on_char)
    glfwSetMouseButtonCallback(main_window,on_button)
    glfwSetCursorPosCallback(main_window,on_pos)
    glfwSetScrollCallback(main_window,on_scroll)
    glfwSetDropCallback(main_window,on_drop)
    #trigger on_resize
    on_resize(main_window, *glfwGetFramebufferSize(main_window))

    g_pool.gui.configuration = session_settings.get('ui_config',{})

    # gl_state settings
    basic_gl_setup()
    g_pool.image_tex = Named_Texture()

    #set up performace graphs:
    pid = os.getpid()
    ps = psutil.Process(pid)
    ts = cap.get_timestamp()-.03

    cpu_graph = graph.Bar_Graph()
    cpu_graph.pos = (20,110)
    cpu_graph.update_fn = ps.cpu_percent
    cpu_graph.update_rate = 5
    cpu_graph.label = 'CPU %0.1f'

    fps_graph = graph.Bar_Graph()
    fps_graph.pos = (140,110)
    fps_graph.update_rate = 5
    fps_graph.label = "%0.0f REC FPS"

    pupil_graph = graph.Bar_Graph(max_val=1.0)
    pupil_graph.pos = (260,110)
    pupil_graph.update_rate = 5
    pupil_graph.label = "Confidence: %0.2f"

    while not glfwWindowShouldClose(main_window):

        #grab new frame
        if g_pool.play or g_pool.new_seek:
            g_pool.new_seek = False
            try:
                new_frame = cap.get_frame_nowait()
            except EndofVideoFileError:
                #end of video logic: pause at last frame.
                g_pool.play=False
            update_graph = True
        else:
            update_graph = False


        frame = new_frame.copy()
        events = {}
        #report time between now and the last loop interation
        events['dt'] = get_dt()
        #new positons we make a deepcopy just like the image is a copy.
        events['gaze_positions'] = deepcopy(g_pool.gaze_positions_by_frame[frame.index])
        events['pupil_positions'] = deepcopy(g_pool.pupil_positions_by_frame[frame.index])

        if update_graph:
            #update performace graphs
            for p in  events['pupil_positions']:
                pupil_graph.add(p['confidence'])

            t = new_frame.timestamp
            if ts != t:
                dt,ts = t-ts,t
            fps_graph.add(1./dt)

            g_pool.play_button.status_text = str(frame.index)
        #always update the CPU graph
        cpu_graph.update()

        # notify each plugin if there are new notifactions:
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
        make_coord_system_norm_based()
        g_pool.image_tex.update_from_frame(frame)
        g_pool.image_tex.draw()
        make_coord_system_pixel_based(frame.img.shape)
        # render visual feedback from loaded plugins
        for p in g_pool.plugins:
            p.gl_display()

        graph.push_view()
        fps_graph.draw()
        cpu_graph.draw()
        pupil_graph.draw()
        graph.pop_view()
        g_pool.gui.update()

        #present frames at appropriate speed
        cap.wait(frame)

        glfwSwapBuffers(main_window)
        glfwPollEvents()

    session_settings['loaded_plugins'] = g_pool.plugins.get_initializers()
    session_settings['gui_scale'] = g_pool.gui.scale
    session_settings['ui_config'] = g_pool.gui.configuration
    session_settings['window_size'] = glfwGetWindowSize(main_window)
    session_settings['window_position'] = glfwGetWindowPos(main_window)
    session_settings['version'] = g_pool.version
    session_settings.close()

    # de-init all running plugins
    for p in g_pool.plugins:
        p.alive = False
    g_pool.plugins.clean()

    cap.close()
    g_pool.gui.terminate()
    glfwDestroyWindow(main_window)



def show_no_rec_window():
    from pyglui.pyfontstash import fontstash
    from pyglui.ui import get_roboto_font_path

    def on_drop(window,count,paths):
        for x in range(count):
            new_rec_dir =  paths[x]
            if is_pupil_rec_dir(new_rec_dir):
                logger.debug("Starting new session with '%s'"%new_rec_dir)
                global rec_dir
                rec_dir = new_rec_dir
                glfwSetWindowShouldClose(window,True)
            else:
                logger.error("'%s' is not a valid pupil recording"%new_rec_dir)

    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(user_dir,"user_settings"))
    if session_settings.get("version",VersionFormat('0.0')) < get_version(version_file):
        logger.info("Session setting are from older version of this app. I will not use those.")
        session_settings.clear()
    w,h = session_settings.get('window_size',(1280,720))
    window_pos = session_settings.get('window_position',(0,0))

    glfwWindowHint(GLFW_RESIZABLE,0)
    window = glfwCreateWindow(w, h,'Pupil Player')
    glfwWindowHint(GLFW_RESIZABLE,1)

    glfwMakeContextCurrent(window)
    glfwSetWindowPos(window,window_pos[0],window_pos[1])
    glfwSetDropCallback(window,on_drop)

    adjust_gl_view(w,h)
    glfont = fontstash.Context()
    glfont.add_font('roboto',get_roboto_font_path())
    glfont.set_align_string(v_align="center",h_align="middle")
    glfont.set_color_float((0.2,0.2,0.2,0.9))
    basic_gl_setup()
    glClearColor(0.5,.5,0.5,0.0)
    text = 'Drop a recording directory onto this window.'
    tip = '(Tip: You can drop a recording directory onto the app icon.)'
    # text = "Please supply a Pupil recording directory as first arg when calling Pupil Player."
    while not glfwWindowShouldClose(window):
        clear_gl_screen()
        glfont.set_blur(10.5)
        glfont.set_color_float((0.0,0.0,0.0,1.))
        glfont.set_size(w/25.)
        glfont.draw_text(w/2,.3*h,text)
        glfont.set_size(w/30.)
        glfont.draw_text(w/2,.4*h,tip)
        glfont.set_blur(0.96)
        glfont.set_color_float((1.,1.,1.,1.))
        glfont.set_size(w/25.)
        glfont.draw_text(w/2,.3*h,text)
        glfont.set_size(w/30.)
        glfont.draw_text(w/2,.4*h,tip)
        glfwSwapBuffers(window)
        glfwPollEvents()

    session_settings['window_position'] = glfwGetWindowPos(window)
    session_settings['version'] = get_version(version_file)
    session_settings.close()
    del glfont
    glfwDestroyWindow(window)



if __name__ == '__main__':
    freeze_support()
    try:
        rec_dir = os.path.expanduser(sys.argv[1])
    except:
        #for dev, supply hardcoded dir:
        rec_dir = '/Users/mkassner/Desktop/Marker_Tracking_Demo_Recording'
        if os.path.isdir(rec_dir):
            logger.debug("Dev option: Using hardcoded data dir.")
        else:
            logger.warning("You did not supply a data directory when you called this script!")
    glfwInit()
    while rec_dir is not None:
        if not is_pupil_rec_dir(rec_dir):
            rec_dir = None
            show_no_rec_window()
        else:
            this_session_dir = rec_dir
            rec_dir = None
            session(this_session_dir)
    glfwTerminate()

    # import cProfile,subprocess,os
    # cProfile.runctx("main()",{},locals(),"player.pstats")
    # loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    # gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    # subprocess.call("python "+gprof2dot_loc+" -f pstats player.pstats | dot -Tpng -o player_cpu_time.png", shell=True)
    # print "created cpu time graph for pupil player . Please check out the png next to the main.py file"
