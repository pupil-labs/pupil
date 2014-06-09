'''

(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


import sys, os,platform
from time import sleep
from copy import deepcopy
from ctypes import c_bool, c_int

#bundle relevant imports
try:
    from billiard import freeze_support
except:
    from multiprocessing import freeze_support

if getattr(sys, 'frozen', False):
    if platform.system() == 'Darwin':
        # Specifiy user dirs.
        user_dir = os.path.expanduser('~/Desktop/pupil_player_settings')
        version_file = os.path.join(sys._MEIPASS,'_version_string_')
    else:
        # Specifiy user dirs.
        user_dir = os.path.join(sys._MEIPASS.rsplit(os.path.sep,1)[0],"player_settings")
        version_file = os.path.join(sys._MEIPASS,'_version_string_')

else:
    # We are running in a normal Python environment.
    # Make all pupil shared_modules available to this Python session.
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))
    # Specifiy user dirs.
    user_dir = os.path.join(pupil_base_dir,'player_settings')


# create folder for user settings, tmp data
if not os.path.isdir(user_dir):
    os.mkdir(user_dir)

import logging
#set up root logger before other imports
logger = logging.getLogger()
logger.setLevel(logging.INFO) # <-- use this to set verbosity
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
logger = logging.getLogger(__name__)

from file_methods import Persistent_Dict
from time import time,sleep
from ctypes import  c_int,c_bool,c_float,create_string_buffer
import numpy as np

#display
from glfw import *
import atb

from uvc_capture import autoCreateCapture,EndofVideoFileError,FileSeekError,FakeCapture

# helpers/utils
from methods import normalize, denormalize,Temp
from player_methods import correlate_gaze,patch_meta_info,is_pupil_rec_dir
from gl_utils import basic_gl_setup,adjust_gl_view, clear_gl_screen, draw_gl_point_norm,make_coord_system_pixel_based,make_coord_system_norm_based,create_named_texture,draw_named_texture


#get the current software version
if getattr(sys, 'frozen', False):
    with open(version_file) as f:
        version = f.read()
else:
    from git_version import get_tag_commit
    version = get_tag_commit()


# Plug-ins
from vis_circle import Vis_Circle
from vis_cross import Vis_Cross
from vis_polyline import Vis_Polyline
from display_gaze import Display_Gaze
from vis_light_points import Vis_Light_Points
from seek_bar import Seek_Bar
from trim_marks import Trim_Marks
from export_launcher import Export_Launcher
from scan_path import Scan_Path
from offline_marker_detector import Offline_Marker_Detector
from pupil_server import Pupil_Server
from filter_fixations import Filter_Fixations
from manual_gaze_correction import Manual_Gaze_Correction

plugin_by_index =  (Vis_Circle,Vis_Cross, Vis_Polyline, Vis_Light_Points,Scan_Path,Filter_Fixations,Manual_Gaze_Correction,Offline_Marker_Detector,Pupil_Server)
name_by_index = [p.__name__ for p in plugin_by_index]
index_by_name = dict(zip(name_by_index,range(len(name_by_index))))
plugin_by_name = dict(zip(name_by_index,plugin_by_index))
additive_plugins = (Vis_Circle,Vis_Cross,Vis_Polyline)


def main():

    # Callback functions
    def on_resize(window,w, h):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        adjust_gl_view(w,h,window)
        norm_size = normalize((w,h),glfwGetWindowSize(window))
        fb_size = denormalize(norm_size,glfwGetFramebufferSize(window))
        atb.TwWindowSize(*map(int,fb_size))
        glfwMakeContextCurrent(active_window)
        for p in g.plugins:
            p.on_window_resize(window,w,h)

    def on_key(window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,action):
            if action == GLFW_PRESS:
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
        norm_pos = normalize((x,y),glfwGetWindowSize(window))
        fb_x,fb_y = denormalize(norm_pos,glfwGetFramebufferSize(window))
        if atb.TwMouseMotion(int(fb_x),int(fb_y)):
            pass

    def on_scroll(window,x,y):
        if not atb.TwMouseWheel(int(x)):
            pass

    def on_close(window):
        glfwSetWindowShouldClose(main_window,True)
        logger.debug('Process closing from window')


    try:
        rec_dir = sys.argv[1]
    except:
        #for dev, supply hardcoded dir:
        rec_dir = '/home/mkassner/Desktop/003'
        if os.path.isdir(rec_dir):
            logger.debug("Dev option: Using hadcoded data dir.")
        else:
            if getattr(sys, 'frozen', False):
                logger.warning("You did not supply a data directory when you called this script! \
                   \nPlease drag a Pupil recoding directory onto the launch icon.")
            else:
                logger.warning("You did not supply a data directory when you called this script! \
                       \nPlease supply a Pupil recoding directory as first arg when calling Pupil Player.")
            return

    if not is_pupil_rec_dir(rec_dir):
        logger.error("You did not supply a dir with the required files inside.")
        return

    #backwards compatibility fn.
    patch_meta_info(rec_dir)

    #parse and load data folder info
    video_path = rec_dir + "/world.avi"
    timestamps_path = rec_dir + "/timestamps.npy"
    gaze_positions_path = rec_dir + "/gaze_positions.npy"
    meta_info_path = rec_dir + "/info.csv"


    #parse info.csv file
    with open(meta_info_path) as info:
        meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )
    rec_version = meta_info["Capture Software Version"]
    rec_version_float = int(filter(type(rec_version).isdigit, rec_version)[:3])/100. #(get major,minor,fix of version)
    logger.debug("Recording version: %s , %s"%(rec_version,rec_version_float))


    #load gaze information
    gaze_list = np.load(gaze_positions_path)
    timestamps = np.load(timestamps_path)

    #correlate data
    positions_by_frame = correlate_gaze(gaze_list,timestamps)


    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(user_dir,"user_settings"))
    def load(var_name,default):
        return session_settings.get(var_name,default)
    def save(var_name,var):
        session_settings[var_name] = var


    # Initialize capture
    cap = autoCreateCapture(video_path,timestamps=timestamps_path)

    if isinstance(cap,FakeCapture):
        logger.error("could not start capture.")
        return

    width,height = cap.get_size()


    # Initialize glfw
    glfwInit()
    main_window = glfwCreateWindow(width, height, "Pupil Player: "+meta_info["Recording Name"]+" - "+ rec_dir.split(os.path.sep)[-1], None, None)
    glfwMakeContextCurrent(main_window)

    # Register callbacks main_window
    glfwSetWindowSizeCallback(main_window,on_resize)
    glfwSetWindowCloseCallback(main_window,on_close)
    glfwSetKeyCallback(main_window,on_key)
    glfwSetCharCallback(main_window,on_char)
    glfwSetMouseButtonCallback(main_window,on_button)
    glfwSetCursorPosCallback(main_window,on_pos)
    glfwSetScrollCallback(main_window,on_scroll)


    # create container for globally scoped varfs (within world)
    g = Temp()
    g.plugins = []
    g.play = False
    g.new_seek = True
    g.user_dir = user_dir
    g.rec_dir = rec_dir
    g.app = 'player'
    g.timestamps = timestamps
    g.positions_by_frame = positions_by_frame



    # helpers called by the main atb bar
    def update_fps():
        old_time, bar.timestamp = bar.timestamp, time()
        dt = bar.timestamp - old_time
        if dt:
            bar.fps.value += .1 * (1. / dt - bar.fps.value)

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

    def next_frame():
        try:
            cap.seek_to_frame(cap.get_frame_index())
        except FileSeekError:
            pass
        g.new_seek = True

    def prev_frame():
        try:
            cap.seek_to_frame(cap.get_frame_index()-2)
        except FileSeekError:
            pass
        g.new_seek = True



    def open_plugin(selection,data):
        if plugin_by_index[selection] not in additive_plugins:
            for p in g.plugins:
                if isinstance(p,plugin_by_index[selection]):
                    return

        g.plugins = [p for p in g.plugins if p.alive]
        logger.debug('Open Plugin: %s'%name_by_index[selection])
        new_plugin = plugin_by_index[selection](g)
        g.plugins.append(new_plugin)
        g.plugins.sort(key=lambda p: p.order)

        if hasattr(new_plugin,'init_gui'):
            new_plugin.init_gui()
        # save the value for atb bar
        data.value=selection

    def get_from_data(data):
        """
        helper for atb getter and setter use
        """
        return data.value

    atb.init()
    # add main controls ATB bar
    bar = atb.Bar(name = "Controls", label="Controls",
            help="Scene controls", color=(50, 50, 50), alpha=100,valueswidth=150,
            text='light', position=(10, 10),refresh=.1, size=(300, 160))
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
    bar.add_button('step next',next_frame,key='right')
    bar.add_button('step prev',prev_frame,key='left')
    bar.add_var("frame index",vtype=c_int,getter=lambda:cap.get_frame_index()-1 )

    bar.plugin_to_load = c_int(0)
    plugin_type_enum = atb.enum("Plug In",index_by_name)
    bar.add_var("plugin",setter=open_plugin,getter=get_from_data,data=bar.plugin_to_load, vtype=plugin_type_enum)
    bar.add_var("version of recording",bar.recording_version, readonly=True, help="version of the capture software used to make this recording")
    bar.add_var("version of player",bar.version, readonly=True, help="version of the Pupil Player")
    bar.add_button("exit", on_close,data=main_window,key="esc")

    #set the last saved window size
    set_window_size(bar.window_size.value,bar.window_size)
    on_resize(main_window, *glfwGetWindowSize(main_window))
    glfwSetWindowPos(main_window,0,0)


    #we always load these plugins
    g.plugins.append(Export_Launcher(g,data_dir=rec_dir,frame_count=len(timestamps)))
    g.plugins.append(Seek_Bar(g,capture=cap))
    g.trim_marks = Trim_Marks(g,capture=cap)
    g.plugins.append(g.trim_marks)

    #these are loaded based on user settings
    for initializer in load('plugins',[]):
        name, args = initializer
        logger.debug("Loading plugin: %s with settings %s"%(name, args))
        try:
            p = plugin_by_name[name](g,**args)
            g.plugins.append(p)
        except:
            logger.warning("Plugin '%s' failed to load from settings file." %name)

    if load('plugins',"_") == "_":
        #lets load some default if we dont have presets
        g.plugins.append(Scan_Path(g))
        g.plugins.append(Vis_Polyline(g))
        g.plugins.append(Vis_Circle(g))
        # g.plugins.append(Vis_Light_Points(g))

    #sort by exec order
    g.plugins.sort(key=lambda p: p.order)

    #init gui
    for p in g.plugins:
        if hasattr(p,'init_gui'):
            p.init_gui()

    # gl_state settings
    basic_gl_setup()
    g.image_tex = create_named_texture((height,width,3))


    while not glfwWindowShouldClose(main_window):

        update_fps()

        #grab new frame
        if g.play or g.new_seek:
            try:
                new_frame = cap.get_frame()
            except EndofVideoFileError:
                #end of video logic: pause at last frame.
                g.play=False

            if g.new_seek:
                display_time = new_frame.timestamp
                g.new_seek = False

        frame = new_frame.copy()
        #new positons and events we make a deepcopy just like the image is a copy.
        current_pupil_positions = deepcopy(positions_by_frame[frame.index])
        events = []

        # allow each Plugin to do its work.
        for p in g.plugins:
            p.update(frame,current_pupil_positions,events)

        #check if a plugin need to be destroyed
        g.plugins = [p for p in g.plugins if p.alive]

        # render camera image
        glfwMakeContextCurrent(main_window)
        make_coord_system_norm_based()
        draw_named_texture(g.image_tex,frame.img)
        make_coord_system_pixel_based(frame.img.shape)
        # render visual feedback from loaded plugins
        for p in g.plugins:
            p.gl_display()

        #present frames at appropriate speed
        wait_time = frame.timestamp - display_time
        display_time = frame.timestamp
        try:
            spent_time = time()-timestamp
            sleep(wait_time-spent_time)
        except:
            pass
        timestamp = time()


        atb.draw()
        glfwSwapBuffers(main_window)
        glfwPollEvents()

    plugin_save = []
    for p in g.plugins:
        try:
            p_initializer = p.get_class_name(),p.get_init_dict()
            plugin_save.append(p_initializer)
        except AttributeError:
            #not all plugins need to be savable, they will not have the init dict.
            # any object without a get_init_dict method will throw this exception.
            pass

    # de-init all running plugins
    for p in g.plugins:
        p.alive = False
        #reading p.alive actually runs plug-in cleanup
        _ = p.alive

    save('plugins',plugin_save)
    save('window_size',bar.window_size.value)
    session_settings.close()

    cap.close()
    bar.destroy()
    glfwDestroyWindow(main_window)
    glfwTerminate()
    logger.debug("Process done")


if __name__ == '__main__':
    freeze_support()
    if 1:
        main()
    else:
        import cProfile,subprocess,os
        cProfile.runctx("main()",{},locals(),"player.pstats")
        loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
        gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
        subprocess.call("python "+gprof2dot_loc+" -f pstats player.pstats | dot -Tpng -o player_cpu_time.png", shell=True)
        print "created cpu time graph for pupil player . Please check out the png next to the main.py file"
