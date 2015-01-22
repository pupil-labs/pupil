'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
from time import sleep
from file_methods import Persistent_Dict
import logging
import numpy as np

#display
from glfw import *
from pyglui import ui,graph
from pyglui.cygl.utils import init as cygl_init
from pyglui.cygl.utils import draw_points as cygl_draw_points
from pyglui.cygl.utils import RGBA as cygl_rgba

# check versions for our own depedencies as they are fast-changing
from pyglui import __version__ as pyglui_version
assert pyglui_version >= '0.1'

#monitoring
import psutil

# helpers/utils
from gl_utils import basic_gl_setup,adjust_gl_view, clear_gl_screen, draw_gl_point_norm,make_coord_system_pixel_based,make_coord_system_norm_based,create_named_texture,draw_named_texture,draw_gl_polyline
from methods import *
from uvc_capture import autoCreateCapture, FileCaptureError, EndofVideoFileError, CameraCaptureError

# Pupil detectors
from pupil_detectors import Canny_Detector

def eye(g_pool,cap_src,cap_size,eye_id=0):
    """
    Creates a window, gl context.
    Grabs images from a capture.
    Streams Pupil coordinates into g_pool.pupil_queue
    """

    # modify the root logger for this process
    logger = logging.getLogger()
    # remove inherited handlers
    logger.handlers = []
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(g_pool.user_dir,'eye.log'),mode='w')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('Eye Process: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    formatter = logging.Formatter('EYE Process [%(levelname)s] %(name)s : %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # create logger for the context of this function
    logger = logging.getLogger(__name__)


    # Callback functions
    def on_resize(window,w, h):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        hdpi_factor = glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0]
        w,h = w*hdpi_factor, h*hdpi_factor
        g_pool.gui.update_window(w,h)
        graph.adjust_size(w,h)
        adjust_gl_view(w,h)
        # for p in g_pool.plugins:
            # p.on_window_resize(window,w,h)
        glfwMakeContextCurrent(active_window)

    def on_key(window, key, scancode, action, mods):
        g_pool.gui.update_key(key,scancode,action,mods)
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                on_close(window)

    def on_char(window,char):
        g_pool.gui.update_char(char)


    def on_button(window,button, action, mods):
        if g_pool.display_mode == 'roi':
            if action == GLFW_RELEASE and u_r.active_edit_pt:
                u_r.active_edit_pt = False
                return # if the roi interacts we dont what the gui to interact as well
            elif action == GLFW_PRESS:
                pos = glfwGetCursorPos(window)
                pos = normalize(pos,glfwGetWindowSize(eye_window))
                pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
                if u_r.mouse_over_edit_pt(pos,u_r.handle_size,u_r.handle_size):
                    return # if the roi interacts we dont what the gui to interact as well

        g_pool.gui.update_button(button,action,mods)



    def on_pos(window,x, y):
        hdpi_factor = float(glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0])
        g_pool.gui.update_mouse(x*hdpi_factor,y*hdpi_factor)

        if u_r.active_edit_pt:
            pos = normalize((x,y),glfwGetWindowSize(eye_window))
            pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) )
            u_r.move_vertex(u_r.active_pt_idx,pos)

    def on_scroll(window,x,y):
        g_pool.gui.update_scroll(x,y)

    def on_close(window):
        g_pool.quit.value = True
        logger.info('Process closing from window')


    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_eye'))

    # Initialize capture
    cap = autoCreateCapture(cap_src, cap_size, 24, timebase=g_pool.timebase)

    # Test capture
    try:
        frame = cap.get_frame()
    except CameraCaptureError:
        logger.error("Could not retrieve image from capture")
        cap.close()
        return

    g_pool.capture = cap

    # any object we attach to the g_pool object *from now on* will only be visible to this process!
    # vars should be declared here to make them visible to the code reader.
    g_pool.window_size = session_settings.get('window_size',1.)
    g_pool.display_mode = session_settings.get('display_mode','camera_image')
    # g_pool.draw_pupil = session_settings.get('draw_pupil',True)

    u_r = UIRoi(frame.img.shape)
    u_r.set(session_settings.get('roi',u_r.get()))

    writer = None

    pupil_detector = Canny_Detector(g_pool)


    # UI callback functions
    def set_window_size(size):
        w,h = int(frame.width*size),int(frame.height*size)
        glfwSetWindowSize(eye_window,w,h)

    def set_scale(new_scale):
        g_pool.gui.scale = new_scale

    def get_scale():
        return g_pool.gui.scale

    # Initialize glfw
    glfwInit()
    eye_window = glfwCreateWindow(frame.width, frame.height, "Eye", None, None)
    glfwMakeContextCurrent(eye_window)
    cygl_init()

    # Register callbacks eye_window
    glfwSetWindowSizeCallback(eye_window,on_resize)
    glfwSetWindowCloseCallback(eye_window,on_close)
    glfwSetKeyCallback(eye_window,on_key)
    glfwSetCharCallback(eye_window,on_char)
    glfwSetMouseButtonCallback(eye_window,on_button)
    glfwSetCursorPosCallback(eye_window,on_pos)
    glfwSetScrollCallback(eye_window,on_scroll)

    # gl_state settings
    basic_gl_setup()
    g_pool.image_tex = create_named_texture(frame.img)

    # refresh speed settings
    glfwSwapInterval(0)
    glfwSetWindowPos(eye_window,800,0)


    #setup GUI
    g_pool.gui = ui.UI()
    g_pool.gui.scale = session_settings.get('gui_scale',1)
    g_pool.sidebar = ui.Scrolling_Menu("Settings",pos=(-300,0),size=(0,0),header_pos='left')
    g_pool.sidebar.configuration = session_settings.get('side_bar_config',{'collapsed':True})
    general_settings = ui.Growing_Menu('General')
    general_settings.configuration = session_settings.get('general_menu_config',{})
    general_settings.append(ui.Slider('scale', setter=set_scale,getter=get_scale,step = .05,min=1.,max=2.5,label='Interface Size'))
    general_settings.append(ui.Selector('display_mode',g_pool,selection=['camera_image','roi','algorithm','cpu_save'], labels=['Camera Image', 'ROI', 'Algorithm', 'CPU Save'], label="Mode") )
    g_pool.sidebar.append(general_settings)
    g_pool.pupil_detector_menu = ui.Growing_Menu('Pupil Detector')
    g_pool.pupil_detector_menu.configuration = session_settings.get('pupil_detector_menu_config',{'collapsed':True})
    g_pool.sidebar.append(g_pool.pupil_detector_menu)

    g_pool.gui.append(g_pool.sidebar)

    # let the camera add its GUI
    g_pool.capture.init_gui(g_pool.sidebar)
    g_pool.capture.menu.configuration = session_settings.get('capture_menu_config',{'collapsed':True})

    # let detector add its GUI
    pupil_detector.init_gui()


    #set the last saved window size
    set_window_size(g_pool.window_size)
    on_resize(eye_window, *glfwGetWindowSize(eye_window))

    #set up performance graphs
    pid = os.getpid()
    ps = psutil.Process(pid)
    ts = frame.timestamp

    cpu_graph = graph.Bar_Graph()
    cpu_graph.pos = (20,110)
    cpu_graph.update_fn = ps.get_cpu_percent
    cpu_graph.update_rate = 5
    cpu_graph.label = 'CPU %0.1f'

    fps_graph = graph.Bar_Graph()
    fps_graph.pos = (140,110)
    fps_graph.update_rate = 5
    fps_graph.label = "%0.0f FPS"

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
        t = frame.timestamp
        dt,ts = t-ts,t
        fps_graph.add(1./dt)
        cpu_graph.update()


        ###  RECORDING of Eye Video (on demand) ###
        # Setup variables and lists for recording
        if g_pool.eye_rx.poll():
            command = g_pool.eye_rx.recv()
            if command is not None:
                record_path = command
                logger.info("Will save eye video to: %s"%record_path)
                video_path = os.path.join(record_path, "eye.mkv")
                timestamps_path = os.path.join(record_path, "eye_timestamps.npy")
                writer = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), float(cap.frame_rate), (frame.img.shape[1], frame.img.shape[0]))
                timestamps = []
            else:
                logger.info("Done recording eye.")
                writer = None
                np.save(timestamps_path,np.asarray(timestamps))
                del timestamps

        if writer:
            writer.write(frame.img)
            timestamps.append(frame.timestamp)


        # pupil ellipse detection
        result = pupil_detector.detect(frame,user_roi=u_r,visualize=g_pool.display_mode == 'algorithm')
        result['id'] = eye_id
        # stream the result
        g_pool.pupil_queue.put(result)


        # GL drawing
        glfwMakeContextCurrent(eye_window)
        clear_gl_screen()

        # switch to work in normalized coordinate space
        make_coord_system_norm_based()
        if g_pool.display_mode != 'cpu_save':
            draw_named_texture(g_pool.image_tex,frame.img)
        else:
            draw_named_texture(g_pool.image_tex)

        # switch to work in pixel space
        make_coord_system_pixel_based(frame.img.shape)

        if result['confidence'] >0:
            if result.has_key('axes'):
                pts = cv2.ellipse2Poly( (int(result['center'][0]),int(result['center'][1])),
                                        (int(result["axes"][0]/2),int(result["axes"][1]/2)),
                                        int(result["angle"]),0,360,15)
                draw_gl_polyline(pts,(1.,0,0,.5))
            # draw_gl_point_norm(result['norm_pos'],color=(1.,0.,0.,0.5))
            cygl_draw_points([result['center']],size=20,color=cygl_rgba(1.,0.,0.,.5),sharpness=1.)

        # render graphs
        graph.push_view()
        fps_graph.draw()
        cpu_graph.draw()
        graph.pop_view()

        # render GUI
        g_pool.gui.update()

        #render the ROI
        if g_pool.display_mode == 'roi':
            u_r.draw()

        #update screen
        glfwSwapBuffers(eye_window)
        glfwPollEvents()


    # END while running

    # in case eye recording was still runnnig: Save&close
    if writer:
        logger.info("Done recording eye.")
        writer = None
        np.save(timestamps_path,np.asarray(timestamps))


    # save session persistent settings
    session_settings['gui_scale'] = g_pool.gui.scale
    session_settings['roi'] = u_r.get()
    session_settings['window_size'] = g_pool.window_size
    session_settings['display_mode'] = g_pool.display_mode
    session_settings['side_bar_config'] = g_pool.sidebar.configuration
    session_settings['capture_menu_config'] = g_pool.capture.menu.configuration
    session_settings['general_menu_config'] = general_settings.configuration
    session_settings['pupil_detector_menu_config'] = g_pool.pupil_detector_menu.configuration
    session_settings.close()

    pupil_detector.cleanup()
    cap.close()
    glfwDestroyWindow(eye_window)
    glfwTerminate()

    #flushing queue in case world process did not exit gracefully
    while not g_pool.pupil_queue.empty():
        g_pool.pupil_queue.get()
    g_pool.pupil_queue.close()

    logger.debug("Process done")

def eye_profiled(g_pool,cap_src,cap_size):
    import cProfile,subprocess,os
    from eye import eye
    cProfile.runctx("eye(g_pool,cap_src,cap_size)",{"g_pool":g_pool,'cap_src':cap_src,'cap_size':cap_size},locals(),"eye.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats eye.pstats | dot -Tpng -o eye_cpu_time.png", shell=True)
    print "created cpu time graph for eye process. Please check out the png next to the eye.py file"



class UIRoi(Roi):
    """
    this object inherits from ROI and adds some UI helper functions
    """
    def __init__(self,array_shape):
        super(UIRoi, self).__init__(array_shape)
        self.max_x = array_shape[1]
        self.max_y = array_shape[0]

        self.handle_size = 20
        self.active_edit_pt = False
        self.active_pt_idx = None
        self.handle_color = cygl_rgba(.5,.5,.9,.9)
        self.handle_color_selected = cygl_rgba(.5,.9,.9,.9)
        self.handle_color_shadow = cygl_rgba(.0,.0,.0,.5)

    @property
    def rect(self):
        return [[self.lX,self.lY],
                [self.uX,self.lY],
                [self.uX,self.uY],
                [self.lX,self.uY]]

    def move_vertex(self,vert_idx,(x,y)):
        nx,ny = min(self.max_x,int(x)),min(self.max_y,int(y))
        thresh = 25
        if vert_idx == 0:
            if self.uX-nx > thresh and self.uY-ny > thresh:
                self.lX,self.lY = max(0,nx),max(0,ny)
        if vert_idx == 1:
            if nx-self.lX > thresh and self.uY-ny > thresh:
                self.uX,self.lY = min(self.max_x,nx),max(0,ny)
        if vert_idx == 2:
            if nx-self.lX > thresh and ny-self.lY > thresh:
                self.uX,self.uY = min(self.max_x,nx),min(self.max_y,ny)
        if vert_idx == 3:
            if self.uX-nx > thresh and ny-self.lY > thresh:
                self.lX,self.uY = max(0,nx),min(self.max_y,ny)

    def mouse_over_center(self,edit_pt,mouse_pos,w,h):
        return edit_pt[0]-w/2 <= mouse_pos[0] <=edit_pt[0]+w/2 and edit_pt[1]-h/2 <= mouse_pos[1] <=edit_pt[1]+h/2

    def mouse_over_edit_pt(self,mouse_pos,w,h):
        for p,i in zip(self.rect,range(4)):
            if self.mouse_over_center(p,mouse_pos,w,h):
                self.active_pt_idx = i
                self.active_edit_pt = True
                return True

    def draw(self):
        draw_gl_polyline(self.rect,(.8,.8,.8,0.9),thickness=2)
        if self.active_edit_pt:
            inactive_pts = self.rect[:self.active_pt_idx]+self.rect[self.active_pt_idx+1:]
            active_pt = [self.rect[self.active_pt_idx]]
            cygl_draw_points(inactive_pts,size=self.handle_size+10,color=self.handle_color_shadow,sharpness=0.3)
            cygl_draw_points(inactive_pts,size=self.handle_size,color=self.handle_color,sharpness=0.9)
            cygl_draw_points(active_pt,size=self.handle_size+30,color=self.handle_color_shadow,sharpness=0.3)                        
            cygl_draw_points(active_pt,size=self.handle_size+10,color=self.handle_color_selected,sharpness=0.9)
        else:
            cygl_draw_points(self.rect,size=self.handle_size+10,color=self.handle_color_shadow,sharpness=0.3)
            cygl_draw_points(self.rect,size=self.handle_size,color=self.handle_color,sharpness=0.9)

