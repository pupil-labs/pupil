'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import os, sys, platform

class Global_Container(object):
    pass



class Is_Alive_Manager(object):
    '''
    A context manager to wrap the is_alive flag.
    Is alive will stay true as long is the eye process is running.
    '''
    def __init__(self, is_alive,ipc_socket,eye_id):
        self.is_alive = is_alive
        self.ipc_socket = ipc_socket
        self.eye_id = eye_id

    def __enter__( self ):
        self.is_alive.value = True
        self.ipc_socket.notify({'subject':'eye_process.started', 'eye_id':self.eye_id})

    def __exit__(self, type, value, traceback):
        if type is not None:
            pass # Exception occurred
        self.is_alive.value = False
        self.ipc_socket.notify({'subject':'eye_process.stopped', 'eye_id':self.eye_id})


def eye(timebase, is_alive_flag, ipc_pub_url, ipc_sub_url,ipc_push_url, user_dir, version, eye_id, cap_src):
    """reads eye video and detects the pupil.

    Creates a window, gl context.
    Grabs images from a capture.
    Streams Pupil coordinates.

    Reacts to notifications:
       ``set_detection_mapping_mode``: Sets detection method
       ``eye_process.should_stop``: Stops the eye process
       ``recording.started``: Starts recording eye video
       ``recording.stopped``: Stops recording eye video
       ``frame_publishing.started``: Starts frame publishing
       ``frame_publishing.stopped``: Stops frame publishing

    Emits notifications:
        ``eye_process.started``: Eye process started
        ``eye_process.stopped``: Eye process stopped

    Emits data:
        ``pupil.<eye id>``: Pupil data for eye with id ``<eye id>``
        ``frame.eye.<eye id>``: Eye frames with id ``<eye id>``
    """


    # We deferr the imports becasue of multiprocessing.
    # Otherwise the world process each process also loads the other imports.
    import zmq
    import zmq_tools
    zmq_ctx = zmq.Context()
    ipc_socket = zmq_tools.Msg_Dispatcher(zmq_ctx,ipc_push_url)
    pupil_socket = zmq_tools.Msg_Streamer(zmq_ctx,ipc_pub_url)
    notify_sub = zmq_tools.Msg_Receiver(zmq_ctx,ipc_sub_url,topics=("notify",))

    with Is_Alive_Manager(is_alive_flag,ipc_socket,eye_id):

        #logging setup
        import logging
        logging.getLogger("OpenGL").setLevel(logging.ERROR)
        logger = logging.getLogger()
        logger.handlers = []
        logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx,ipc_push_url))
        # create logger for the context of this function
        logger = logging.getLogger(__name__)

        #general imports
        import numpy as np
        import cv2

        #display
        import glfw
        from pyglui import ui,graph,cygl
        from pyglui.cygl.utils import draw_points, RGBA, draw_polyline, Named_Texture, Sphere
        import OpenGL.GL as gl
        from gl_utils import basic_gl_setup,adjust_gl_view, clear_gl_screen ,make_coord_system_pixel_based,make_coord_system_norm_based, make_coord_system_eye_camera_based,is_window_visible
        from ui_roi import UIRoi
        #monitoring
        import psutil
        import math


        # helpers/utils
        from uvc import get_time_monotonic, StreamError
        from file_methods import Persistent_Dict
        from version_utils import VersionFormat
        from methods import normalize, denormalize, Roi, timer
        from video_capture import FileCaptureError, EndofVideoFileError, CameraCaptureError
        from new_video_capture import Manager as Capture_Manager
        from av_writer import JPEG_Writer,AV_Writer

        # Pupil detectors
        from pupil_detectors import Detector_2D, Detector_3D
        pupil_detectors = {Detector_2D.__name__:Detector_2D,Detector_3D.__name__:Detector_3D}



        #UI Platform tweaks
        if platform.system() == 'Linux':
            scroll_factor = 10.0
            window_position_default = (600,300*eye_id)
        elif platform.system() == 'Windows':
            scroll_factor = 1.0
            window_position_default = (600,31+300*eye_id)
        else:
            scroll_factor = 1.0
            window_position_default = (600,300*eye_id)


        #g_pool holds variables for this process
        g_pool = Global_Container()

        # make some constants avaiable
        g_pool.user_dir = user_dir
        g_pool.version = version
        g_pool.app = 'capture'
        g_pool.timebase = timebase
        def get_timestamp():
            return get_time_monotonic()-g_pool.timebase.value
        g_pool.get_timestamp = get_timestamp
        g_pool.get_now = get_time_monotonic

        # Callback functions
        def on_resize(window,w, h):
            if is_window_visible(window):
                active_window = glfw.glfwGetCurrentContext()
                glfw.glfwMakeContextCurrent(window)
                g_pool.gui.update_window(w,h)
                graph.adjust_size(w,h)
                adjust_gl_view(w,h)
                glfw.glfwMakeContextCurrent(active_window)

        def on_key(window, key, scancode, action, mods):
            g_pool.gui.update_key(key,scancode,action,mods)

        def on_char(window,char):
            g_pool.gui.update_char(char)

        def on_iconify(window,iconified):
            g_pool.iconified = iconified

        def on_button(window,button, action, mods):
            if g_pool.display_mode == 'roi':
                if action == glfw.GLFW_RELEASE and g_pool.u_r.active_edit_pt:
                    g_pool.u_r.active_edit_pt = False
                    return # if the roi interacts we dont what the gui to interact as well
                elif action == glfw.GLFW_PRESS:
                    pos = glfw.glfwGetCursorPos(window)
                    pos = normalize(pos,glfw.glfwGetWindowSize(main_window))
                    if g_pool.flip:
                        pos = 1-pos[0],1-pos[1]
                    pos = denormalize(pos,(frame.width,frame.height)) # Position in img pixels
                    if g_pool.u_r.mouse_over_edit_pt(pos,g_pool.u_r.handle_size+40,g_pool.u_r.handle_size+40):
                        return # if the roi interacts we dont what the gui to interact as well

            g_pool.gui.update_button(button,action,mods)



        def on_pos(window,x, y):
            hdpi_factor = float(glfw.glfwGetFramebufferSize(window)[0]/glfw.glfwGetWindowSize(window)[0])
            g_pool.gui.update_mouse(x*hdpi_factor,y*hdpi_factor)

            if g_pool.u_r.active_edit_pt:
                pos = normalize((x,y),glfw.glfwGetWindowSize(main_window))
                if g_pool.flip:
                    pos = 1-pos[0],1-pos[1]
                pos = denormalize(pos,(frame.width,frame.height) )
                g_pool.u_r.move_vertex(g_pool.u_r.active_pt_idx,pos)

        def on_scroll(window,x,y):
            g_pool.gui.update_scroll(x,y*scroll_factor)


        # load session persistent settings
        session_settings = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_eye%s'%eye_id))
        if session_settings.get("version",VersionFormat('0.0')) < g_pool.version:
            logger.info("Session setting are from older version of this app. I will not use those.")
            session_settings.clear()
        # Initialize capture
        previous_settings = session_settings.get('capture_settings')
        fallback_settings = {
            'active_backend': {
                'source_type'  : 'Local / UVC',
                'active_source': {
                    'names'     : cap_src,
                    'frame_size': (640,480),
                    'frame_rate': 60
                }
            }
        }
        cap = Capture_Manager(g_pool,fallback_settings,previous_settings)

        g_pool.iconified = False
        g_pool.capture = cap
        g_pool.flip = session_settings.get('flip',False)
        g_pool.display_mode = session_settings.get('display_mode','camera_image')
        g_pool.display_mode_info_text = {'camera_image': "Raw eye camera image. This uses the least amount of CPU power",
                                    'roi': "Click and drag on the blue circles to adjust the region of interest. The region should be as small as possible, but large enough to capture all pupil movements.",
                                    'algorithm': "Algorithm display mode overlays a visualization of the pupil detection parameters on top of the eye video. Adjust parameters within the Pupil Detection menu below."}



        g_pool.u_r = UIRoi((cap.frame_size[1],cap.frame_size[0]))
        roi_user_settings = session_settings.get('roi')
        if roi_user_settings and roi_user_settings[-1] == g_pool.u_r.get()[-1]:
            g_pool.u_r.set(roi_user_settings)

        def on_frame_size_change(new_size):
            logger.warning(str(new_size))
            g_pool.u_r = UIRoi((new_size[1],new_size[0]))

        cap.on_frame_size_change = on_frame_size_change

        writer = None

        pupil_detector_settings = session_settings.get('pupil_detector_settings',None)
        last_pupil_detector = pupil_detectors[session_settings.get('last_pupil_detector',Detector_2D.__name__)]
        g_pool.pupil_detector = last_pupil_detector(g_pool,pupil_detector_settings)

        # UI callback functions
        def set_scale(new_scale):
            g_pool.gui.scale = new_scale
            g_pool.gui.collect_menus()


        def set_display_mode_info(val):
            g_pool.display_mode = val
            g_pool.display_mode_info.text = g_pool.display_mode_info_text[val]


        def set_detector(new_detector):
            g_pool.pupil_detector.cleanup()
            g_pool.pupil_detector = new_detector(g_pool)
            g_pool.pupil_detector.init_gui(g_pool.sidebar)


        # Initialize glfw
        glfw.glfwInit()
        title = "eye %s"%eye_id
        width,height = session_settings.get('window_size',cap.frame_size)
        main_window = glfw.glfwCreateWindow(width,height, title, None, None)
        window_pos = session_settings.get('window_position',window_position_default)
        glfw.glfwSetWindowPos(main_window,window_pos[0],window_pos[1])
        glfw.glfwMakeContextCurrent(main_window)
        cygl.utils.init()

        # gl_state settings
        basic_gl_setup()
        g_pool.image_tex = Named_Texture()
        glfw.glfwSwapInterval(0)

        #setup GUI
        g_pool.gui = ui.UI()
        g_pool.gui.scale = session_settings.get('gui_scale',1)
        g_pool.sidebar = ui.Scrolling_Menu("Settings",pos=(-300,0),size=(0,0),header_pos='left')
        general_settings = ui.Growing_Menu('General')
        general_settings.append(ui.Slider('scale',g_pool.gui, setter=set_scale,step = .05,min=1.,max=2.5,label='Interface Size'))
        general_settings.append(ui.Button('Reset window size',lambda: glfw.glfwSetWindowSize(main_window,frame.width,frame.height)) )
        general_settings.append(ui.Switch('flip',g_pool,label='Flip image display'))
        general_settings.append(ui.Selector('display_mode',g_pool,setter=set_display_mode_info,selection=['camera_image','roi','algorithm'], labels=['Camera Image', 'ROI', 'Algorithm'], label="Mode") )
        g_pool.display_mode_info = ui.Info_Text(g_pool.display_mode_info_text[g_pool.display_mode])
        general_settings.append(g_pool.display_mode_info)
        g_pool.sidebar.append(general_settings)
        g_pool.gui.append(g_pool.sidebar)
        detector_selector = ui.Selector('pupil_detector',getter = lambda: g_pool.pupil_detector.__class__ ,setter=set_detector,selection=[Detector_2D, Detector_3D],labels=['C++ 2d detector', 'C++ 3d detector'], label="Detection method")
        general_settings.append(detector_selector)

        # let detector add its GUI
        g_pool.pupil_detector.init_gui(g_pool.sidebar)
        # let the camera add its GUI
        g_pool.capture.init_gui(g_pool.sidebar)


        # Register callbacks main_window
        glfw.glfwSetFramebufferSizeCallback(main_window,on_resize)
        glfw.glfwSetWindowIconifyCallback(main_window,on_iconify)
        glfw.glfwSetKeyCallback(main_window,on_key)
        glfw.glfwSetCharCallback(main_window,on_char)
        glfw.glfwSetMouseButtonCallback(main_window,on_button)
        glfw.glfwSetCursorPosCallback(main_window,on_pos)
        glfw.glfwSetScrollCallback(main_window,on_scroll)

        #set the last saved window size
        on_resize(main_window, *glfw.glfwGetWindowSize(main_window))


        # load last gui configuration
        g_pool.gui.configuration = session_settings.get('ui_config',{})


        #set up performance graphs
        pid = os.getpid()
        ps = psutil.Process(pid)
        ts = g_pool.get_timestamp()

        cpu_graph = graph.Bar_Graph()
        cpu_graph.pos = (20,130)
        cpu_graph.update_fn = ps.cpu_percent
        cpu_graph.update_rate = 5
        cpu_graph.label = 'CPU %0.1f'

        fps_graph = graph.Bar_Graph()
        fps_graph.pos = (140,130)
        fps_graph.update_rate = 5
        fps_graph.label = "%0.0f FPS"

        should_publish_frames = False
        frame_publish_format = 'jpeg'

        #create a timer to control window update frequency
        window_update_timer = timer(1/60.)
        def window_should_update():
            return next(window_update_timer)

        logger.warning('Process started.')

        # Event loop
        while not glfw.glfwWindowShouldClose(main_window):

            if notify_sub.new_data:
                t,notification = notify_sub.recv()
                subject = notification['subject']
                if subject == 'eye_process.should_stop':
                    if notification['eye_id'] == eye_id:
                        break
                elif subject == 'set_detection_mapping_mode':
                    if notification['mode'] == '3d':
                        if not isinstance(g_pool.pupil_detector,Detector_3D):
                            set_detector(Detector_3D)
                        detector_selector.read_only  = True
                    else:
                        if not isinstance(g_pool.pupil_detector,Detector_2D):
                            set_detector(Detector_2D)
                        detector_selector.read_only = False
                elif subject == 'recording.started':
                    if notification['record_eye']:
                        record_path = notification['rec_path']
                        raw_mode = notification['compression']
                        logger.info("Will save eye video to: %s"%record_path)
                        timestamps_path = os.path.join(record_path, "eye%s_timestamps.npy"%eye_id)
                        if raw_mode and frame.jpeg_buffer:
                            video_path = os.path.join(record_path, "eye%s.mp4"%eye_id)
                            writer = JPEG_Writer(video_path,cap.frame_rate)
                        else:
                            video_path = os.path.join(record_path, "eye%s.mp4"%eye_id)
                            writer = AV_Writer(video_path,cap.frame_rate)
                        timestamps = []
                elif subject == 'recording.stopped':
                    if writer:
                        logger.info("Done recording.")
                        writer.release()
                        writer = None
                        np.save(timestamps_path,np.asarray(timestamps))
                        del timestamps
                elif subject.startswith('meta.should_doc'):
                    ipc_socket.notify({
                        'subject':'meta.doc',
                        'actor':'eye%i'%eye_id,
                        'doc':eye.__doc__
                        })
                elif subject.startswith('frame_publishing.started'):
                    should_publish_frames = True
                    frame_publish_format = notification.get('format','jpeg')
                elif subject.startswith('frame_publishing.stopped'):
                    should_publish_frames = False
                    frame_publish_format = 'jpeg'

            # Get an image from the grabber
            try:
                frame = cap.get_frame()
            except StreamError:
                break
            except EndofVideoFileError:
                logger.warning("Video File is done. Stopping")
                cap.seek_to_frame(0)
                frame = cap.get_frame()

            if should_publish_frames and frame.jpeg_buffer:
                if   frame_publish_format == "jpeg":
                    data = frame.jpeg_buffer
                elif frame_publish_format == "yuv":
                    data = frame.yuv_buffer
                elif frame_publish_format == "bgr":
                    data = frame.bgr
                elif frame_publish_format == "gray":
                    data = frame.gray
                pupil_socket.send('frame.eye.%s'%eye_id,{
                    'width': frame.width,
                    'height': frame.width,
                    'index': frame.index,
                    'timestamp': frame.timestamp,
                    'format': frame_publish_format,
                    '__raw_data__': [data]
                })


            #update performace graphs
            t = frame.timestamp
            dt,ts = t-ts,t
            try:
                fps_graph.add(1./dt)
            except ZeroDivisionError:
                pass
            cpu_graph.update()



            if writer:
                writer.write_video_frame(frame)
                timestamps.append(frame.timestamp)


            # pupil ellipse detection
            result = g_pool.pupil_detector.detect(frame, g_pool.u_r, g_pool.display_mode == 'algorithm')
            result['id'] = eye_id
            # stream the result
            pupil_socket.send('pupil.%s'%eye_id,result)

            # GL drawing
            if window_should_update():
                if is_window_visible(main_window):
                    glfw.glfwMakeContextCurrent(main_window)
                    clear_gl_screen()

                    # switch to work in normalized coordinate space
                    if g_pool.display_mode == 'algorithm':
                        g_pool.image_tex.update_from_ndarray(frame.img)
                    elif g_pool.display_mode in ('camera_image','roi'):
                        g_pool.image_tex.update_from_ndarray(frame.gray)
                    else:
                        pass

                    make_coord_system_norm_based(g_pool.flip)
                    g_pool.image_tex.draw()

                    window_size =  glfw.glfwGetWindowSize(main_window)
                    make_coord_system_pixel_based((frame.height,frame.width,3),g_pool.flip)

                    if result['method'] == '3d c++':

                        eye_ball = result['projected_sphere']
                        try:
                            pts = cv2.ellipse2Poly( (int(eye_ball['center'][0]),int(eye_ball['center'][1])),
                                                (int(eye_ball['axes'][0]/2),int(eye_ball['axes'][1]/2)),
                                                int(eye_ball['angle']),0,360,8)
                        except ValueError as e:
                            pass
                        else:
                            draw_polyline(pts,2,RGBA(0.,.9,.1,result['model_confidence']) )

                    if result['confidence'] >0:
                        if result.has_key('ellipse'):
                            pts = cv2.ellipse2Poly( (int(result['ellipse']['center'][0]),int(result['ellipse']['center'][1])),
                                            (int(result['ellipse']['axes'][0]/2),int(result['ellipse']['axes'][1]/2)),
                                            int(result['ellipse']['angle']),0,360,15)
                            confidence = result['confidence'] * 0.7 #scale it a little
                            draw_polyline(pts,1,RGBA(1.,0,0,confidence))
                            draw_points([result['ellipse']['center']],size=20,color=RGBA(1.,0.,0.,confidence),sharpness=1.)

                    # render graphs
                    graph.push_view()
                    fps_graph.draw()
                    cpu_graph.draw()
                    graph.pop_view()

                    # render GUI
                    g_pool.gui.update()

                    #render the ROI
                    g_pool.u_r.draw(g_pool.gui.scale)
                    if g_pool.display_mode == 'roi':
                        g_pool.u_r.draw_points(g_pool.gui.scale)

                    #update screen
                    glfw.glfwSwapBuffers(main_window)
                glfw.glfwPollEvents()
                g_pool.pupil_detector.visualize() #detector decides if we visualize or not


        # END while running

        # in case eye recording was still runnnig: Save&close
        if writer:
            logger.info("Done recording eye.")
            writer = None
            np.save(timestamps_path,np.asarray(timestamps))

        glfw.glfwRestoreWindow(main_window) #need to do this for windows os
        # save session persistent settings
        session_settings['gui_scale'] = g_pool.gui.scale
        session_settings['roi'] = g_pool.u_r.get()
        session_settings['flip'] = g_pool.flip
        session_settings['display_mode'] = g_pool.display_mode
        session_settings['ui_config'] = g_pool.gui.configuration
        session_settings['capture_settings'] = g_pool.capture.settings
        session_settings['window_size'] = glfw.glfwGetWindowSize(main_window)
        session_settings['window_position'] = glfw.glfwGetWindowPos(main_window)
        session_settings['version'] = g_pool.version
        session_settings['last_pupil_detector'] = g_pool.pupil_detector.__class__.__name__
        session_settings['pupil_detector_settings'] = g_pool.pupil_detector.get_settings()
        session_settings.close()

        g_pool.pupil_detector.cleanup()
        g_pool.gui.terminate()
        glfw.glfwDestroyWindow(main_window)
        glfw.glfwTerminate()
        cap.close()
        logger.info("Process shutting down.")

def eye_profiled(timebase, is_alive_flag,ipc_pub_url,ipc_sub_url,ipc_push_url, user_dir, version, eye_id, cap_src):
    import cProfile,subprocess,os
    from eye import eye
    cProfile.runctx("eye(timebase, is_alive_flag,ipc_pub_url,ipc_sub_url,ipc_push_url, user_dir, version, eye_id, cap_src)",{'timebase':timebase,'is_alive_flag':is_alive_flag,'ipc_pub_url':ipc_pub_url,'ipc_sub_url':ipc_sub_url,'ipc_push_url':ipc_push_url, 'user_dir':user_dir, 'version':version, 'eye_id':eye_id,'cap_src':cap_src},locals(),"eye%s.pstats"%eye_id)
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats eye%s.pstats | dot -Tpng -o eye%s_cpu_time.png"%(eye_id,eye_id), shell=True)
    print "created cpu time graph for eye%s process. Please check out the png next to the eye.py file"%eye_id

