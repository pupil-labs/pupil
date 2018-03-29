'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import platform


class Global_Container(object):
    pass


class Is_Alive_Manager(object):
    '''
    A context manager to wrap the is_alive flag.
    Is alive will stay true as long is the eye process is running.
    '''

    def __init__(self, is_alive, ipc_socket, eye_id, logger):
        self.is_alive = is_alive
        self.ipc_socket = ipc_socket
        self.eye_id = eye_id
        self.logger = logger

    def __enter__(self):
        self.is_alive.value = True
        self.ipc_socket.notify({'subject': 'eye_process.started',
                                'eye_id': self.eye_id})
        return self

    def __exit__(self, etype, value, traceback):
        if etype is not None:
            import traceback as tb
            self.logger.error('Process Eye{} crashed with trace:\n'.format(self.eye_id) +
                              ''.join(tb.format_exception(etype, value, traceback)))
            self.ipc_socket.notify({'subject': 'eye_process.stopped',
                                    'eye_id': self.eye_id})

        self.is_alive.value = False
        self.ipc_socket.notify({'subject': 'eye_process.stopped',
                                'eye_id': self.eye_id})
        return True  # do not propergate exception


def eye(timebase, is_alive_flag, ipc_pub_url, ipc_sub_url, ipc_push_url,
        user_dir, version, eye_id, overwrite_cap_settings=None):
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
    ipc_socket = zmq_tools.Msg_Dispatcher(zmq_ctx, ipc_push_url)
    pupil_socket = zmq_tools.Msg_Streamer(zmq_ctx, ipc_pub_url)
    notify_sub = zmq_tools.Msg_Receiver(zmq_ctx, ipc_sub_url, topics=("notify",))

    # logging setup
    import logging
    logging.getLogger("OpenGL").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    if is_alive_flag.value:
        # indicates eye process that this is a duplicated startup
        logger.warning('Aborting redundant eye process startup')
        return

    with Is_Alive_Manager(is_alive_flag, ipc_socket, eye_id, logger):
        # general imports
        import numpy as np
        import cv2

        # display
        import glfw
        from pyglui import ui, graph, cygl
        from pyglui.cygl.utils import draw_points, RGBA, draw_polyline
        from pyglui.cygl.utils import Named_Texture
        from gl_utils import basic_gl_setup, adjust_gl_view, clear_gl_screen
        from gl_utils import make_coord_system_pixel_based
        from gl_utils import make_coord_system_norm_based
        from gl_utils import is_window_visible, glViewport
        from ui_roi import UIRoi
        # monitoring
        import psutil

        # helpers/utils
        from uvc import get_time_monotonic
        from file_methods import Persistent_Dict
        from version_utils import VersionFormat
        from methods import normalize, denormalize, timer
        from av_writer import JPEG_Writer, AV_Writer
        from ndsi import H264Writer
        from video_capture import source_classes
        from video_capture import manager_classes

        # Pupil detectors
        from pupil_detectors import Detector_2D, Detector_3D, Detector_Dummy
        pupil_detectors = {Detector_2D.__name__: Detector_2D,
                           Detector_3D.__name__: Detector_3D,
                           Detector_Dummy.__name__: Detector_Dummy}

        # UI Platform tweaks
        if platform.system() == 'Linux':
            scroll_factor = 10.0
            window_position_default = (600, 300 * eye_id + 30)
        elif platform.system() == 'Windows':
            scroll_factor = 10.0
            window_position_default = (600, 90 + 300 * eye_id)
        else:
            scroll_factor = 1.0
            window_position_default = (600, 300 * eye_id)

        icon_bar_width = 50
        window_size = None
        camera_render_size = None
        hdpi_factor = 1.

        # g_pool holds variables for this process
        g_pool = Global_Container()

        # make some constants avaiable
        g_pool.user_dir = user_dir
        g_pool.version = version
        g_pool.app = 'capture'
        g_pool.process = 'eye{}'.format(eye_id)
        g_pool.timebase = timebase

        g_pool.ipc_pub = ipc_socket

        def get_timestamp():
            return get_time_monotonic() - g_pool.timebase.value
        g_pool.get_timestamp = get_timestamp
        g_pool.get_now = get_time_monotonic

        # Callback functions
        def on_resize(window, w, h):
            nonlocal window_size
            nonlocal camera_render_size
            nonlocal hdpi_factor

            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwMakeContextCurrent(window)
            hdpi_factor = glfw.getHDPIFactor(window)
            g_pool.gui.scale = g_pool.gui_user_scale * hdpi_factor
            window_size = w, h
            camera_render_size = w-int(icon_bar_width*g_pool.gui.scale), h
            g_pool.gui.update_window(w, h)
            g_pool.gui.collect_menus()
            for g in g_pool.graphs:
                g.scale = hdpi_factor
                g.adjust_window_size(w, h)
            adjust_gl_view(w, h)
            glfw.glfwMakeContextCurrent(active_window)

        def on_window_key(window, key, scancode, action, mods):
            g_pool.gui.update_key(key, scancode, action, mods)

        def on_window_char(window, char):
            g_pool.gui.update_char(char)

        def on_iconify(window, iconified):
            g_pool.iconified = iconified

        def on_window_mouse_button(window, button, action, mods):
            g_pool.gui.update_button(button, action, mods)

        def on_pos(window, x, y):
            x *= hdpi_factor
            y *= hdpi_factor
            g_pool.gui.update_mouse(x, y)

            if g_pool.u_r.active_edit_pt:
                pos = normalize((x, y), camera_render_size)
                if g_pool.flip:
                    pos = 1 - pos[0], 1 - pos[1]
                pos = denormalize(pos, g_pool.capture.frame_size)
                g_pool.u_r.move_vertex(g_pool.u_r.active_pt_idx, pos)

        def on_scroll(window, x, y):
            g_pool.gui.update_scroll(x, y * scroll_factor)

        def on_drop(window, count, paths):
            paths = [paths[x].decode('utf-8') for x in range(count)]
            g_pool.capture_manager.on_drop(paths)
            g_pool.capture.on_drop(paths)

        # load session persistent settings
        session_settings = Persistent_Dict(os.path.join(g_pool.user_dir, 'user_settings_eye{}'.format(eye_id)))
        if VersionFormat(session_settings.get("version", '0.0')) != g_pool.version:
            logger.info("Session setting are from a different version of this app. I will not use those.")
            session_settings.clear()

        g_pool.iconified = False
        g_pool.capture = None
        g_pool.capture_manager = None
        g_pool.flip = session_settings.get('flip', False)
        g_pool.display_mode = session_settings.get(
            'display_mode', 'camera_image')
        g_pool.display_mode_info_text = {'camera_image': "Raw eye camera image. This uses the least amount of CPU power",
                                         'roi': "Click and drag on the blue circles to adjust the region of interest. The region should be as small as possible, but large enough to capture all pupil movements.",
                                         'algorithm': "Algorithm display mode overlays a visualization of the pupil detection parameters on top of the eye video. Adjust parameters within the Pupil Detection menu below."}

        capture_manager_settings = session_settings.get(
            'capture_manager_settings', ('UVC_Manager',{}))

        manager_class_name, manager_settings = capture_manager_settings
        manager_class_by_name = {c.__name__: c for c in manager_classes}
        g_pool.capture_manager = manager_class_by_name[manager_class_name](g_pool, **manager_settings)

        if eye_id == 0:
            cap_src = ["Pupil Cam2 ID0", "Pupil Cam1 ID0", "HD-6000"]
        else:
            cap_src = ["Pupil Cam2 ID1", "Pupil Cam1 ID1"]

        # Initialize capture
        default_settings = ('UVC_Source', {
                            'preferred_names': cap_src,
                            'frame_size': (320, 240),
                            'frame_rate': 120
                            })

        capture_source_settings = overwrite_cap_settings or session_settings.get('capture_settings', default_settings)
        source_class_name, source_settings = capture_source_settings
        source_class_by_name = {c.__name__:c for c in source_classes}
        g_pool.capture = source_class_by_name[source_class_name](g_pool,**source_settings)
        assert g_pool.capture

        g_pool.u_r = UIRoi((g_pool.capture.frame_size[1],g_pool.capture.frame_size[0]))
        roi_user_settings = session_settings.get('roi')
        if roi_user_settings and tuple(roi_user_settings[-1]) == g_pool.u_r.get()[-1]:
            g_pool.u_r.set(roi_user_settings)

        pupil_detector_settings = session_settings.get(
            'pupil_detector_settings', None)
        last_pupil_detector = pupil_detectors[session_settings.get(
            'last_pupil_detector', Detector_2D.__name__)]
        g_pool.pupil_detector = last_pupil_detector(
            g_pool, pupil_detector_settings)

        def set_display_mode_info(val):
            g_pool.display_mode = val
            g_pool.display_mode_info.text = g_pool.display_mode_info_text[val]

        def set_detector(new_detector):
            g_pool.pupil_detector.deinit_ui()
            g_pool.pupil_detector.cleanup()
            g_pool.pupil_detector = new_detector(g_pool)
            g_pool.pupil_detector.init_ui()

        def toggle_general_settings(collapsed):
            #this is the menu toggle logic.
            # Only one menu can be open.
            # If no menu is open the menubar should collapse.
            g_pool.menubar.collapsed = collapsed
            for m in g_pool.menubar.elements:
                m.collapsed = True
            general_settings.collapsed = collapsed

        # Initialize glfw
        glfw.glfwInit()
        title = "Pupil Capture - eye {}".format(eye_id)

        width, height = g_pool.capture.frame_size
        width *= 2
        height *= 2
        width += icon_bar_width
        width, height = session_settings.get('window_size', (width, height))

        main_window = glfw.glfwCreateWindow(width, height, title, None, None)
        window_pos = session_settings.get('window_position', window_position_default)
        glfw.glfwSetWindowPos(main_window, window_pos[0], window_pos[1])
        glfw.glfwMakeContextCurrent(main_window)
        cygl.utils.init()

        # UI callback functions
        def set_scale(new_scale):
            g_pool.gui_user_scale = new_scale
            on_resize(main_window, *glfw.glfwGetFramebufferSize(main_window))

        # gl_state settings
        basic_gl_setup()
        g_pool.image_tex = Named_Texture()
        g_pool.image_tex.update_from_ndarray(np.ones((1,1),dtype=np.uint8)+125)

        # setup GUI
        g_pool.gui = ui.UI()
        g_pool.gui_user_scale = session_settings.get('gui_scale', 1.)
        g_pool.menubar = ui.Scrolling_Menu("Settings", pos=(-500, 0), size=(-icon_bar_width, 0), header_pos='left')
        g_pool.iconbar = ui.Scrolling_Menu("Icons",pos=(-icon_bar_width,0),size=(0,0),header_pos='hidden')
        g_pool.gui.append(g_pool.menubar)
        g_pool.gui.append(g_pool.iconbar)

        general_settings = ui.Growing_Menu('General',header_pos='headline')
        general_settings.append(ui.Selector('gui_user_scale', g_pool,
                                          setter=set_scale,
                                          selection=[.8, .9, 1., 1.1, 1.2],
                                          label='Interface Size'))

        def set_window_size():
            f_width, f_height = g_pool.capture.frame_size
            f_width *= 2
            f_height *= 2
            f_width += int(icon_bar_width * g_pool.gui.scale)
            glfw.glfwSetWindowSize(main_window, f_width, f_height)

        def uroi_on_mouse_button(button, action, mods):
            if g_pool.display_mode == 'roi':
                if action == glfw.GLFW_RELEASE and g_pool.u_r.active_edit_pt:
                    g_pool.u_r.active_edit_pt = False
                    # if the roi interacts we dont want
                    # the gui to interact as well
                    return
                elif action == glfw.GLFW_PRESS:
                    x, y = glfw.glfwGetCursorPos(main_window)
                    # pos = normalize(pos, glfw.glfwGetWindowSize(main_window))
                    x *= hdpi_factor
                    y *= hdpi_factor
                    pos = normalize((x, y), camera_render_size)
                    if g_pool.flip:
                        pos = 1 - pos[0], 1 - pos[1]
                    # Position in img pixels
                    pos = denormalize(pos, g_pool.capture.frame_size)  # Position in img pixels
                    if g_pool.u_r.mouse_over_edit_pt(pos, g_pool.u_r.handle_size, g_pool.u_r.handle_size):
                        # if the roi interacts we dont want
                        # the gui to interact as well
                        return

        general_settings.append(ui.Button('Reset window size', set_window_size))
        general_settings.append(ui.Switch('flip',g_pool,label='Flip image display'))
        general_settings.append(ui.Selector('display_mode',
                                            g_pool,
                                            setter=set_display_mode_info,
                                            selection=['camera_image','roi','algorithm'],
                                            labels=['Camera Image', 'ROI', 'Algorithm'],
                                            label="Mode")
                                            )
        g_pool.display_mode_info = ui.Info_Text(g_pool.display_mode_info_text[g_pool.display_mode])

        general_settings.append(g_pool.display_mode_info)

        detector_selector = ui.Selector('pupil_detector',
                                        getter=lambda: g_pool.pupil_detector.__class__,
                                        setter=set_detector,
                                        selection=[Detector_Dummy,
                                                   Detector_2D,
                                                   Detector_3D],
                                        labels=['disabled',
                                                'C++ 2d detector',
                                                'C++ 3d detector'],
                                        label="Detection method")
        general_settings.append(detector_selector)

        g_pool.menubar.append(general_settings)
        icon = ui.Icon('collapsed', general_settings, label=chr(0xe8b8), on_val=False, off_val=True, setter=toggle_general_settings, label_font='pupil_icons')
        icon.tooltip = 'General Settings'
        g_pool.iconbar.append(icon)
        toggle_general_settings(False)

        g_pool.pupil_detector.init_ui()
        g_pool.capture.init_ui()
        g_pool.capture_manager.init_ui()
        g_pool.writer = None

        def replace_source(source_class_name, source_settings):
            g_pool.capture.deinit_ui()
            g_pool.capture.cleanup()
            g_pool.capture = source_class_by_name[source_class_name](g_pool,**source_settings)
            g_pool.capture.init_ui()
            if g_pool.writer:
                logger.info("Done recording.")
                try:
                    g_pool.writer.release()
                except RuntimeError:
                    logger.error('No eye video recorded')
                g_pool.writer = None

        g_pool.replace_source = replace_source # for ndsi capture


        # Register callbacks main_window
        glfw.glfwSetFramebufferSizeCallback(main_window, on_resize)
        glfw.glfwSetWindowIconifyCallback(main_window, on_iconify)
        glfw.glfwSetKeyCallback(main_window, on_window_key)
        glfw.glfwSetCharCallback(main_window, on_window_char)
        glfw.glfwSetMouseButtonCallback(main_window, on_window_mouse_button)
        glfw.glfwSetCursorPosCallback(main_window, on_pos)
        glfw.glfwSetScrollCallback(main_window, on_scroll)
        glfw.glfwSetDropCallback(main_window, on_drop)

        # load last gui configuration
        g_pool.gui.configuration = session_settings.get('ui_config', {})

        # set up performance graphs
        pid = os.getpid()
        ps = psutil.Process(pid)
        ts = g_pool.get_timestamp()

        cpu_graph = graph.Bar_Graph()
        cpu_graph.pos = (20, 50)
        cpu_graph.update_fn = ps.cpu_percent
        cpu_graph.update_rate = 5
        cpu_graph.label = 'CPU %0.1f'

        fps_graph = graph.Bar_Graph()
        fps_graph.pos = (140, 50)
        fps_graph.update_rate = 5
        fps_graph.label = "%0.0f FPS"
        g_pool.graphs = [cpu_graph, fps_graph]

        # set the last saved window size
        on_resize(main_window, *glfw.glfwGetFramebufferSize(main_window))

        should_publish_frames = False
        frame_publish_format = 'jpeg'
        frame_publish_format_recent_warning = False

        # create a timer to control window update frequency
        window_update_timer = timer(1 / 60)

        def window_should_update():
            return next(window_update_timer)

        logger.warning('Process started.')

        frame = None

        # Event loop
        while not glfw.glfwWindowShouldClose(main_window):

            if notify_sub.new_data:
                t, notification = notify_sub.recv()
                subject = notification['subject']
                if subject.startswith('eye_process.should_stop'):
                    if notification['eye_id'] == eye_id:
                        break
                elif subject == 'set_detection_mapping_mode':
                    if notification['mode'] == '3d':
                        if not isinstance(g_pool.pupil_detector, Detector_3D):
                            set_detector(Detector_3D)
                        detector_selector.read_only = True
                    elif notification['mode'] == '2d':
                        if not isinstance(g_pool.pupil_detector, Detector_2D):
                            set_detector(Detector_2D)
                        detector_selector.read_only = False
                    else:
                        if not isinstance(g_pool.pupil_detector, Detector_Dummy):
                            set_detector(Detector_Dummy)
                        detector_selector.read_only = True
                elif subject == 'recording.started':
                    if notification['record_eye'] and g_pool.capture.online:
                        record_path = notification['rec_path']
                        raw_mode = notification['compression']
                        logger.info("Will save eye video to: {}".format(record_path))
                        video_path = os.path.join(record_path, "eye{}.mp4".format(eye_id))
                        if raw_mode and frame and g_pool.capture.jpeg_support:
                            g_pool.writer = JPEG_Writer(video_path, g_pool.capture.frame_rate)
                        elif hasattr(g_pool.capture._recent_frame, 'h264_buffer'):
                            g_pool.writer = H264Writer(video_path,
                                                       g_pool.capture.frame_size[0],
                                                       g_pool.capture.frame_size[1],
                                                       g_pool.capture.frame_rate)
                        else:
                            g_pool.writer = AV_Writer(video_path, g_pool.capture.frame_rate)
                elif subject == 'recording.stopped':
                    if g_pool.writer:
                        logger.info("Done recording.")
                        try:
                            g_pool.writer.release()
                        except RuntimeError:
                            logger.error('No eye video recorded')
                        g_pool.writer = None
                elif subject.startswith('meta.should_doc'):
                    ipc_socket.notify({
                        'subject': 'meta.doc',
                        'actor': 'eye{}'.format(eye_id),
                        'doc': eye.__doc__
                    })
                elif subject.startswith('frame_publishing.started'):
                    should_publish_frames = True
                    frame_publish_format = notification.get('format', 'jpeg')
                elif subject.startswith('frame_publishing.stopped'):
                    should_publish_frames = False
                    frame_publish_format = 'jpeg'
                elif subject.startswith('start_eye_capture') and notification['target'] == g_pool.process:
                    replace_source(notification['name'],notification['args'])

                g_pool.capture.on_notify(notification)

            # Get an image from the grabber
            event = {}
            g_pool.capture.recent_events(event)
            frame = event.get('frame')
            g_pool.capture_manager.recent_events(event)
            if frame:
                f_width, f_height = g_pool.capture.frame_size
                if (g_pool.u_r.array_shape[0], g_pool.u_r.array_shape[1]) != (f_height, f_width):
                    g_pool.pupil_detector.on_resolution_change((g_pool.u_r.array_shape[1], g_pool.u_r.array_shape[0]), g_pool.capture.frame_size)
                    g_pool.u_r = UIRoi((f_height, f_width))
                if should_publish_frames:
                    try:
                        if frame_publish_format == "jpeg":
                            data = frame.jpeg_buffer
                        elif frame_publish_format == "yuv":
                            data = frame.yuv_buffer
                        elif frame_publish_format == "bgr":
                            data = frame.bgr
                        elif frame_publish_format == "gray":
                            data = frame.gray
                        assert data is not None
                    except (AttributeError, AssertionError, NameError):
                        if not frame_publish_format_recent_warning:
                            frame_publish_format_recent_warning = True
                            logger.warning('{}s are not compatible with format "{}"'.format(type(frame), frame_publish_format))
                    else:
                        frame_publish_format_recent_warning = False
                        pupil_socket.send('frame.eye.%s'%eye_id,{
                            'width': frame.width,
                            'height': frame.height,
                            'index': frame.index,
                            'timestamp': frame.timestamp,
                            'format': frame_publish_format,
                            '__raw_data__': [data]
                        })

                t = frame.timestamp
                dt, ts = t - ts, t
                try:
                    fps_graph.add(1./dt)
                except ZeroDivisionError:
                    pass

                if g_pool.writer:
                    g_pool.writer.write_video_frame(frame)

                # pupil ellipse detection
                result = g_pool.pupil_detector.detect(frame, g_pool.u_r, g_pool.display_mode == 'algorithm')
                if result is not None:
                    result['id'] = eye_id

                    # stream the result
                    pupil_socket.send('pupil.%s'%eye_id,result)

            cpu_graph.update()

            # GL drawing
            if window_should_update():
                if is_window_visible(main_window):
                    glfw.glfwMakeContextCurrent(main_window)
                    clear_gl_screen()

                    if frame:
                        # switch to work in normalized coordinate space
                        if g_pool.display_mode == 'algorithm':
                            g_pool.image_tex.update_from_ndarray(frame.img)
                        elif g_pool.display_mode in ('camera_image', 'roi'):
                            g_pool.image_tex.update_from_ndarray(frame.gray)
                        else:
                            pass
                    glViewport(0, 0, *camera_render_size)
                    make_coord_system_norm_based(g_pool.flip)
                    g_pool.image_tex.draw()

                    f_width, f_height = g_pool.capture.frame_size
                    make_coord_system_pixel_based((f_height, f_width, 3), g_pool.flip)
                    if frame and result:
                        if result['method'] == '3d c++':
                            eye_ball = result['projected_sphere']
                            try:
                                pts = cv2.ellipse2Poly(
                                    (int(eye_ball['center'][0]),
                                     int(eye_ball['center'][1])),
                                    (int(eye_ball['axes'][0] / 2),
                                     int(eye_ball['axes'][1] / 2)),
                                    int(eye_ball['angle']), 0, 360, 8)
                            except ValueError as e:
                                pass
                            else:
                                draw_polyline(pts, 2, RGBA(0., .9, .1, result['model_confidence']))
                        if result['confidence'] > 0:
                            if 'ellipse' in result:
                                pts = cv2.ellipse2Poly(
                                    (int(result['ellipse']['center'][0]),
                                     int(result['ellipse']['center'][1])),
                                    (int(result['ellipse']['axes'][0] / 2),
                                     int(result['ellipse']['axes'][1] / 2)),
                                    int(result['ellipse']['angle']), 0, 360, 15)
                                confidence = result['confidence'] * 0.7
                                draw_polyline(pts, 1, RGBA(1., 0, 0, confidence))
                                draw_points([result['ellipse']['center']],
                                            size=20,
                                            color=RGBA(1., 0., 0., confidence),
                                            sharpness=1.)

                    glViewport(0, 0, *camera_render_size)
                    make_coord_system_pixel_based((f_height, f_width, 3), g_pool.flip)
                    # render the ROI
                    g_pool.u_r.draw(g_pool.gui.scale)
                    if g_pool.display_mode == 'roi':
                        g_pool.u_r.draw_points(g_pool.gui.scale)

                    glViewport(0, 0, *window_size)
                    make_coord_system_pixel_based((*window_size[::-1], 3), g_pool.flip)
                    # render graphs
                    fps_graph.draw()
                    cpu_graph.draw()

                    # render GUI
                    unused_elements = g_pool.gui.update()
                    for butt in unused_elements.buttons:
                        uroi_on_mouse_button(*butt)

                    make_coord_system_pixel_based((*window_size[::-1], 3), g_pool.flip)

                    g_pool.pupil_detector.visualize()  # detector decides if we visualize or not

                    # update screen
                    glfw.glfwSwapBuffers(main_window)
                glfw.glfwPollEvents()

        # END while running

        # in case eye recording was still runnnig: Save&close
        if g_pool.writer:
            logger.info("Done recording eye.")
            g_pool.writer = None

        glfw.glfwRestoreWindow(main_window)  # need to do this for windows os
        # save session persistent settings
        session_settings['gui_scale'] = g_pool.gui_user_scale
        session_settings['roi'] = g_pool.u_r.get()
        session_settings['flip'] = g_pool.flip
        session_settings['display_mode'] = g_pool.display_mode
        session_settings['ui_config'] = g_pool.gui.configuration
        session_settings['capture_settings'] = g_pool.capture.class_name, g_pool.capture.get_init_dict()
        session_settings['capture_manager_settings'] = g_pool.capture_manager.class_name, g_pool.capture_manager.get_init_dict()
        session_settings['window_position'] = glfw.glfwGetWindowPos(main_window)
        session_settings['version'] = str(g_pool.version)
        session_settings['last_pupil_detector'] = g_pool.pupil_detector.__class__.__name__
        session_settings['pupil_detector_settings'] = g_pool.pupil_detector.get_settings()

        session_window_size = glfw.glfwGetWindowSize(main_window)
        if 0 not in session_window_size:
            session_settings['window_size'] = session_window_size

        session_settings.close()

        g_pool.capture.deinit_ui()
        g_pool.capture_manager.deinit_ui()
        g_pool.pupil_detector.deinit_ui()

        g_pool.pupil_detector.cleanup()
        g_pool.capture_manager.cleanup()
        g_pool.capture.cleanup()

        glfw.glfwDestroyWindow(main_window)
        g_pool.gui.terminate()
        glfw.glfwTerminate()
        logger.info("Process shutting down.")


def eye_profiled(timebase, is_alive_flag, ipc_pub_url, ipc_sub_url, ipc_push_url, user_dir, version, eye_id, overwrite_cap_settings=None):
    import cProfile
    import subprocess
    import os
    from .eye import eye
    cProfile.runctx("eye(timebase, is_alive_flag,ipc_pub_url,ipc_sub_url,ipc_push_url, user_dir, version, eye_id, overwrite_cap_settings)",
                    {'timebase': timebase, 'is_alive_flag': is_alive_flag, 'ipc_pub_url': ipc_pub_url,
                     'ipc_sub_url': ipc_sub_url, 'ipc_push_url': ipc_push_url, 'user_dir': user_dir,
                     'version': version, 'eye_id': eye_id, 'overwrite_cap_settings': overwrite_cap_settings},
                    locals(), "eye{}.pstats".format(eye_id))
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules', 'gprof2dot.py')
    subprocess.call("python " + gprof2dot_loc + " -f pstats eye{0}.pstats | dot -Tpng -o eye{0}_cpu_time.png".format(eye_id), shell=True)
    print("created cpu time graph for eye{} process. Please check out the png next to the eye.py file".format(eye_id))
