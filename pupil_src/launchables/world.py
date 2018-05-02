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


def world(timebase, eyes_are_alive, ipc_pub_url, ipc_sub_url,
          ipc_push_url, user_dir, version):
    """Reads world video and runs plugins.

    Creates a window, gl context.
    Grabs images from a capture.
    Maps pupil to gaze data
    Can run various plug-ins.

    Reacts to notifications:
        ``set_detection_mapping_mode``
        ``eye_process.started``
        ``start_plugin``

    Emits notifications:
        ``eye_process.should_start``
        ``eye_process.should_stop``
        ``set_detection_mapping_mode``
        ``world_process.started``
        ``world_process.stopped``
        ``recording.should_stop``: Emits on camera failure
        ``launcher_process.should_stop``

    Emits data:
        ``gaze``: Gaze data from current gaze mapping plugin.``
        ``*``: any other plugin generated data in the events
               that it not [dt,pupil,gaze].
    """

    # We defer the imports because of multiprocessing.
    # Otherwise the world process each process also loads the other imports.
    # This is not harmful but unnecessary.

    # general imports
    from time import sleep
    import logging

    # networking
    import zmq
    import zmq_tools

    # zmq ipc setup
    zmq_ctx = zmq.Context()
    ipc_pub = zmq_tools.Msg_Dispatcher(zmq_ctx, ipc_push_url)
    notify_sub = zmq_tools.Msg_Receiver(zmq_ctx, ipc_sub_url, topics=('notify',))

    # log setup
    logging.getLogger("OpenGL").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    def launch_eye_process(eye_id, delay=0):
        n = {'subject': 'eye_process.should_start.{}'.format(eye_id),
             'eye_id': eye_id, 'delay': delay}
        ipc_pub.notify(n)

    def stop_eye_process(eye_id):
        n = {'subject': 'eye_process.should_stop.{}'.format(eye_id), 'eye_id': eye_id,'delay':0.2}
        ipc_pub.notify(n)

    def start_stop_eye(eye_id, make_alive):
        if make_alive:
            launch_eye_process(eye_id)
        else:
            stop_eye_process(eye_id)

    def set_detection_mapping_mode(new_mode):
        n = {'subject': 'set_detection_mapping_mode', 'mode': new_mode}
        ipc_pub.notify(n)

    try:

        # display
        import glfw
        from version_utils import VersionFormat
        from pyglui import ui, cygl, __version__ as pyglui_version
        assert VersionFormat(pyglui_version) >= VersionFormat('1.21'), 'pyglui out of date, please upgrade to newest version'
        from pyglui.cygl.utils import Named_Texture
        import gl_utils

        # helpers/utils
        from file_methods import Persistent_Dict
        from methods import normalize, denormalize, delta_t, get_system_info, timer
        from uvc import get_time_monotonic
        logger.info('Application Version: {}'.format(version))
        logger.info('System Info: {}'.format(get_system_info()))

        import audio

        # trigger pupil detector cpp build:
        import pupil_detectors
        del pupil_detectors

        # Plug-ins
        from plugin import Plugin, System_Plugin_Base, Plugin_List, import_runtime_plugins
        from plugin_manager import Plugin_Manager
        from calibration_routines import calibration_plugins, gaze_mapping_plugins, Calibration_Plugin, Gaze_Mapping_Plugin
        from fixation_detector import Fixation_Detector
        from recorder import Recorder
        from display_recent_gaze import Display_Recent_Gaze
        from time_sync import Time_Sync
        from pupil_remote import Pupil_Remote
        from pupil_groups import Pupil_Groups
        from surface_tracker import Surface_Tracker
        from log_display import Log_Display
        from annotations import Annotation_Capture
        from log_history import Log_History
        from frame_publisher import Frame_Publisher
        from blink_detection import Blink_Detection
        from video_capture import source_classes, manager_classes,Base_Manager,Base_Source
        from pupil_data_relay import Pupil_Data_Relay
        from remote_recorder import Remote_Recorder
        from audio_capture import Audio_Capture
        from accuracy_visualizer import Accuracy_Visualizer
        # from saccade_detector import Saccade_Detector
        from system_graphs import System_Graphs
        from camera_intrinsics_estimation import Camera_Intrinsics_Estimation
        from hololens_relay import Hololens_Relay

        # UI Platform tweaks
        if platform.system() == 'Linux':
            scroll_factor = 10.0
            window_position_default = (30, 30)
        elif platform.system() == 'Windows':
            scroll_factor = 10.0
            window_position_default = (8, 90)
        else:
            scroll_factor = 1.0
            window_position_default = (0, 0)

        icon_bar_width = 50
        window_size = None
        camera_render_size = None
        hdpi_factor = 1.0

        # g_pool holds variables for this process they are accessible to all plugins
        g_pool = Global_Container()
        g_pool.app = 'capture'
        g_pool.process = 'world'
        g_pool.user_dir = user_dir
        g_pool.version = version
        g_pool.timebase = timebase
        g_pool.zmq_ctx = zmq_ctx
        g_pool.ipc_pub = ipc_pub
        g_pool.ipc_pub_url = ipc_pub_url
        g_pool.ipc_sub_url = ipc_sub_url
        g_pool.ipc_push_url = ipc_push_url
        g_pool.eyes_are_alive = eyes_are_alive

        def get_timestamp():
            return get_time_monotonic() - g_pool.timebase.value
        g_pool.get_timestamp = get_timestamp
        g_pool.get_now = get_time_monotonic

        # manage plugins
        runtime_plugins = import_runtime_plugins(os.path.join(g_pool.user_dir, 'plugins'))
        user_plugins = [Audio_Capture,
                        Pupil_Groups,
                        Frame_Publisher,
                        Pupil_Remote,
                        Time_Sync,
                        Surface_Tracker,
                        Annotation_Capture,
                        Log_History,
                        Fixation_Detector,
                        Blink_Detection,
                        Remote_Recorder,
                        Accuracy_Visualizer,
                        Camera_Intrinsics_Estimation,
                        Hololens_Relay]
        system_plugins = [Log_Display, Display_Recent_Gaze, Recorder, Pupil_Data_Relay, Plugin_Manager, System_Graphs] + manager_classes + source_classes
        plugins = system_plugins + user_plugins + runtime_plugins + calibration_plugins + gaze_mapping_plugins
        user_plugins += [p for p in runtime_plugins if not isinstance(p, (Base_Manager, Base_Source, System_Plugin_Base,
                                                                          Calibration_Plugin, Gaze_Mapping_Plugin))]
        g_pool.plugin_by_name = {p.__name__: p for p in plugins}

        default_capture_settings = {
            'preferred_names': ["Pupil Cam1 ID2", "Logitech Camera", "(046d:081d)",
                                "C510", "B525", "C525", "C615", "C920", "C930e"],
            'frame_size': (1280, 720),
            'frame_rate': 30
        }

        default_plugins = [("UVC_Source", default_capture_settings),
                           ('Pupil_Data_Relay', {}),
                           ('UVC_Manager', {}),
                           ('Log_Display', {}),
                           ('Dummy_Gaze_Mapper', {}),
                           ('Display_Recent_Gaze', {}),
                           ('Screen_Marker_Calibration', {}),
                           ('Recorder', {}),
                           ('Pupil_Remote', {}),
                           ('Accuracy_Visualizer', {}),
                           ('Plugin_Manager', {}),
                           ('System_Graphs', {})]

        # Callback functions
        def on_resize(window, w, h):
            nonlocal window_size
            nonlocal camera_render_size
            nonlocal hdpi_factor
            if w == 0 or h == 0:
                return
            hdpi_factor = glfw.getHDPIFactor(window)
            g_pool.gui.scale = g_pool.gui_user_scale * hdpi_factor
            window_size = w,h
            camera_render_size = w-int(icon_bar_width*g_pool.gui.scale), h
            g_pool.gui.update_window(*window_size)
            g_pool.gui.collect_menus()

            for p in g_pool.plugins:
                p.on_window_resize(window, *camera_render_size)

        def on_window_key(window, key, scancode, action, mods):
            g_pool.gui.update_key(key, scancode, action, mods)

        def on_window_char(window, char):
            g_pool.gui.update_char(char)

        def on_window_mouse_button(window, button, action, mods):
            g_pool.gui.update_button(button, action, mods)

        def on_pos(window, x, y):
            x, y = x * hdpi_factor, y * hdpi_factor
            g_pool.gui.update_mouse(x, y)
            pos = x, y
            pos = normalize(pos, camera_render_size)
            # Position in img pixels
            pos = denormalize(pos, g_pool.capture.frame_size)
            for p in g_pool.plugins:
                p.on_pos(pos)

        def on_scroll(window, x, y):
            g_pool.gui.update_scroll(x, y * scroll_factor)

        def on_drop(window, count, paths):
            paths = [paths[x].decode('utf-8') for x in range(count)]
            for p in g_pool.plugins:
                p.on_drop(paths)

        tick = delta_t()

        def get_dt():
            return next(tick)

        # load session persistent settings
        session_settings = Persistent_Dict(os.path.join(g_pool.user_dir, 'user_settings_world'))
        if VersionFormat(session_settings.get("version", '0.0')) != g_pool.version:
            logger.info("Session setting are from a different version of this app. I will not use those.")
            session_settings.clear()

        g_pool.min_calibration_confidence = session_settings.get('min_calibration_confidence', 0.8)
        g_pool.detection_mapping_mode = session_settings.get('detection_mapping_mode', '3d')
        g_pool.active_calibration_plugin = None
        g_pool.active_gaze_mapping_plugin = None
        g_pool.capture = None

        audio.audio_mode = session_settings.get('audio_mode', audio.default_audio_mode)

        def handle_notifications(n):
            subject = n['subject']
            if subject == 'set_detection_mapping_mode':
                if n['mode'] == '2d':
                    if ("Vector_Gaze_Mapper" in
                            g_pool.active_gaze_mapping_plugin.class_name):
                        logger.warning("The gaze mapper is not supported in 2d mode. Please recalibrate.")
                        g_pool.plugins.add(g_pool.plugin_by_name['Dummy_Gaze_Mapper'])
                g_pool.detection_mapping_mode = n['mode']
            elif subject == 'start_plugin':
                g_pool.plugins.add(g_pool.plugin_by_name[n['name']], args=n.get('args', {}))
            elif subject == 'stop_plugin':
                for p in g_pool.plugins:
                    if p.class_name == n['name']:
                        p.alive = False
                        g_pool.plugins.clean()
            elif subject == 'eye_process.started':
                n = {'subject': 'set_detection_mapping_mode',
                     'mode': g_pool.detection_mapping_mode}
                ipc_pub.notify(n)
            elif subject.startswith('meta.should_doc'):
                ipc_pub.notify({'subject': 'meta.doc',
                                'actor': g_pool.app,
                                'doc': world.__doc__})
                for p in g_pool.plugins:
                    if (p.on_notify.__doc__
                            and p.__class__.on_notify != Plugin.on_notify):
                        ipc_pub.notify({'subject': 'meta.doc',
                                        'actor': p.class_name,
                                        'doc': p.on_notify.__doc__})

        width, height = session_settings.get('window_size', (1280 + icon_bar_width, 720))

        # window and gl setup
        glfw.glfwInit()
        main_window = glfw.glfwCreateWindow(width, height, "Pupil Capture - World")
        window_pos = session_settings.get('window_position', window_position_default)
        glfw.glfwSetWindowPos(main_window, window_pos[0], window_pos[1])
        glfw.glfwMakeContextCurrent(main_window)
        cygl.utils.init()
        g_pool.main_window = main_window

        def set_scale(new_scale):
            g_pool.gui_user_scale = new_scale
            window_size = camera_render_size[0] + \
                int(icon_bar_width * g_pool.gui_user_scale * hdpi_factor), \
                glfw.glfwGetFramebufferSize(main_window)[1]
            logger.warning(icon_bar_width*g_pool.gui_user_scale*hdpi_factor)
            glfw.glfwSetWindowSize(main_window, *window_size)

        def reset_restart():
            logger.warning("Resetting all settings and restarting Capture.")
            glfw.glfwSetWindowShouldClose(main_window, True)
            ipc_pub.notify({'subject': 'clear_settings_process.should_start'})
            ipc_pub.notify({'subject': 'world_process.should_start', 'delay': 2.})

        def toggle_general_settings(collapsed):
            # this is the menu toggle logic.
            # Only one menu can be open.
            # If no menu is opened, the menubar should collapse.
            g_pool.menubar.collapsed = collapsed
            for m in g_pool.menubar.elements:
                m.collapsed = True
            general_settings.collapsed = collapsed

        # setup GUI
        g_pool.gui = ui.UI()
        g_pool.gui_user_scale = session_settings.get('gui_scale', 1.)
        g_pool.menubar = ui.Scrolling_Menu("Settings", pos=(-400, 0), size=(-icon_bar_width, 0), header_pos='left')
        g_pool.iconbar = ui.Scrolling_Menu("Icons",pos=(-icon_bar_width,0),size=(0,0),header_pos='hidden')
        g_pool.quickbar = ui.Stretching_Menu('Quick Bar', (0, 100), (120, -100))
        g_pool.gui.append(g_pool.menubar)
        g_pool.gui.append(g_pool.iconbar)
        g_pool.gui.append(g_pool.quickbar)

        general_settings = ui.Growing_Menu('General',header_pos='headline')
        general_settings.append(ui.Selector('gui_user_scale', g_pool, setter=set_scale, selection=[.6, .8, 1., 1.2, 1.4], label='Interface size'))

        def set_window_size():
            f_width, f_height = g_pool.capture.frame_size
            f_width += int(icon_bar_width*g_pool.gui.scale)
            glfw.glfwSetWindowSize(main_window, f_width, f_height)

        general_settings.append(ui.Button('Reset window size', set_window_size))
        general_settings.append(ui.Selector('audio_mode', audio, selection=audio.audio_modes))
        general_settings.append(ui.Selector('detection_mapping_mode',
                                            g_pool,
                                            label='detection & mapping mode',
                                            setter=set_detection_mapping_mode,
                                            selection=['disabled', '2d', '3d']))
        general_settings.append(ui.Switch('eye0_process',
                                            label='Detect eye 0',
                                            setter=lambda alive: start_stop_eye(0,alive),
                                            getter=lambda: eyes_are_alive[0].value
                                        ))
        general_settings.append(ui.Switch('eye1_process',
                                            label='Detect eye 1',
                                            setter=lambda alive: start_stop_eye(1,alive),
                                            getter=lambda: eyes_are_alive[1].value
                                        ))

        general_settings.append(ui.Info_Text('Capture Version: {}'.format(g_pool.version)))
        general_settings.append(ui.Button('Restart with default settings', reset_restart))

        g_pool.menubar.append(general_settings)
        icon = ui.Icon('collapsed', general_settings, label=chr(0xe8b8), on_val=False, off_val=True, setter=toggle_general_settings, label_font='pupil_icons')
        icon.tooltip = 'General Settings'
        g_pool.iconbar.append(icon)

        user_plugin_separator = ui.Separator()
        user_plugin_separator.order = 0.35
        g_pool.iconbar.append(user_plugin_separator)

        # plugins that are loaded based on user settings from previous session
        g_pool.plugins = Plugin_List(g_pool, session_settings.get('loaded_plugins', default_plugins))

        # Register callbacks main_window
        glfw.glfwSetFramebufferSizeCallback(main_window, on_resize)
        glfw.glfwSetKeyCallback(main_window, on_window_key)
        glfw.glfwSetCharCallback(main_window, on_window_char)
        glfw.glfwSetMouseButtonCallback(main_window, on_window_mouse_button)
        glfw.glfwSetCursorPosCallback(main_window, on_pos)
        glfw.glfwSetScrollCallback(main_window, on_scroll)
        glfw.glfwSetDropCallback(main_window, on_drop)

        # gl_state settings
        gl_utils.basic_gl_setup()
        g_pool.image_tex = Named_Texture()

        toggle_general_settings(True)

        # now that we have a proper window we can load the last gui configuration
        g_pool.gui.configuration = session_settings.get('ui_config', {})

        # create a timer to control window update frequency
        window_update_timer = timer(1 / 60)
        def window_should_update():
            return next(window_update_timer)

        # trigger setup of window and gl sizes
        on_resize(main_window, *glfw.glfwGetFramebufferSize(main_window))

        if session_settings.get('eye1_process_alive', True):
            launch_eye_process(1, delay=0.6)
        if session_settings.get('eye0_process_alive', True):
            launch_eye_process(0, delay=0.3)

        ipc_pub.notify({'subject': 'world_process.started'})
        logger.warning('Process started.')

        # Event loop
        while not glfw.glfwWindowShouldClose(main_window):

            # fetch newest notifications
            new_notifications = []
            while notify_sub.new_data:
                t, n = notify_sub.recv()
                new_notifications.append(n)

            # notify each plugin if there are new notifications:
            for n in new_notifications:
                handle_notifications(n)
                for p in g_pool.plugins:
                    p.on_notify(n)

            # a dictionary that allows plugins to post and read events
            events = {}
            # report time between now and the last loop interation
            events['dt'] = get_dt()

            # allow each Plugin to do its work.
            for p in g_pool.plugins:
                p.recent_events(events)

            # check if a plugin need to be destroyed
            g_pool.plugins.clean()

            # send new events to ipc:
            del events['pupil_positions']  # already on the wire
            del events['gaze_positions']  # sent earlier
            if 'frame' in events:
                del events['frame']  # send explicitly with frame publisher
            if 'depth_frame' in events:
                del events['depth_frame']
            if 'audio_packets' in events:
                del events['audio_packets']
            del events['dt']  # no need to send this
            for topic, data in events.items():
                assert(isinstance(data, (list, tuple)))
                for d in data:
                    ipc_pub.send(topic, d)

            glfw.glfwMakeContextCurrent(main_window)
            # render visual feedback from loaded plugins
            if window_should_update() and gl_utils.is_window_visible(main_window):

                gl_utils.glViewport(0, 0, *camera_render_size)
                for p in g_pool.plugins:
                    p.gl_display()

                gl_utils.glViewport(0, 0, *window_size)
                try:
                    clipboard = glfw.glfwGetClipboardString(main_window).decode()
                except AttributeError:  # clipboard is None, might happen on startup
                    clipboard = ''
                g_pool.gui.update_clipboard(clipboard)
                user_input = g_pool.gui.update()
                if user_input.clipboard != clipboard:
                    # only write to clipboard if content changed
                    glfw.glfwSetClipboardString(main_window, user_input.clipboard.encode())

                for button, action, mods in user_input.buttons:
                    x, y = glfw.glfwGetCursorPos(main_window)
                    pos = x * hdpi_factor, y * hdpi_factor
                    pos = normalize(pos, camera_render_size)
                    # Position in img pixels
                    pos = denormalize(pos, g_pool.capture.frame_size)
                    for p in g_pool.plugins:
                        p.on_click(pos, button, action)

                for key, scancode, action, mods in user_input.keys:
                    for p in g_pool.plugins:
                        p.on_key(key, scancode, action, mods)

                for char_ in user_input.chars:
                    for p in g_pool.plugins:
                        p.on_char(char_)

                glfw.glfwSwapBuffers(main_window)
            glfw.glfwPollEvents()

        glfw.glfwRestoreWindow(main_window)  # need to do this for windows os
        session_settings['loaded_plugins'] = g_pool.plugins.get_initializers()
        session_settings['gui_scale'] = g_pool.gui_user_scale
        session_settings['ui_config'] = g_pool.gui.configuration
        session_settings['window_position'] = glfw.glfwGetWindowPos(main_window)
        session_settings['version'] = str(g_pool.version)
        session_settings['eye0_process_alive'] = eyes_are_alive[0].value
        session_settings['eye1_process_alive'] = eyes_are_alive[1].value
        session_settings['min_calibration_confidence'] = g_pool.min_calibration_confidence
        session_settings['detection_mapping_mode'] = g_pool.detection_mapping_mode
        session_settings['audio_mode'] = audio.audio_mode

        session_window_size = glfw.glfwGetWindowSize(main_window)
        if 0 not in session_window_size:
            session_settings['window_size'] = session_window_size

        session_settings.close()

        # de-init all running plugins
        for p in g_pool.plugins:
            p.alive = False
        g_pool.plugins.clean()

        g_pool.gui.terminate()
        glfw.glfwDestroyWindow(main_window)
        glfw.glfwTerminate()

    except:
        import traceback
        trace = traceback.format_exc()
        logger.error('Process Capture crashed with trace:\n{}'.format(trace))

    finally:
        # shut down eye processes:
        stop_eye_process(0)
        stop_eye_process(1)

        logger.info("Process shutting down.")
        ipc_pub.notify({'subject': 'world_process.stopped'})
        sleep(1.0)


def world_profiled(timebase, eyes_are_alive, ipc_pub_url, ipc_sub_url,
                   ipc_push_url, user_dir, version):
    import cProfile
    import subprocess
    import os
    from .world import world
    cProfile.runctx("world(timebase, eyes_are_alive, ipc_pub_url,ipc_sub_url,ipc_push_url,user_dir,version)",
                    {'timebase': timebase, 'eyes_are_alive': eyes_are_alive, 'ipc_pub_url': ipc_pub_url,
                     'ipc_sub_url': ipc_sub_url, 'ipc_push_url': ipc_push_url, 'user_dir': user_dir,
                     'version': version}, locals(), "world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(
        loc[0], 'pupil_src', 'shared_modules', 'gprof2dot.py')
    subprocess.call("python " + gprof2dot_loc + " -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print("created cpu time graph for world process. Please check out the png next to the world.py file")
