'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

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
    logger.setLevel(logging.INFO)
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    # display
    import glfw
    from pyglui import ui, graph, cygl, __version__ as pyglui_version
    assert pyglui_version >= '1.3'
    from pyglui.cygl.utils import Named_Texture
    import gl_utils

    # monitoring
    import psutil

    # helpers/utils
    from version_utils import VersionFormat
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
    from plugin import Plugin, Plugin_List, import_runtime_plugins
    from calibration_routines import calibration_plugins, gaze_mapping_plugins, Calibration_Plugin
    from fixation_detector import Fixation_Detector_3D
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
    from video_capture import source_classes, manager_classes,Base_Manager
    from pupil_data_relay import Pupil_Data_Relay
    from remote_recorder import Remote_Recorder
    from audio_capture import Audio_Capture

    # UI Platform tweaks
    if platform.system() == 'Linux':
        scroll_factor = 10.0
        window_position_default = (0, 0)
    elif platform.system() == 'Windows':
        scroll_factor = 10.0
        window_position_default = (8, 31)
    else:
        scroll_factor = 1.0
        window_position_default = (0, 0)

    # g_pool holds variables for this process they are accesible to all plugins
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
    calibration_plugins += [p for p in runtime_plugins if issubclass(p, Calibration_Plugin)]
    runtime_plugins = [p for p in runtime_plugins if not issubclass(p, Calibration_Plugin)]
    manager_classes += [p for p in runtime_plugins if issubclass(p, Base_Manager)]
    runtime_plugins = [p for p in runtime_plugins if not issubclass(p, Base_Manager)]
    user_launchable_plugins = [Audio_Capture, Pupil_Groups, Frame_Publisher, Pupil_Remote, Time_Sync, Surface_Tracker,
                               Annotation_Capture, Log_History, Fixation_Detector_3D, Blink_Detection,
                               Remote_Recorder] + runtime_plugins
    system_plugins = [Log_Display, Display_Recent_Gaze, Recorder, Pupil_Data_Relay]
    plugin_by_index = (system_plugins + user_launchable_plugins + calibration_plugins
                       + gaze_mapping_plugins + manager_classes + source_classes)
    name_by_index = [p.__name__ for p in plugin_by_index]
    plugin_by_name = dict(zip(name_by_index, plugin_by_index))

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
                       ('Pupil_Remote', {})]

    # Callback functions
    def on_resize(window, w, h):
        if gl_utils.is_window_visible(window):
            hdpi_factor = float(glfw.glfwGetFramebufferSize(window)[0] / glfw.glfwGetWindowSize(window)[0])
            g_pool.gui.scale = g_pool.gui_user_scale * hdpi_factor
            g_pool.gui.update_window(w, h)
            g_pool.gui.collect_menus()
            for g in g_pool.graphs:
                g.scale = hdpi_factor
                g.adjust_window_size(w, h)
            gl_utils.adjust_gl_view(w, h)
            for p in g_pool.plugins:
                p.on_window_resize(window, w, h)

    def on_iconify(window, iconified):
        g_pool.iconified = iconified

    def on_key(window, key, scancode, action, mods):
        g_pool.gui.update_key(key, scancode, action, mods)

    def on_char(window, char):
        g_pool.gui.update_char(char)

    def on_button(window, button, action, mods):
        g_pool.gui.update_button(button, action, mods)
        pos = glfw.glfwGetCursorPos(window)
        pos = normalize(pos, glfw.glfwGetWindowSize(main_window))
        # Position in img pixels
        pos = denormalize(pos, g_pool.capture.frame_size)
        for p in g_pool.plugins:
            p.on_click(pos, button, action)

    def on_pos(window, x, y):
        hdpi_factor = float(glfw.glfwGetFramebufferSize(
            window)[0] / glfw.glfwGetWindowSize(window)[0])
        x, y = x * hdpi_factor, y * hdpi_factor
        g_pool.gui.update_mouse(x, y)

    def on_scroll(window, x, y):
        g_pool.gui.update_scroll(x, y * scroll_factor)

    tick = delta_t()

    def get_dt():
        return next(tick)

    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(g_pool.user_dir, 'user_settings_world'))
    if VersionFormat(session_settings.get("version", '0.0')) != g_pool.version:
        logger.info("Session setting are from a different version of this app. I will not use those.")
        session_settings.clear()

    g_pool.iconified = False
    g_pool.detection_mapping_mode = session_settings.get('detection_mapping_mode', '3d')
    g_pool.active_calibration_plugin = None
    g_pool.active_gaze_mapping_plugin = None
    g_pool.capture_manager = None

    audio.audio_mode = session_settings.get('audio_mode', audio.default_audio_mode)

    def open_plugin(plugin):
        if plugin == "Select to load":
            return
        g_pool.plugins.add(plugin)

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

    def handle_notifications(n):
        subject = n['subject']
        if subject == 'set_detection_mapping_mode':
            if n['mode'] == '2d':
                if ("Vector_Gaze_Mapper" in
                        g_pool.active_gaze_mapping_plugin.class_name):
                    logger.warning("The gaze mapper is not supported in 2d mode. Please recalibrate.")
                    g_pool.plugins.add(plugin_by_name['Dummy_Gaze_Mapper'])
            g_pool.detection_mapping_mode = n['mode']
        elif subject == 'start_plugin':
            g_pool.plugins.add(
                plugin_by_name[n['name']], args=n.get('args', {}))
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

    # window and gl setup
    glfw.glfwInit()
    width, height = session_settings.get('window_size', (1280, 720))
    main_window = glfw.glfwCreateWindow(width, height, "Pupil Capture - World")
    window_pos = session_settings.get('window_position', window_position_default)
    glfw.glfwSetWindowPos(main_window, window_pos[0], window_pos[1])
    glfw.glfwMakeContextCurrent(main_window)
    cygl.utils.init()
    g_pool.main_window = main_window

    def set_scale(new_scale):
        g_pool.gui_user_scale = new_scale
        on_resize(main_window, *glfw.glfwGetFramebufferSize(main_window))

    def reset_restart():
        logger.warning("Resetting all settings and restarting Capture.")
        glfw.glfwSetWindowShouldClose(main_window, True)
        ipc_pub.notify({'subject': 'reset_restart_process.should_start'})

    # setup GUI
    g_pool.gui = ui.UI()
    g_pool.gui_user_scale = session_settings.get('gui_scale', 1.)
    g_pool.sidebar = ui.Scrolling_Menu("Settings", pos=(-350, 0), size=(0, 0), header_pos='left')
    general_settings = ui.Growing_Menu('General')
    general_settings.append(ui.Button('Reset to default settings',reset_restart))
    general_settings.append(ui.Selector('gui_user_scale', g_pool, setter=set_scale, selection=[.8, .9, 1., 1.1, 1.2], label='Interface size'))
    general_settings.append(ui.Button('Reset window size', lambda: glfw.glfwSetWindowSize(main_window,g_pool.capture.frame_size[0],g_pool.capture.frame_size[1])) )
    general_settings.append(ui.Selector('audio_mode', audio, selection=audio.audio_modes))
    general_settings.append(ui.Selector('detection_mapping_mode',
                                        g_pool,
                                        label='detection & mapping mode',
                                        setter=set_detection_mapping_mode,
                                        selection=['2d','3d']
                                    ))
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
    selector_label = "Select to load"
    labels = [p.__name__.replace('_', ' ') for p in user_launchable_plugins]
    user_launchable_plugins.insert(0, selector_label)
    labels.insert(0, selector_label)
    general_settings.append(ui.Selector('Open plugin',
                                        selection=user_launchable_plugins,
                                        labels=labels,
                                        setter=open_plugin,
                                        getter=lambda: selector_label))

    general_settings.append(ui.Info_Text('Capture Version: {}'.format(g_pool.version)))

    g_pool.quickbar = ui.Stretching_Menu('Quick Bar', (0, 100), (120, -100))

    g_pool.capture_source_menu = ui.Growing_Menu('Capture Source')
    g_pool.capture_source_menu.collapsed = True

    g_pool.calibration_menu = ui.Growing_Menu('Calibration')
    g_pool.calibration_menu.collapsed = True
    g_pool.capture_selector_menu = ui.Growing_Menu('Capture Selection')

    g_pool.sidebar.append(general_settings)
    g_pool.sidebar.append(g_pool.capture_selector_menu)
    g_pool.sidebar.append(g_pool.capture_source_menu)
    g_pool.sidebar.append(g_pool.calibration_menu)

    g_pool.gui.append(g_pool.sidebar)
    g_pool.gui.append(g_pool.quickbar)

    # plugins that are loaded based on user settings from previous session
    g_pool.plugins = Plugin_List(g_pool, plugin_by_name, session_settings.get('loaded_plugins', default_plugins))

    #We add the calibration menu selector, after a calibration has been added:
    g_pool.calibration_menu.insert(0,ui.Selector(
                        'active_calibration_plugin',
                        getter=lambda: g_pool.active_calibration_plugin.__class__,
                        selection = calibration_plugins,
                        labels = [p.__name__.replace('_',' ') for p in calibration_plugins],
                        setter= open_plugin,label='Method'
                                ))

    #We add the capture selection menu, after a manager has been added:
    g_pool.capture_selector_menu.insert(0,ui.Selector(
                            'capture_manager',
                            setter    = open_plugin,
                            getter    = lambda: g_pool.capture_manager.__class__,
                            selection = manager_classes,
                            labels    = [b.gui_name for b in manager_classes],
                            label     = 'Manager'
                        ))

    # Register callbacks main_window
    glfw.glfwSetFramebufferSizeCallback(main_window, on_resize)
    glfw.glfwSetWindowIconifyCallback(main_window, on_iconify)
    glfw.glfwSetKeyCallback(main_window, on_key)
    glfw.glfwSetCharCallback(main_window, on_char)
    glfw.glfwSetMouseButtonCallback(main_window, on_button)
    glfw.glfwSetCursorPosCallback(main_window, on_pos)
    glfw.glfwSetScrollCallback(main_window, on_scroll)

    # gl_state settings
    gl_utils.basic_gl_setup()
    g_pool.image_tex = Named_Texture()

    # now the we have  aproper window we can load the last gui configuration
    g_pool.gui.configuration = session_settings.get('ui_config', {})

    # create a timer to control window update frequency
    window_update_timer = timer(1 / 60)
    def window_should_update():
        return next(window_update_timer)

    # set up performace graphs:
    pid = os.getpid()
    ps = psutil.Process(pid)
    ts = g_pool.get_timestamp()

    cpu_graph = graph.Bar_Graph()
    cpu_graph.pos = (20, 130)
    cpu_graph.update_fn = ps.cpu_percent
    cpu_graph.update_rate = 5
    cpu_graph.label = 'CPU %0.1f'

    fps_graph = graph.Bar_Graph()
    fps_graph.pos = (140, 130)
    fps_graph.update_rate = 5
    fps_graph.label = "%0.0f FPS"

    pupil0_graph = graph.Bar_Graph(max_val=1.0)
    pupil0_graph.pos = (260, 130)
    pupil0_graph.update_rate = 5
    pupil0_graph.label = "id0 conf: %0.2f"
    pupil1_graph = graph.Bar_Graph(max_val=1.0)
    pupil1_graph.pos = (380, 130)
    pupil1_graph.update_rate = 5
    pupil1_graph.label = "id1 conf: %0.2f"
    pupil_graphs = pupil0_graph, pupil1_graph
    g_pool.graphs = [cpu_graph, fps_graph, pupil0_graph, pupil1_graph]

    # trigger setup of window and gl sizes
    on_resize(main_window, *glfw.glfwGetFramebufferSize(main_window))

    if session_settings.get('eye1_process_alive', False):
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


        #a dictionary that allows plugins to post and read events
        events = {}
        # report time between now and the last loop interation
        events['dt'] = get_dt()

        # allow each Plugin to do its work.
        for p in g_pool.plugins:
            p.recent_events(events)

        # check if a plugin need to be destroyed
        g_pool.plugins.clean()

        # update performace graphs
        if 'frame' in events:
            t = events["frame"].timestamp
            dt, ts = t-ts, t
            try:
                fps_graph.add(1./dt)
            except ZeroDivisionError:
                pass
        for p in events["pupil_positions"]:
            pupil_graphs[p['id']].add(p['confidence'])
        cpu_graph.update()

        # send new events to ipc:
        del events['pupil_positions']  # already on the wire
        del events['gaze_positions']  # sent earlier
        if 'frame' in events:
            del events['frame']  # send explicity with frame publisher
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
            for p in g_pool.plugins:
                p.gl_display()

            for g in g_pool.graphs:
                g.draw()

            g_pool.gui.update()
            glfw.glfwSwapBuffers(main_window)
        glfw.glfwPollEvents()

    glfw.glfwRestoreWindow(main_window)  # need to do this for windows os
    session_settings['loaded_plugins'] = g_pool.plugins.get_initializers()
    session_settings['gui_scale'] = g_pool.gui_user_scale
    session_settings['ui_config'] = g_pool.gui.configuration
    session_settings['window_size'] = glfw.glfwGetWindowSize(main_window)
    session_settings['window_position'] = glfw.glfwGetWindowPos(main_window)
    session_settings['version'] = str(g_pool.version)
    session_settings['eye0_process_alive'] = eyes_are_alive[0].value
    session_settings['eye1_process_alive'] = eyes_are_alive[1].value
    session_settings['detection_mapping_mode'] = g_pool.detection_mapping_mode
    session_settings['audio_mode'] = audio.audio_mode
    session_settings.close()

    # de-init all running plugins
    for p in g_pool.plugins:
        p.alive = False
    g_pool.plugins.clean()

    g_pool.gui.terminate()
    glfw.glfwDestroyWindow(main_window)
    glfw.glfwTerminate()


    # shut down eye processes:
    stop_eye_process(0)
    stop_eye_process(1)

    logger.info("Process shutting down.")
    ipc_pub.notify({'subject': 'world_process.stopped'})



def reset_restart(ipc_push_url,user_dir):
    import glob, os, time

    # networking
    import zmq
    import zmq_tools
    # zmq ipc setup
    zmq_ctx = zmq.Context()
    ipc_pub = zmq_tools.Msg_Dispatcher(zmq_ctx, ipc_push_url)

    time.sleep(1)

    for f in glob.glob(os.path.join(user_dir,'user_settings_*')):
        print(f)
        os.remove(f)

    ipc_pub.notify({'subject': 'world_process.should_start'})
    time.sleep(1)


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
