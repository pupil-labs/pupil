'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
import sys
import os
import platform


class Global_Container(object):
    pass


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


def player(rec_dir, ipc_pub_url, ipc_sub_url,
           ipc_push_url, user_dir, app_version):
    # general imports
    import logging
    import errno
    from glob import glob
    from copy import deepcopy
    from time import time
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



    # imports
    from file_methods import Persistent_Dict, load_object
    import numpy as np

    # display
    import glfw
    # check versions for our own depedencies as they are fast-changing
    from pyglui import __version__ as pyglui_version

    from pyglui import ui, graph, cygl
    from pyglui.cygl.utils import Named_Texture
    import gl_utils
    # capture
    from video_capture import File_Source, EndofVideoFileError, FileSeekError

    # helpers/utils
    from version_utils import VersionFormat
    from methods import normalize, denormalize, delta_t, get_system_info
    from player_methods import correlate_data, is_pupil_rec_dir, load_meta_info

    # monitoring
    import psutil

    # Plug-ins
    from plugin import Plugin, Plugin_List, import_runtime_plugins, Visualizer_Plugin_Base, Analysis_Plugin_Base, Producer_Plugin_Base
    from vis_circle import Vis_Circle
    from vis_cross import Vis_Cross
    from vis_polyline import Vis_Polyline
    from vis_light_points import Vis_Light_Points
    from vis_watermark import Vis_Watermark
    from vis_fixation import Vis_Fixation
    from vis_scan_path import Vis_Scan_Path
    from vis_eye_video_overlay import Vis_Eye_Video_Overlay
    from seek_bar import Seek_Bar
    from trim_marks import Trim_Marks
    from video_export_launcher import Video_Export_Launcher
    from offline_surface_tracker import Offline_Surface_Tracker
    from marker_auto_trim_marks import Marker_Auto_Trim_Marks
    from fixation_detector import Gaze_Position_2D_Fixation_Detector, Pupil_Angle_3D_Fixation_Detector
    # from manual_gaze_correction import Manual_Gaze_Correction
    from batch_exporter import Batch_Exporter
    from log_display import Log_Display
    from annotations import Annotation_Player
    from raw_data_exporter import Raw_Data_Exporter
    from log_history import Log_History
    from pupil_producers import Pupil_From_Recording, Offline_Pupil_Detection
    from gaze_producers import Gaze_From_Recording, Offline_Calibration

    assert pyglui_version >= '1.5'

    runtime_plugins = import_runtime_plugins(os.path.join(user_dir, 'plugins'))
    system_plugins = [Log_Display, Seek_Bar, Trim_Marks]
    user_launchable_plugins = [Vis_Circle, Vis_Fixation, Vis_Polyline, Vis_Light_Points, Vis_Cross, Vis_Watermark,
                               Vis_Eye_Video_Overlay, Vis_Scan_Path, Gaze_Position_2D_Fixation_Detector,
                               Pupil_Angle_3D_Fixation_Detector, Video_Export_Launcher,
                               Offline_Surface_Tracker, Raw_Data_Exporter, Batch_Exporter, Annotation_Player,
                               Log_History, Marker_Auto_Trim_Marks, Pupil_From_Recording, Offline_Pupil_Detection,
                               Gaze_From_Recording, Offline_Calibration] + runtime_plugins

    available_plugins = system_plugins + user_launchable_plugins
    name_by_index = [p.__name__ for p in available_plugins]
    plugin_by_name = dict(zip(name_by_index, available_plugins))

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

    def on_key(window, key, scancode, action, mods):
        g_pool.gui.update_key(key, scancode, action, mods)

    def on_char(window, char):
        g_pool.gui.update_char(char)

    def on_button(window, button, action, mods):
        g_pool.gui.update_button(button, action, mods)

    def on_pos(window, x, y):
        hdpi_factor = float(glfw.glfwGetFramebufferSize(window)[0]/glfw.glfwGetWindowSize(window)[0])
        g_pool.gui.update_mouse(x*hdpi_factor, y*hdpi_factor)

    def on_scroll(window, x, y):
        g_pool.gui.update_scroll(x, y*scroll_factor)

    def on_drop(window, count, paths):
        for x in range(count):
            new_rec_dir = paths[x].decode('utf-8')
            if is_pupil_rec_dir(new_rec_dir):
                logger.debug("Starting new session with '{}'".format(new_rec_dir))
                ipc_pub.notify({"subject": "player_drop_process.should_start", "rec_dir": new_rec_dir})
                glfw.glfwSetWindowShouldClose(window, True)
            else:
                logger.error("'{}' is not a valid pupil recording".format(new_rec_dir))

    tick = delta_t()

    def get_dt():
        return next(tick)

    video_path = [f for f in glob(os.path.join(rec_dir, "world.*"))
                  if os.path.splitext(f)[1] in ('.mp4', '.mkv', '.avi', '.h264', '.mjpeg')][0]
    timestamps_path = os.path.join(rec_dir, "world_timestamps.npy")
    pupil_data_path = os.path.join(rec_dir, "pupil_data")

    meta_info = load_meta_info(rec_dir)

    # log info about Pupil Platform and Platform in player.log
    logger.info('Application Version: {}'.format(app_version))
    logger.info('System Info: {}'.format(get_system_info()))

    timestamps = np.load(timestamps_path)

    # create container for globally scoped vars
    g_pool = Global_Container()
    g_pool.app = 'player'
    g_pool.zmq_ctx = zmq_ctx
    g_pool.ipc_pub = ipc_pub
    g_pool.ipc_pub_url = ipc_pub_url
    g_pool.ipc_sub_url = ipc_sub_url
    g_pool.ipc_push_url = ipc_push_url

    # Initialize capture
    cap = File_Source(g_pool, video_path, timestamps=list(timestamps))

    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(user_dir, "user_settings"))
    if VersionFormat(session_settings.get("version", '0.0')) != app_version:
        logger.info("Session setting are a different version of this app. I will not use those.")
        session_settings.clear()

    width, height = session_settings.get('window_size', cap.frame_size)
    window_pos = session_settings.get('window_position', window_position_default)
    glfw.glfwInit()
    main_window = glfw.glfwCreateWindow(width, height, "Pupil Player: "+meta_info["Recording Name"]+" - "
                                   + rec_dir.split(os.path.sep)[-1], None, None)
    glfw.glfwSetWindowPos(main_window, window_pos[0], window_pos[1])
    glfw.glfwMakeContextCurrent(main_window)
    cygl.utils.init()

    def set_scale(new_scale):
        g_pool.gui_user_scale = new_scale
        on_resize(main_window, *glfw.glfwGetFramebufferSize(main_window))

    # load pupil_positions, gaze_positions
    g_pool.pupil_data = load_object(pupil_data_path)
    g_pool.binocular = meta_info.get('Eye Mode', 'monocular') == 'binocular'
    g_pool.version = app_version
    g_pool.capture = cap
    g_pool.timestamps = timestamps
    g_pool.get_timestamp = lambda: 0.
    g_pool.play = False
    g_pool.new_seek = True
    g_pool.user_dir = user_dir
    g_pool.rec_dir = rec_dir
    g_pool.meta_info = meta_info
    g_pool.min_data_confidence = session_settings.get('min_data_confidence', 0.6)

    g_pool.pupil_positions = []
    g_pool.gaze_positions = []
    g_pool.fixations = []

    g_pool.notifications_by_frame = correlate_data(g_pool.pupil_data['notifications'], g_pool.timestamps)
    g_pool.pupil_positions_by_frame = [[] for x in g_pool.timestamps] # populated by producer`
    g_pool.gaze_positions_by_frame = [[] for x in g_pool.timestamps] # populated by producer
    g_pool.fixations_by_frame = [[] for x in g_pool.timestamps]  # populated by the fixation detector plugin

    def next_frame(_):
        try:
            cap.seek_to_frame(cap.get_frame_index() + 1)
        except(FileSeekError):
            logger.warning("Could not seek to next frame.")
        else:
            g_pool.new_seek = True

    def prev_frame(_):
        try:
            cap.seek_to_frame(cap.get_frame_index() - 1)
        except(FileSeekError):
            logger.warning("Could not seek to previous frame.")
        else:
            g_pool.new_seek = True

    def toggle_play(new_state):
        if cap.get_frame_index() >= cap.get_frame_count()-5:
            cap.seek_to_frame(1)  # avoid pause set by hitting trimmark pause.
            logger.warning("End of video - restart at beginning.")
        g_pool.play = new_state

    def set_data_confidence(new_confidence):
        g_pool.min_data_confidence = new_confidence
        notification = {'subject': 'min_data_confidence_changed'}
        notification['_notify_time_'] = time()+.8
        g_pool.ipc_pub.notify(notification)

    def open_plugin(plugin):
        if plugin == "Select to load":
            return
        g_pool.plugins.add(plugin)

    def purge_plugins():
        for p in g_pool.plugins:
            if p.__class__ in user_launchable_plugins:
                p.alive = False
        g_pool.plugins.clean()

    def do_export(_):
        export_range = g_pool.trim_marks.in_mark, g_pool.trim_marks.out_mark
        export_dir = os.path.join(g_pool.rec_dir, 'exports', '{}-{}'.format(*export_range))
        try:
            os.makedirs(export_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                logger.error("Could not create export dir")
                raise e
            else:
                overwrite_warning = "Previous export for range [{}-{}] already exsits - overwriting."
                logger.warning(overwrite_warning.format(*export_range))
        else:
            logger.info('Created export dir at "{}"'.format(export_dir))

        notification = {'subject': 'should_export', 'range': export_range, 'export_dir': export_dir}
        g_pool.ipc_pub.notify(notification)

    g_pool.gui = ui.UI()
    g_pool.gui_user_scale = session_settings.get('gui_scale', 1.)
    g_pool.main_menu = ui.Scrolling_Menu("Settings", pos=(-350, 20), size=(300, 560))
    g_pool.main_menu.append(ui.Button('Reset window size',
                                      lambda: glfw.glfwSetWindowSize(main_window, cap.frame_size[0], cap.frame_size[1])))
    g_pool.main_menu.append(ui.Selector('gui_user_scale', g_pool, setter=set_scale, selection=[.8, .9, 1., 1.1, 1.2], label='Interface Size'))
    g_pool.main_menu.append(ui.Info_Text('Player Version: {}'.format(g_pool.version)))
    g_pool.main_menu.append(ui.Info_Text('Capture Version: {}'.format(meta_info['Capture Software Version'])))
    g_pool.main_menu.append(ui.Info_Text('Data Format Version: {}'.format(meta_info['Data Format Version'])))
    g_pool.main_menu.append(ui.Slider('min_data_confidence', g_pool, setter=set_data_confidence,
                                      step=.05, min=0.0, max=1.0, label='Confidence threshold'))

    g_pool.main_menu.append(ui.Info_Text('Open plugins'))

    selector_label = "Select to load"

    def append_selector(label, plugins):
        plugins.sort(key=lambda p: p.__name__)
        plugin_labels = [p.__name__.replace('_', ' ') for p in plugins]
        g_pool.main_menu.append(ui.Selector(label,
                                            selection=[selector_label] + plugins,
                                            labels=[selector_label] + plugin_labels,
                                            setter=open_plugin,
                                            getter=lambda: selector_label))

    base_plugins = [Visualizer_Plugin_Base, Analysis_Plugin_Base, Producer_Plugin_Base]
    base_labels = ['Visualizer:', 'Analyser:', 'Data Source:']
    launchable = user_launchable_plugins.copy()
    for base_class, label in zip(base_plugins, base_labels):
        member_plugins = []
        for p in user_launchable_plugins:
            if issubclass(p, base_class):
                member_plugins.append(p)
                launchable.remove(p)
        append_selector(label, member_plugins)

    # launchable only contains plugins that could not be assigned to any of the above categories
    append_selector('Other', launchable)

    g_pool.main_menu.append(ui.Button('Close all plugins', purge_plugins))
    g_pool.quickbar = ui.Stretching_Menu('Quick Bar', (0, 100), (120, -100))
    g_pool.play_button = ui.Thumb('play',
                                  g_pool,
                                  label=chr(0xf04b),
                                  setter=toggle_play,
                                  hotkey=glfw.GLFW_KEY_SPACE,
                                  label_font='fontawesome',
                                  label_offset_x=5,
                                  label_offset_y=0,
                                  label_offset_size=-24)
    g_pool.play_button.on_color[:] = (0, 1., .0, .8)
    g_pool.forward_button = ui.Thumb('forward',
                                     label=chr(0xf04e),
                                     getter=lambda: False,
                                     setter=next_frame,
                                     hotkey=glfw.GLFW_KEY_RIGHT,
                                     label_font='fontawesome',
                                     label_offset_x=5,
                                     label_offset_y=0,
                                     label_offset_size=-24)
    g_pool.backward_button = ui.Thumb('backward',
                                      label=chr(0xf04a),
                                      getter=lambda: False,
                                      setter=prev_frame,
                                      hotkey=glfw.GLFW_KEY_LEFT,
                                      label_font='fontawesome',
                                      label_offset_x=-5,
                                      label_offset_y=0,
                                      label_offset_size=-24)
    g_pool.export_button = ui.Thumb('export',
                                    label=chr(0xf063),
                                    getter=lambda: False,
                                    setter=do_export,
                                    hotkey='e',
                                    label_font='fontawesome',
                                    label_offset_x=0,
                                    label_offset_y=2,
                                    label_offset_size=-24)
    g_pool.quickbar.extend([g_pool.play_button, g_pool.forward_button, g_pool.backward_button, g_pool.export_button])
    g_pool.gui.append(g_pool.quickbar)
    g_pool.gui.append(g_pool.main_menu)

    # we always load these plugins
    system_plugins = [('Trim_Marks', {}), ('Seek_Bar', {})]
    default_plugins = [('Log_Display', {}), ('Vis_Scan_Path', {}), ('Vis_Polyline', {}),
                       ('Vis_Circle', {}), ('Video_Export_Launcher', {}),
                       ('Pupil_From_Recording', {}), ('Gaze_From_Recording', {})]
    previous_plugins = session_settings.get('loaded_plugins', default_plugins)
    g_pool.plugins = Plugin_List(g_pool, plugin_by_name, system_plugins+previous_plugins)


    # Register callbacks main_window
    glfw.glfwSetFramebufferSizeCallback(main_window, on_resize)
    glfw.glfwSetKeyCallback(main_window, on_key)
    glfw.glfwSetCharCallback(main_window, on_char)
    glfw.glfwSetMouseButtonCallback(main_window, on_button)
    glfw.glfwSetCursorPosCallback(main_window, on_pos)
    glfw.glfwSetScrollCallback(main_window, on_scroll)
    glfw.glfwSetDropCallback(main_window, on_drop)

    g_pool.gui.configuration = session_settings.get('ui_config', {})

    # gl_state settings
    gl_utils.basic_gl_setup()
    g_pool.image_tex = Named_Texture()

    # set up performace graphs:
    pid = os.getpid()
    ps = psutil.Process(pid)
    ts = None

    cpu_graph = graph.Bar_Graph()
    cpu_graph.pos = (20, 110)
    cpu_graph.update_fn = ps.cpu_percent
    cpu_graph.update_rate = 5
    cpu_graph.label = 'CPU %0.1f'

    fps_graph = graph.Bar_Graph()
    fps_graph.pos = (140, 110)
    fps_graph.update_rate = 5
    fps_graph.label = "%0.0f REC FPS"

    pupil_graph = graph.Bar_Graph(max_val=1.0)
    pupil_graph.pos = (260, 110)
    pupil_graph.update_rate = 5
    pupil_graph.label = "Confidence: %0.2f"
    g_pool.graphs = [cpu_graph, fps_graph, pupil_graph]

    # trigger on_resize
    on_resize(main_window, *glfw.glfwGetFramebufferSize(main_window))

    def handle_notifications(n):
        subject = n['subject']
        if subject == 'start_plugin':
            g_pool.plugins.add(
                plugin_by_name[n['name']], args=n.get('args', {}))
        elif subject.startswith('meta.should_doc'):
            ipc_pub.notify({'subject': 'meta.doc',
                            'actor': g_pool.app,
                            'doc': player.__doc__})
            for p in g_pool.plugins:
                if (p.on_notify.__doc__
                        and p.__class__.on_notify != Plugin.on_notify):
                    ipc_pub.notify({'subject': 'meta.doc',
                                    'actor': p.class_name,
                                    'doc': p.on_notify.__doc__})

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

        # grab new frame
        if g_pool.play or g_pool.new_seek:
            g_pool.new_seek = False
            try:
                new_frame = cap.get_frame()
            except EndofVideoFileError:
                # end of video logic: pause at last frame.
                g_pool.play = False
                logger.warning("end of video")
            update_graph = True
        else:
            update_graph = False

        frame = new_frame.copy()
        events = {}
        events['frame'] = frame
        # report time between now and the last loop interation
        events['dt'] = get_dt()
        # new positons we make a deepcopy just like the image is a copy.
        events['gaze_positions'] = deepcopy(g_pool.gaze_positions_by_frame[frame.index])
        events['pupil_positions'] = deepcopy(g_pool.pupil_positions_by_frame[frame.index])

        if update_graph:
            # update performace graphs
            for p in events['pupil_positions']:
                pupil_graph.add(p['confidence'])

            t = new_frame.timestamp
            if ts and ts != t:
                dt, ts = t-ts, t
                fps_graph.add(1./dt)
            else:
                ts = new_frame.timestamp

            g_pool.play_button.status_text = str(frame.index)
        # always update the CPU graph
        cpu_graph.update()

        # allow each Plugin to do its work.
        for p in g_pool.plugins:
            p.recent_events(events)

        # check if a plugin need to be destroyed
        g_pool.plugins.clean()

        # render camera image
        glfw.glfwMakeContextCurrent(main_window)
        gl_utils.make_coord_system_norm_based()
        g_pool.image_tex.update_from_frame(frame)
        g_pool.image_tex.draw()
        gl_utils.make_coord_system_pixel_based(frame.img.shape)
        # render visual feedback from loaded plugins
        for p in g_pool.plugins:
            p.gl_display()

        fps_graph.draw()
        cpu_graph.draw()
        pupil_graph.draw()
        unused_buttons = g_pool.gui.update()
        for b in unused_buttons:
            button,action,mods = b
            pos = glfw.glfwGetCursorPos(main_window)
            pos = normalize(pos, glfw.glfwGetWindowSize(main_window))
            pos = denormalize(pos, (frame.img.shape[1], frame.img.shape[0]))  # Position in img pixels
            for p in g_pool.plugins:
                p.on_click(pos, button, action)

        # present frames at appropriate speed
        cap.wait(frame)

        glfw.glfwSwapBuffers(main_window)
        glfw.glfwPollEvents()

    session_settings['loaded_plugins'] = g_pool.plugins.get_initializers()
    session_settings['min_data_confidence'] = g_pool.min_data_confidence
    session_settings['gui_scale'] = g_pool.gui_user_scale
    session_settings['ui_config'] = g_pool.gui.configuration
    session_settings['window_size'] = glfw.glfwGetWindowSize(main_window)
    session_settings['window_position'] = glfw.glfwGetWindowPos(main_window)
    session_settings['version'] = str(g_pool.version)
    session_settings.close()

    # de-init all running plugins
    for p in g_pool.plugins:
        p.alive = False
    g_pool.plugins.clean()

    cap.cleanup()
    g_pool.gui.terminate()
    glfw.glfwDestroyWindow(main_window)

    logger.info("Process shutting down.")
    ipc_pub.notify({'subject': 'player_process.stopped'})


def player_drop(rec_dir, ipc_pub_url, ipc_sub_url,
                ipc_push_url, user_dir, app_version):
    # general imports
    import logging
    # networking
    import zmq
    import zmq_tools
    from time import sleep


    # zmq ipc setup
    zmq_ctx = zmq.Context()
    ipc_pub = zmq_tools.Msg_Dispatcher(zmq_ctx, ipc_push_url)

    # log setup
    logging.getLogger("OpenGL").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    import glfw
    import gl_utils
    from OpenGL.GL import glClearColor
    from version_utils import VersionFormat
    from file_methods import Persistent_Dict
    from pyglui.pyfontstash import fontstash
    from pyglui.ui import get_roboto_font_path
    from player_methods import is_pupil_rec_dir, update_recording_to_recent

    def on_drop(window, count, paths):
        nonlocal rec_dir
        rec_dir = paths[0].decode('utf-8')

    if rec_dir:
        if not is_pupil_rec_dir(rec_dir):
            rec_dir = None
    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(user_dir, "user_settings"))
    if VersionFormat(session_settings.get("version", '0.0')) != app_version:
        logger.info("Session setting are from a  different version of this app. I will not use those.")
        session_settings.clear()
    w, h = session_settings.get('window_size', (1280, 720))
    window_pos = session_settings.get('window_position', window_position_default)

    glfw.glfwInit()
    glfw.glfwWindowHint(glfw.GLFW_RESIZABLE, 0)
    window = glfw.glfwCreateWindow(w, h, 'Pupil Player')
    glfw.glfwWindowHint(glfw.GLFW_RESIZABLE, 1)

    glfw.glfwMakeContextCurrent(window)
    glfw.glfwSetWindowPos(window, window_pos[0], window_pos[1])
    glfw.glfwSetDropCallback(window, on_drop)

    glfont = fontstash.Context()
    glfont.add_font('roboto', get_roboto_font_path())
    glfont.set_align_string(v_align="center", h_align="middle")
    glfont.set_color_float((0.2, 0.2, 0.2, 0.9))
    gl_utils.basic_gl_setup()
    glClearColor(0.5, .5, 0.5, 0.0)
    text = 'Drop a recording directory onto this window.'
    tip = '(Tip: You can drop a recording directory onto the app icon.)'
    # text = "Please supply a Pupil recording directory as first arg when calling Pupil Player."
    while not glfw.glfwWindowShouldClose(window):

        fb_size = glfw.glfwGetFramebufferSize(window)
        hdpi_factor = float(fb_size[0] / glfw.glfwGetWindowSize(window)[0])
        gl_utils.adjust_gl_view(*fb_size)

        if rec_dir:
            if is_pupil_rec_dir(rec_dir):
                logger.info("Starting new session with '{}'".format(rec_dir))
                text = "Updating recording format."
                tip = "This may take a while!"
            else:
                logger.error("'{}' is not a valid pupil recording".format(rec_dir))
                tip = "Oops! That was not a valid recording."
                rec_dir = None

        gl_utils.clear_gl_screen()
        glfont.set_blur(10.5)
        glfont.set_color_float((0.0, 0.0, 0.0, 1.))
        glfont.set_size(w/25.*hdpi_factor)
        glfont.draw_text(w/2*hdpi_factor, .3*h*hdpi_factor, text)
        glfont.set_size(w/30.*hdpi_factor)
        glfont.draw_text(w/2*hdpi_factor, .4*h*hdpi_factor, tip)
        glfont.set_blur(0.96)
        glfont.set_color_float((1., 1., 1., 1.))
        glfont.set_size(w/25.*hdpi_factor)
        glfont.draw_text(w/2*hdpi_factor, .3*h*hdpi_factor, text)
        glfont.set_size(w/30.*hdpi_factor)
        glfont.draw_text(w/2*hdpi_factor, .4*h*hdpi_factor, tip)

        glfw.glfwSwapBuffers(window)

        if rec_dir:
            update_recording_to_recent(rec_dir)
            glfw.glfwSetWindowShouldClose(window, True)

        glfw.glfwPollEvents()

    session_settings['window_position'] = glfw.glfwGetWindowPos(window)
    session_settings.close()
    glfw.glfwDestroyWindow(window)
    if rec_dir:
        ipc_pub.notify({"subject": "player_process.should_start", "rec_dir": rec_dir})
    sleep(1.0)
