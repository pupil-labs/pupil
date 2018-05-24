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


def player(rec_dir, ipc_pub_url, ipc_sub_url,
           ipc_push_url, user_dir, app_version):
    # general imports
    from time import sleep
    import logging
    from glob import glob
    from time import time, strftime, localtime
    # networking
    import zmq
    import zmq_tools

    import numpy as np

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

    try:

        # imports
        from file_methods import Persistent_Dict, load_object, next_export_sub_dir

        # display
        import glfw
        # check versions for our own depedencies as they are fast-changing
        from pyglui import __version__ as pyglui_version

        from pyglui import ui, cygl
        from pyglui.cygl.utils import Named_Texture, RGBA
        import gl_utils
        # capture
        from video_capture import init_playback_source, EndofVideoError

        # helpers/utils
        from version_utils import VersionFormat
        from methods import normalize, denormalize, delta_t, get_system_info
        from player_methods import correlate_data, is_pupil_rec_dir, load_meta_info
        from csv_utils import write_key_value_file

        # Plug-ins
        from plugin import Plugin, Plugin_List, import_runtime_plugins
        from plugin_manager import Plugin_Manager
        from vis_circle import Vis_Circle
        from vis_cross import Vis_Cross
        from vis_polyline import Vis_Polyline
        from vis_light_points import Vis_Light_Points
        from vis_watermark import Vis_Watermark
        from vis_fixation import Vis_Fixation
        from vis_scan_path import Vis_Scan_Path
        from vis_eye_video_overlay import Vis_Eye_Video_Overlay
        from seek_control import Seek_Control
        from video_export_launcher import Video_Export_Launcher
        from offline_surface_tracker import Offline_Surface_Tracker
        # from marker_auto_trim_marks import Marker_Auto_Trim_Marks
        from fixation_detector import Offline_Fixation_Detector
        from batch_exporter import Batch_Exporter, Batch_Export
        from log_display import Log_Display
        from annotations import Annotation_Player
        from raw_data_exporter import Raw_Data_Exporter
        from log_history import Log_History
        from pupil_producers import Pupil_From_Recording, Offline_Pupil_Detection
        from gaze_producers import Gaze_From_Recording, Offline_Calibration
        from system_graphs import System_Graphs
        from system_timelines import System_Timelines
        from blink_detection import Offline_Blink_Detection
        from audio_playback import Audio_Playback
        from imotions_exporter import iMotions_Exporter

        assert VersionFormat(pyglui_version) >= VersionFormat('1.22'), 'pyglui out of date, please upgrade to newest version'

        runtime_plugins = import_runtime_plugins(os.path.join(user_dir, 'plugins'))
        system_plugins = [Log_Display, Seek_Control, Plugin_Manager, System_Graphs, Batch_Export, System_Timelines, Audio_Playback]
        user_plugins = [Vis_Circle,
                        Vis_Fixation,
                        Vis_Polyline,
                        Vis_Light_Points,
                        Vis_Cross,
                        Vis_Watermark,
                        Vis_Eye_Video_Overlay,
                        Vis_Scan_Path,
                        Offline_Fixation_Detector,
                        Offline_Blink_Detection,
                        Batch_Exporter,
                        Video_Export_Launcher,
                        Offline_Surface_Tracker,
                        Raw_Data_Exporter,
                        Annotation_Player,
                        Log_History,
                        Pupil_From_Recording,
                        Offline_Pupil_Detection,
                        Gaze_From_Recording,
                        iMotions_Exporter,
                        Offline_Calibration
                        ] + runtime_plugins

        plugins = system_plugins + user_plugins

        # Callback functions
        def on_resize(window, w, h):
            nonlocal window_size
            nonlocal hdpi_factor
            if w == 0 or h == 0:
                return
            hdpi_factor = glfw.getHDPIFactor(window)
            g_pool.gui.scale = g_pool.gui_user_scale * hdpi_factor
            window_size = w, h
            g_pool.camera_render_size = w-int(icon_bar_width*g_pool.gui.scale), h
            g_pool.gui.update_window(*window_size)
            g_pool.gui.collect_menus()
            for p in g_pool.plugins:
                p.on_window_resize(window, *g_pool.camera_render_size)

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
            pos = normalize(pos, g_pool.camera_render_size)
            # Position in img pixels
            pos = denormalize(pos, g_pool.capture.frame_size)
            for p in g_pool.plugins:
                p.on_pos(pos)

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

        pupil_data_path = os.path.join(rec_dir, "pupil_data")

        meta_info = load_meta_info(rec_dir)

        # log info about Pupil Platform and Platform in player.log
        logger.info('Application Version: {}'.format(app_version))
        logger.info('System Info: {}'.format(get_system_info()))

        icon_bar_width = 50
        window_size = None
        hdpi_factor = 1.0

        # create container for globally scoped vars
        g_pool = Global_Container()
        g_pool.app = 'player'
        g_pool.zmq_ctx = zmq_ctx
        g_pool.ipc_pub = ipc_pub
        g_pool.ipc_pub_url = ipc_pub_url
        g_pool.ipc_sub_url = ipc_sub_url
        g_pool.ipc_push_url = ipc_push_url
        g_pool.plugin_by_name = {p.__name__: p for p in plugins}
        g_pool.camera_render_size = None

        valid_ext = ('.mp4', '.mkv', '.avi', '.h264', '.mjpeg', '.fake')
        video_path = [f for f in glob(os.path.join(rec_dir, "world.*"))
                      if os.path.splitext(f)[1] in valid_ext][0]
        init_playback_source(g_pool, timing='external', source_path=video_path)

        # load session persistent settings
        session_settings = Persistent_Dict(os.path.join(user_dir, "user_settings_player"))
        if VersionFormat(session_settings.get("version", '0.0')) != app_version:
            logger.info("Session setting are a different version of this app. I will not use those.")
            session_settings.clear()

        width, height = g_pool.capture.frame_size
        width += icon_bar_width
        width, height = session_settings.get('window_size', (width, height))

        window_pos = session_settings.get('window_position', window_position_default)
        window_name = "Pupil Player: {} - {}".format(meta_info["Recording Name"],
                                                     os.path.split(rec_dir)[-1])

        glfw.glfwInit()
        main_window = glfw.glfwCreateWindow(width, height, window_name, None, None)
        glfw.glfwSetWindowPos(main_window, window_pos[0], window_pos[1])
        glfw.glfwMakeContextCurrent(main_window)
        cygl.utils.init()
        g_pool.main_window = main_window

        def set_scale(new_scale):
            g_pool.gui_user_scale = new_scale
            window_size = (g_pool.camera_render_size[0] + int(icon_bar_width*g_pool.gui_user_scale*hdpi_factor),
                           glfw.glfwGetFramebufferSize(main_window)[1])
            logger.warning(icon_bar_width*g_pool.gui_user_scale*hdpi_factor)
            glfw.glfwSetWindowSize(main_window, *window_size)

        # load pupil_positions, gaze_positions
        g_pool.pupil_data = load_object(pupil_data_path)
        g_pool.binocular = meta_info.get('Eye Mode', 'monocular') == 'binocular'
        g_pool.version = app_version
        g_pool.timestamps = g_pool.capture.timestamps
        g_pool.get_timestamp = lambda: 0.
        g_pool.user_dir = user_dir
        g_pool.rec_dir = rec_dir
        g_pool.meta_info = meta_info
        g_pool.min_data_confidence = session_settings.get('min_data_confidence', 0.6)
        g_pool.min_calibration_confidence = session_settings.get('min_calibration_confidence', 0.8)

        g_pool.pupil_positions = []
        g_pool.gaze_positions = []
        g_pool.fixations = []

        g_pool.notifications_by_frame = correlate_data(g_pool.pupil_data['notifications'], g_pool.timestamps)
        g_pool.pupil_positions_by_frame = [[] for x in g_pool.timestamps]  # populated by producer`
        g_pool.gaze_positions_by_frame = [[] for x in g_pool.timestamps]  # populated by producer
        g_pool.fixations_by_frame = [[] for x in g_pool.timestamps]  # populated by the fixation detector plugin

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
                if p.__class__ in user_plugins:
                    p.alive = False
            g_pool.plugins.clean()

        def do_export(_):
            left_idx = g_pool.seek_control.trim_left
            right_idx = g_pool.seek_control.trim_right
            export_range = left_idx, right_idx + 1  # exclusive range.stop

            export_dir = os.path.join(g_pool.rec_dir, 'exports')
            export_dir = next_export_sub_dir(export_dir)

            os.makedirs(export_dir)
            logger.info('Created export dir at "{}"'.format(export_dir))

            export_info = {'Player Software Version': str(g_pool.version),
                           'Data Format Version': meta_info['Data Format Version'],
                           'Export Date': strftime("%d.%m.%Y", localtime()),
                           'Export Time': strftime("%H:%M:%S", localtime()),
                           'Frame Index Range:': g_pool.seek_control.get_frame_index_trim_range_string(),
                           'Relative Time Range': g_pool.seek_control.get_rel_time_trim_range_string(),
                           'Absolute Time Range': g_pool.seek_control.get_abs_time_trim_range_string()}
            with open(os.path.join(export_dir, 'export_info.csv'), 'w') as csv:
                write_key_value_file(csv, export_info)

            notification = {'subject': 'should_export', 'range': export_range, 'export_dir': export_dir}
            g_pool.ipc_pub.notify(notification)

        def reset_restart():
            logger.warning("Resetting all settings and restarting Player.")
            glfw.glfwSetWindowShouldClose(main_window, True)
            ipc_pub.notify({'subject': 'clear_settings_process.should_start'})
            ipc_pub.notify({'subject': 'player_process.should_start', 'rec_dir': rec_dir, 'delay': 2.})

        def toggle_general_settings(collapsed):
            # this is the menu toggle logic.
            # Only one menu can be open.
            # If no menu is open the menubar should collapse.
            g_pool.menubar.collapsed = collapsed
            for m in g_pool.menubar.elements:
                m.collapsed = True
            general_settings.collapsed = collapsed

        g_pool.gui = ui.UI()
        g_pool.gui_user_scale = session_settings.get('gui_scale', 1.)
        g_pool.menubar = ui.Scrolling_Menu("Settings", pos=(-500, 0), size=(-icon_bar_width, 0), header_pos='left')
        g_pool.iconbar = ui.Scrolling_Menu("Icons", pos=(-icon_bar_width, 0), size=(0, 0), header_pos='hidden')
        g_pool.timelines = ui.Container((0, 0), (0, 0), (0, 0))
        g_pool.timelines.horizontal_constraint = g_pool.menubar
        g_pool.user_timelines = ui.Timeline_Menu('User Timelines', pos=(0., -150.),
                                                 size=(0., 0.), header_pos='headline')
        g_pool.user_timelines.color = RGBA(a=0.)
        g_pool.user_timelines.collapsed = True
        # add container that constaints itself to the seekbar height
        vert_constr = ui.Container((0, 0), (0, -50.), (0, 0))
        vert_constr.append(g_pool.user_timelines)
        g_pool.timelines.append(vert_constr)

        def set_window_size():
            f_width, f_height = g_pool.capture.frame_size
            f_width += int(icon_bar_width * g_pool.gui.scale)
            glfw.glfwSetWindowSize(main_window, f_width, f_height)

        general_settings = ui.Growing_Menu('General', header_pos='headline')
        general_settings.append(ui.Button('Reset window size', set_window_size))
        general_settings.append(ui.Selector('gui_user_scale', g_pool, setter=set_scale, selection=[.8, .9, 1., 1.1, 1.2]+list(np.arange(1.5, 5.1, .5)), label='Interface Size'))
        general_settings.append(ui.Info_Text('Player Version: {}'.format(g_pool.version)))
        general_settings.append(ui.Info_Text('Capture Version: {}'.format(meta_info['Capture Software Version'])))
        general_settings.append(ui.Info_Text('Data Format Version: {}'.format(meta_info['Data Format Version'])))

        general_settings.append(ui.Info_Text('High level data, e.g. fixations, or visualizations only consider gaze data that has an equal or higher confidence than the minimum data confidence.'))
        general_settings.append(ui.Slider('min_data_confidence', g_pool,
                                          setter=set_data_confidence,
                                          step=.05, min=0.0, max=1.0,
                                          label='Minimum data confidence'))

        general_settings.append(ui.Button('Restart with default settings', reset_restart))

        g_pool.menubar.append(general_settings)
        icon = ui.Icon('collapsed', general_settings, label=chr(0xe8b8), on_val=False, off_val=True, setter=toggle_general_settings, label_font='pupil_icons')
        icon.tooltip = 'General Settings'
        g_pool.iconbar.append(icon)

        user_plugin_separator = ui.Separator()
        user_plugin_separator.order = 0.35
        g_pool.iconbar.append(user_plugin_separator)

        g_pool.quickbar = ui.Stretching_Menu('Quick Bar', (0, 100), (100, -100))
        g_pool.export_button = ui.Thumb('export',
                                        label=chr(0xe2c5),
                                        getter=lambda: False,
                                        setter=do_export,
                                        hotkey='e',
                                        label_font='pupil_icons')
        g_pool.quickbar.extend([g_pool.export_button])
        g_pool.gui.append(g_pool.menubar)
        g_pool.gui.append(g_pool.timelines)
        g_pool.gui.append(g_pool.iconbar)
        g_pool.gui.append(g_pool.quickbar)

        # we always load these plugins
        default_plugins = [('Plugin_Manager', {}),
                           ('Seek_Control', {}),
                           ('Log_Display', {}),
                           ('Raw_Data_Exporter', {}),
                           ('Vis_Polyline', {}),
                           ('Vis_Circle', {}),
                           ('System_Graphs', {}),
                           ('System_Timelines', {}),
                           ('Video_Export_Launcher', {}),
                           ('Pupil_From_Recording', {}),
                           ('Gaze_From_Recording', {}),
                           ('Audio_Playback', {})]

        g_pool.plugins = Plugin_List(g_pool, session_settings.get('loaded_plugins', default_plugins))

        # Manually add g_pool.capture to the plugin list
        g_pool.plugins._plugins.append(g_pool.capture)
        g_pool.plugins._plugins.sort(key=lambda p: p.order)
        g_pool.capture.init_ui()

        general_settings.insert(-1, ui.Text_Input('rel_time_trim_section',
                                                  getter=g_pool.seek_control.get_rel_time_trim_range_string,
                                                  setter=g_pool.seek_control.set_rel_time_trim_range_string,
                                                  label='Relative time range to export'))
        general_settings.insert(-1, ui.Text_Input('frame_idx_trim_section',
                                                  getter=g_pool.seek_control.get_frame_index_trim_range_string,
                                                  setter=g_pool.seek_control.set_frame_index_trim_range_string,
                                                  label='Frame index range to export'))

        # Register callbacks main_window
        glfw.glfwSetFramebufferSizeCallback(main_window, on_resize)
        glfw.glfwSetKeyCallback(main_window, on_window_key)
        glfw.glfwSetCharCallback(main_window, on_window_char)
        glfw.glfwSetMouseButtonCallback(main_window, on_window_mouse_button)
        glfw.glfwSetCursorPosCallback(main_window, on_pos)
        glfw.glfwSetScrollCallback(main_window, on_scroll)
        glfw.glfwSetDropCallback(main_window, on_drop)

        toggle_general_settings(True)

        g_pool.gui.configuration = session_settings.get('ui_config', {})

        # gl_state settings
        gl_utils.basic_gl_setup()
        g_pool.image_tex = Named_Texture()

        # trigger on_resize
        on_resize(main_window, *glfw.glfwGetFramebufferSize(main_window))

        def handle_notifications(n):
            subject = n['subject']
            if subject == 'start_plugin':
                g_pool.plugins.add(
                    g_pool.plugin_by_name[n['name']], args=n.get('args', {}))
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

            events = {}
            # report time between now and the last loop interation
            events['dt'] = get_dt()

            # pupil and gaze positions are added by their respective producer plugins
            events['pupil_positions'] = []
            events['gaze_positions'] = []

            # allow each Plugin to do its work.
            for p in g_pool.plugins:
                p.recent_events(events)

            # check if a plugin need to be destroyed
            g_pool.plugins.clean()

            glfw.glfwMakeContextCurrent(main_window)
            # render visual feedback from loaded plugins
            if gl_utils.is_window_visible(main_window):

                gl_utils.glViewport(0, 0, *g_pool.camera_render_size)
                g_pool.capture.gl_display()
                for p in g_pool.plugins:
                    p.gl_display()

                gl_utils.glViewport(0, 0, *window_size)

                try:
                    clipboard = glfw.glfwGetClipboardString(main_window).decode()
                except AttributeError:  # clipbaord is None, might happen on startup
                    clipboard = ''
                g_pool.gui.update_clipboard(clipboard)
                user_input = g_pool.gui.update()
                if user_input.clipboard and user_input.clipboard != clipboard:
                    # only write to clipboard if content changed
                    glfw.glfwSetClipboardString(main_window, user_input.clipboard.encode())

                for b in user_input.buttons:
                    button, action, mods = b
                    x, y = glfw.glfwGetCursorPos(main_window)
                    pos = x * hdpi_factor, y * hdpi_factor
                    pos = normalize(pos, g_pool.camera_render_size)
                    pos = denormalize(pos, g_pool.capture.frame_size)
                    for p in g_pool.plugins:
                        p.on_click(pos, button, action)

                for key, scancode, action, mods in user_input.keys:
                    for p in g_pool.plugins:
                        p.on_key(key, scancode, action, mods)

                for char_ in user_input.chars:
                    for p in g_pool.plugins:
                        p.on_char(char_)

                # present frames at appropriate speed
                g_pool.seek_control.wait(events['frame'].timestamp)
                glfw.glfwSwapBuffers(main_window)

            glfw.glfwPollEvents()

        session_settings['loaded_plugins'] = g_pool.plugins.get_initializers()
        session_settings['min_data_confidence'] = g_pool.min_data_confidence
        session_settings['min_calibration_confidence'] = g_pool.min_calibration_confidence
        session_settings['gui_scale'] = g_pool.gui_user_scale
        session_settings['ui_config'] = g_pool.gui.configuration
        session_settings['window_position'] = glfw.glfwGetWindowPos(main_window)
        session_settings['version'] = str(g_pool.version)

        session_window_size = glfw.glfwGetWindowSize(main_window)
        if 0 not in session_window_size:
            session_settings['window_size'] = session_window_size

        session_settings.close()

        # de-init all running plugins
        for p in g_pool.plugins:
            p.alive = False
        g_pool.plugins.clean()

        g_pool.capture.cleanup()
        g_pool.gui.terminate()
        glfw.glfwDestroyWindow(main_window)

    except:
        import traceback
        trace = traceback.format_exc()
        logger.error('Process Player crashed with trace:\n{}'.format(trace))
    finally:
        logger.info("Process shutting down.")
        ipc_pub.notify({'subject': 'player_process.stopped'})
        sleep(1.0)


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

    try:

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
        session_settings = Persistent_Dict(os.path.join(user_dir, "user_settings_player"))
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
            hdpi_factor = glfw.getHDPIFactor(window)
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
                try:
                    update_recording_to_recent(rec_dir)
                except AssertionError as err:
                    logger.error(str(err))
                    rec_dir = None
                else:
                    glfw.glfwSetWindowShouldClose(window, True)

            glfw.glfwPollEvents()

        session_settings['window_position'] = glfw.glfwGetWindowPos(window)
        session_settings.close()
        glfw.glfwDestroyWindow(window)
        if rec_dir:
            ipc_pub.notify({"subject": "player_process.should_start", "rec_dir": rec_dir})

    except:
        import traceback
        trace = traceback.format_exc()
        logger.error('Process player_drop crashed with trace:\n{}'.format(trace))

    finally:
        sleep(1.0)


def player_profiled(rec_dir, ipc_pub_url, ipc_sub_url,
                    ipc_push_url, user_dir, app_version):
    import cProfile
    import subprocess
    import os
    from .player import player
    cProfile.runctx("player(rec_dir, ipc_pub_url, ipc_sub_url, ipc_push_url, user_dir, app_version)",
                    {'rec_dir': rec_dir, 'ipc_pub_url': ipc_pub_url, 'ipc_sub_url': ipc_sub_url,
                     'ipc_push_url': ipc_push_url, 'user_dir': user_dir,
                     'app_version': app_version}, locals(), "player.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(
        loc[0], 'pupil_src', 'shared_modules', 'gprof2dot.py')
    subprocess.call("python " + gprof2dot_loc + " -f pstats player.pstats | dot -Tpng -o player_cpu_time.png", shell=True)
    print("created cpu time graph for world process. Please check out the png next to the player.py file")
