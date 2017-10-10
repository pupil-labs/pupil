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
    window_position_default = (30, 30)
elif platform.system() == 'Windows':
    scroll_factor = 10.0
    window_position_default = (8, 31)
else:
    scroll_factor = 1.0
    window_position_default = (0, 0)


def player(rec_dir, ipc_pub_url, ipc_sub_url,
           ipc_push_url, user_dir, app_version):
    # general imports
    from time import sleep
    import logging
    import errno
    from glob import glob
    from copy import deepcopy
    from time import time
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
    logger.setLevel(logging.INFO)
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    try:

        # imports
        from file_methods import Persistent_Dict, load_object

        # display
        import glfw
        # check versions for our own depedencies as they are fast-changing
        from pyglui import __version__ as pyglui_version

        from pyglui import ui, cygl
        from pyglui.cygl.utils import Named_Texture
        import gl_utils
        # capture
        from video_capture import File_Source, EndofVideoFileError, FileSeekError

        # helpers/utils
        from version_utils import VersionFormat
        from methods import normalize, denormalize, delta_t, get_system_info
        from player_methods import correlate_data, is_pupil_rec_dir, load_meta_info

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
        from marker_auto_trim_marks import Marker_Auto_Trim_Marks
        from fixation_detector import Offline_Fixation_Detector
        from batch_exporter import Batch_Exporter
        from log_display import Log_Display
        from annotations import Annotation_Player
        from raw_data_exporter import Raw_Data_Exporter
        from log_history import Log_History
        from pupil_producers import Pupil_From_Recording, Offline_Pupil_Detection
        from gaze_producers import Gaze_From_Recording, Offline_Calibration
        from system_graphs import System_Graphs

        assert pyglui_version >= '1.8', 'pyglui out of date, please upgrade to newest version'

        runtime_plugins = import_runtime_plugins(os.path.join(user_dir, 'plugins'))
        system_plugins = [Log_Display, Seek_Control, Plugin_Manager, System_Graphs]
        user_plugins = [Vis_Circle, Vis_Fixation, Vis_Polyline, Vis_Light_Points,
                        Vis_Cross, Vis_Watermark, Vis_Eye_Video_Overlay, Vis_Scan_Path,
                        Offline_Fixation_Detector,
                        Video_Export_Launcher, Offline_Surface_Tracker, Raw_Data_Exporter,
                        Batch_Exporter, Annotation_Player, Log_History, Marker_Auto_Trim_Marks,
                        Pupil_From_Recording, Offline_Pupil_Detection, Gaze_From_Recording,
                        Offline_Calibration] + runtime_plugins

        plugins = system_plugins + user_plugins

        # Callback functions
        def on_resize(window, w, h):
            nonlocal window_size

            if gl_utils.is_window_visible(window):
                hdpi_factor = float(glfw.glfwGetFramebufferSize(window)[0] / glfw.glfwGetWindowSize(window)[0])
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
            hdpi_factor = float(glfw.glfwGetFramebufferSize(window)[0]/glfw.glfwGetWindowSize(window)[0])
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

        video_path = [f for f in glob(os.path.join(rec_dir, "world.*"))
                      if os.path.splitext(f)[1] in ('.mp4', '.mkv', '.avi', '.h264', '.mjpeg')][0]
        pupil_data_path = os.path.join(rec_dir, "pupil_data")

        meta_info = load_meta_info(rec_dir)

        # log info about Pupil Platform and Platform in player.log
        logger.info('Application Version: {}'.format(app_version))
        logger.info('System Info: {}'.format(get_system_info()))

        icon_bar_width = 50
        window_size = None

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

        # sets itself to g_pool.capture
        File_Source(g_pool, video_path)

        # load session persistent settings
        session_settings = Persistent_Dict(os.path.join(user_dir, "user_settings_player"))
        if VersionFormat(session_settings.get("version", '0.0')) != app_version:
            logger.info("Session setting are a different version of this app. I will not use those.")
            session_settings.clear()

        width, height = session_settings.get('window_size', g_pool.capture.frame_size)
        window_pos = session_settings.get('window_position', window_position_default)
        glfw.glfwInit()
        main_window = glfw.glfwCreateWindow(width, height, "Pupil Player: "+meta_info["Recording Name"]+" - "
                                       + rec_dir.split(os.path.sep)[-1], None, None)
        glfw.glfwSetWindowPos(main_window, window_pos[0], window_pos[1])
        glfw.glfwMakeContextCurrent(main_window)
        cygl.utils.init()
        g_pool.main_window = main_window

        def set_scale(new_scale):
            hdpi_factor = float(glfw.glfwGetFramebufferSize(
                main_window)[0]) / glfw.glfwGetWindowSize(main_window)[0]
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
        g_pool.new_seek = True
        g_pool.user_dir = user_dir
        g_pool.rec_dir = rec_dir
        g_pool.meta_info = meta_info
        g_pool.min_data_confidence = session_settings.get('min_data_confidence', 0.6)

        g_pool.pupil_positions = []
        g_pool.gaze_positions = []
        g_pool.fixations = []

        g_pool.notifications_by_frame = correlate_data(g_pool.pupil_data['notifications'], g_pool.timestamps)
        g_pool.pupil_positions_by_frame = [[] for x in g_pool.timestamps]  # populated by producer`
        g_pool.gaze_positions_by_frame = [[] for x in g_pool.timestamps]  # populated by producer
        g_pool.fixations_by_frame = [[] for x in g_pool.timestamps]  # populated by the fixation detector plugin

        def next_frame(_):
            try:
                g_pool.capture.seek_to_frame(g_pool.capture.get_frame_index() + 1)
            except(FileSeekError):
                logger.warning("Could not seek to next frame.")
            else:
                g_pool.new_seek = True

        def prev_frame(_):
            try:
                g_pool.capture.seek_to_frame(g_pool.capture.get_frame_index() - 1)
            except(FileSeekError):
                logger.warning("Could not seek to previous frame.")
            else:
                g_pool.new_seek = True

        def toggle_play(new_state):
            if g_pool.capture.get_frame_index() >= g_pool.capture.get_frame_count()-5:
                g_pool.capture.seek_to_frame(1)  # avoid pause set by hitting trimmark pause.
                logger.warning("End of video - restart at beginning.")
            g_pool.capture.play = new_state

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
            export_range = g_pool.seek_control.trim_left, g_pool.seek_control.trim_right
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

        def reset_restart():
            logger.warning("Resetting all settings and restarting Player.")
            glfw.glfwSetWindowShouldClose(main_window, True)
            ipc_pub.notify({'subject': 'clear_settings_process.should_start'})
            ipc_pub.notify({'subject': 'player_process.should_start', 'rec_dir': rec_dir, 'delay': 2.})

        def toggle_general_settings(collapsed):
            #this is the menu toggle logic.
            # Only one menu can be open.
            # If no menu is open the menubar should collapse.
            g_pool.menubar.collapsed = collapsed
            for m in g_pool.menubar.elements:
                m.collapsed = True
            general_settings.collapsed = collapsed

        g_pool.gui = ui.UI()
        g_pool.gui_user_scale = session_settings.get('gui_scale', 1.)
        g_pool.menubar = ui.Scrolling_Menu("Settings", pos=(-500, 0), size=(-icon_bar_width, 0), header_pos='left')
        g_pool.iconbar = ui.Scrolling_Menu("Icons", pos=(-icon_bar_width,0),size=(0,0),header_pos='hidden')
        g_pool.timelines = ui.Container((0, 0), (0, 0), (0, 0))
        g_pool.timelines.horizontal_constraint = g_pool.menubar

        general_settings = ui.Growing_Menu('General', header_pos='headline')
        general_settings.append(ui.Button('Reset window size',
                                          lambda: glfw.glfwSetWindowSize(main_window, g_pool.capture.frame_size[0], g_pool.capture.frame_size[1])))
        general_settings.append(ui.Selector('gui_user_scale', g_pool, setter=set_scale, selection=[.8, .9, 1., 1.1, 1.2]+list(np.arange(1.5, 5.1, .5)), label='Interface Size'))
        general_settings.append(ui.Info_Text('Player Version: {}'.format(g_pool.version)))
        general_settings.append(ui.Info_Text('Capture Version: {}'.format(meta_info['Capture Software Version'])))
        general_settings.append(ui.Info_Text('Data Format Version: {}'.format(meta_info['Data Format Version'])))
        general_settings.append(ui.Slider('min_data_confidence', g_pool, setter=set_data_confidence,
                                          step=.05, min=0.0, max=1.0, label='Confidence threshold'))
        general_settings.append(ui.Button('Restart with default settings', reset_restart))

        g_pool.menubar.append(general_settings)
        icon = ui.Icon('collapsed', general_settings, label=chr(0xe8b8), on_val=False, off_val=True, setter=toggle_general_settings, label_font='pupil_icons')
        icon.tooltip = 'General Settings'
        g_pool.iconbar.append(icon)

        user_plugin_separator = ui.Separator()
        user_plugin_separator.order = 0.35
        g_pool.iconbar.append(user_plugin_separator)

        g_pool.quickbar = ui.Stretching_Menu('Quick Bar', (0, 100), (120, -100))
        g_pool.capture.play_button = ui.Thumb('play',
                                      g_pool.capture,
                                      label=chr(0xE037),
                                      setter=toggle_play,
                                      hotkey=glfw.GLFW_KEY_SPACE,
                                      label_font='pupil_icons')
        g_pool.capture.play_button.on_color[:] = (0.5, 0.8, 0.75,.9)
        g_pool.forward_button = ui.Thumb('forward',
                                         label=chr(0xE01F),
                                         getter=lambda: False,
                                         setter=next_frame,
                                         hotkey=glfw.GLFW_KEY_RIGHT,
                                         label_font='pupil_icons')
        g_pool.backward_button = ui.Thumb('backward',
                                          label=chr(0xE020),
                                          getter=lambda: False,
                                          setter=prev_frame,
                                          hotkey=glfw.GLFW_KEY_LEFT,
                                          label_font='pupil_icons')
        g_pool.export_button = ui.Thumb('export',
                                        label=chr(0xe2c4),
                                        getter=lambda: False,
                                        setter=do_export,
                                        hotkey='e',
                                        label_font='pupil_icons')
        g_pool.quickbar.extend([g_pool.capture.play_button, g_pool.forward_button, g_pool.backward_button, g_pool.export_button])
        g_pool.gui.append(g_pool.menubar)
        g_pool.gui.append(g_pool.timelines)
        g_pool.gui.append(g_pool.iconbar)
        g_pool.gui.append(g_pool.quickbar)

        # we always load these plugins
        default_plugins = [('Plugin_Manager', {}), ('Seek_Control', {}), ('Log_Display', {}),
                           ('Vis_Scan_Path', {}), ('Vis_Polyline', {}), ('Vis_Circle', {}), ('System_Graphs', {}),
                           ('Video_Export_Launcher', {}), ('Pupil_From_Recording', {}), ('Gaze_From_Recording', {})]
        g_pool.plugins = Plugin_List(g_pool, session_settings.get('loaded_plugins', default_plugins))

        # Register callbacks main_window
        glfw.glfwSetFramebufferSizeCallback(main_window, on_resize)
        glfw.glfwSetKeyCallback(main_window, on_window_key)
        glfw.glfwSetCharCallback(main_window, on_window_char)
        glfw.glfwSetMouseButtonCallback(main_window, on_window_mouse_button)
        glfw.glfwSetCursorPosCallback(main_window, on_pos)
        glfw.glfwSetScrollCallback(main_window, on_scroll)
        glfw.glfwSetDropCallback(main_window, on_drop)

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

            # grab new frame
            if g_pool.capture.play or g_pool.new_seek:
                g_pool.new_seek = False
                try:
                    new_frame = g_pool.capture.get_frame()
                except EndofVideoFileError:
                    # end of video logic: pause at last frame.
                    g_pool.capture.play = False
                    logger.warning("end of video")

            frame = new_frame.copy()
            events = {}
            events['frame'] = frame
            # report time between now and the last loop interation
            events['dt'] = get_dt()
            # new positons we make a deepcopy just like the image is a copy.
            events['gaze_positions'] = deepcopy(g_pool.gaze_positions_by_frame[frame.index])
            events['pupil_positions'] = deepcopy(g_pool.pupil_positions_by_frame[frame.index])

            # allow each Plugin to do its work.
            for p in g_pool.plugins:
                p.recent_events(events)

            # check if a plugin need to be destroyed
            g_pool.plugins.clean()

            glfw.glfwMakeContextCurrent(main_window)
            # render visual feedback from loaded plugins
            if gl_utils.is_window_visible(main_window):

                gl_utils.glViewport(0, 0, *g_pool.camera_render_size)
                g_pool.capture._recent_frame = frame
                g_pool.capture.gl_display()
                for p in g_pool.plugins:
                    p.gl_display()

                gl_utils.glViewport(0, 0, *window_size)

                unused_elements = g_pool.gui.update()
                for b in unused_elements.buttons:
                    button, action, mods = b
                    pos = glfw.glfwGetCursorPos(main_window)
                    pos = normalize(pos, g_pool.camera_render_size)
                    pos = denormalize(pos, g_pool.capture.frame_size)
                    for p in g_pool.plugins:
                        p.on_click(pos, button, action)

                for key, scancode, action, mods in unused_elements.keys:
                    for p in g_pool.plugins:
                        p.on_key(key, scancode, action, mods)

                for char_ in unused_elements.chars:
                    for p in g_pool.plugins:
                        p.on_char(char_)

                glfw.glfwSwapBuffers(main_window)

            # present frames at appropriate speed
            g_pool.capture.wait(frame)
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
