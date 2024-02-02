"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import asyncio
import os
import platform
import signal
from functools import partial
from types import SimpleNamespace

# UI Platform tweaks
if platform.system() == "Linux":
    scroll_factor = 10.0
    window_position_default = (30, 30)
elif platform.system() == "Windows":
    scroll_factor = 10.0
    window_position_default = (8, 90)
else:
    scroll_factor = 1.0
    window_position_default = (0, 0)

MIN_DATA_CONFIDENCE_DEFAULT = 0.6
MIN_CALIBRATION_CONFIDENCE_DEFAULT = 0.8


def player(
    rec_dir, ipc_pub_url, ipc_sub_url, ipc_push_url, user_dir, app_version, debug
):
    # general imports
    import logging
    from glob import glob
    from time import localtime, sleep, strftime, time

    import numpy as np

    # networking
    import zmq
    import zmq_tools

    # zmq ipc setup
    zmq_ctx = zmq.Context()
    ipc_pub = zmq_tools.Msg_Dispatcher(zmq_ctx, ipc_push_url)
    notify_sub = zmq_tools.Msg_Receiver(zmq_ctx, ipc_sub_url, topics=("notify",))

    # log setup
    logging.getLogger("OpenGL").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    try:
        from background_helper import IPC_Logging_Task_Proxy

        IPC_Logging_Task_Proxy.push_url = ipc_push_url

        from tasklib.background.patches import IPCLoggingPatch

        IPCLoggingPatch.ipc_push_url = ipc_push_url

        # imports
        # display
        import glfw
        from file_methods import Persistent_Dict, next_export_sub_dir
        from gl_utils import GLFWErrorReporting
        from OpenGL.GL import GL_COLOR_BUFFER_BIT

        GLFWErrorReporting.set_default()

        # check versions for our own depedencies as they are fast-changing
        import gl_utils
        import player_methods as pm
        from annotations import Annotation_Player
        from audio_playback import Audio_Playback
        from blink_detection import Offline_Blink_Detection
        from csv_utils import write_key_value_file

        # from marker_auto_trim_marks import Marker_Auto_Trim_Marks
        from fixation_detector import Offline_Fixation_Detector
        from gaze_producer.gaze_from_offline_calibration import (
            GazeFromOfflineCalibration,
        )
        from gaze_producer.gaze_from_recording import GazeFromRecording
        from head_pose_tracker.offline_head_pose_tracker import (
            Offline_Head_Pose_Tracker,
        )
        from hotkey import Hotkey
        from imu_timeline import IMUTimeline
        from log_display import Log_Display
        from log_history import Log_History
        from methods import delta_t, denormalize, get_system_info, normalize

        # Plug-ins
        from plugin import Plugin, Plugin_List, import_runtime_plugins
        from plugin_manager import Plugin_Manager
        from pupil_detector_plugins.detector_base_plugin import PupilDetectorPlugin
        from pupil_producers import (
            DisabledPupilProducer,
            Offline_Pupil_Detection,
            Pupil_From_Recording,
        )
        from pupil_recording import (
            InvalidRecordingException,
            PupilRecording,
            assert_valid_recording_type,
        )
        from pyglui import __version__ as pyglui_version
        from pyglui import cygl, ui
        from pyglui.cygl.utils import RGBA, Named_Texture
        from raw_data_exporter import Raw_Data_Exporter
        from seek_control import Seek_Control
        from surface_tracker import Surface_Tracker_Offline
        from system_graphs import System_Graphs
        from system_timelines import System_Timelines

        # helpers/utils
        from version_utils import parse_version

        # capture
        from video_capture import File_Source
        from video_export.plugins.eye_video_exporter import Eye_Video_Exporter
        from video_export.plugins.imotions_exporter import iMotions_Exporter
        from video_export.plugins.world_video_exporter import World_Video_Exporter
        from video_overlay.plugins import Eye_Overlay, Video_Overlay
        from vis_circle import Vis_Circle
        from vis_cross import Vis_Cross
        from vis_fixation import Vis_Fixation
        from vis_light_points import Vis_Light_Points
        from vis_polyline import Vis_Polyline
        from vis_watermark import Vis_Watermark

        assert parse_version(pyglui_version) >= parse_version(
            "1.31.0"
        ), "pyglui out of date, please upgrade to newest version"

        process_was_interrupted = False

        def interrupt_handler(sig, frame):
            import traceback

            trace = traceback.format_stack(f=frame)
            logger.debug(f"Caught signal {sig} in:\n" + "".join(trace))
            nonlocal process_was_interrupted
            process_was_interrupted = True

        signal.signal(signal.SIGINT, interrupt_handler)

        runtime_plugins = import_runtime_plugins(os.path.join(user_dir, "plugins"))
        runtime_plugins = [
            p for p in runtime_plugins if not issubclass(p, PupilDetectorPlugin)
        ]
        system_plugins = [
            Log_Display,
            Seek_Control,
            Plugin_Manager,
            System_Graphs,
            System_Timelines,
            Audio_Playback,
        ]
        user_plugins = [
            Vis_Circle,
            Vis_Fixation,
            Vis_Polyline,
            Vis_Light_Points,
            Vis_Cross,
            Vis_Watermark,
            Eye_Overlay,
            Video_Overlay,
            Offline_Fixation_Detector,
            Offline_Blink_Detection,
            Surface_Tracker_Offline,
            Raw_Data_Exporter,
            Annotation_Player,
            Log_History,
            DisabledPupilProducer,
            Pupil_From_Recording,
            Offline_Pupil_Detection,
            GazeFromRecording,
            GazeFromOfflineCalibration,
            World_Video_Exporter,
            iMotions_Exporter,
            Eye_Video_Exporter,
            Offline_Head_Pose_Tracker,
            IMUTimeline,
        ] + runtime_plugins

        plugins = system_plugins + user_plugins

        def consume_events_and_render_buffer():
            gl_utils.glViewport(0, 0, *g_pool.camera_render_size)
            g_pool.capture.gl_display()
            for p in g_pool.plugins:
                p.gl_display()

            gl_utils.glViewport(0, 0, *window_size)

            try:
                clipboard = glfw.get_clipboard_string(main_window).decode()
            except (AttributeError, glfw.GLFWError):
                # clipbaord is None, might happen on startup
                clipboard = ""
            g_pool.gui.update_clipboard(clipboard)
            user_input = g_pool.gui.update()
            if user_input.clipboard and user_input.clipboard != clipboard:
                # only write to clipboard if content changed
                glfw.set_clipboard_string(main_window, user_input.clipboard)

            for b in user_input.buttons:
                button, action, mods = b
                x, y = glfw.get_cursor_pos(main_window)
                pos = gl_utils.window_coordinate_to_framebuffer_coordinate(
                    main_window, x, y, cached_scale=None
                )
                pos = normalize(pos, g_pool.camera_render_size)
                pos = denormalize(pos, g_pool.capture.frame_size)

                for plugin in g_pool.plugins:
                    if plugin.on_click(pos, button, action):
                        break

            for key, scancode, action, mods in user_input.keys:
                for plugin in g_pool.plugins:
                    if plugin.on_key(key, scancode, action, mods):
                        break

            for char_ in user_input.chars:
                for plugin in g_pool.plugins:
                    if plugin.on_char(char_):
                        break

            glfw.swap_buffers(main_window)

        # Callback functions
        def on_resize(window, w, h):
            nonlocal window_size
            nonlocal content_scale
            if w == 0 or h == 0:
                return

            # Always clear buffers on resize to make sure that there are no overlapping
            # artifacts from previous frames.
            gl_utils.glClear(GL_COLOR_BUFFER_BIT)
            gl_utils.glClearColor(0, 0, 0, 1)

            content_scale = gl_utils.get_content_scale(window)
            framebuffer_scale = gl_utils.get_framebuffer_scale(window)
            g_pool.gui.scale = content_scale
            window_size = w, h
            g_pool.camera_render_size = w - int(icon_bar_width * g_pool.gui.scale), h
            g_pool.gui.update_window(*window_size)
            g_pool.gui.collect_menus()
            for p in g_pool.plugins:
                p.on_window_resize(window, *g_pool.camera_render_size)

            # Minimum window size required, otherwise parts of the UI can cause openGL
            # issues with permanent effects. Depends on the content scale, which can
            # potentially be dynamically modified, so we re-adjust the size limits every
            # time here.
            min_size = int(2 * icon_bar_width * g_pool.gui.scale / framebuffer_scale)
            glfw.set_window_size_limits(
                window,
                min_size,
                min_size,
                glfw.DONT_CARE,
                glfw.DONT_CARE,
            )

            # Needed, to update the window buffer while resizing
            consume_events_and_render_buffer()

        def on_window_key(window, key, scancode, action, mods):
            g_pool.gui.update_key(key, scancode, action, mods)

        def on_window_char(window, char):
            g_pool.gui.update_char(char)

        def on_window_mouse_button(window, button, action, mods):
            g_pool.gui.update_button(button, action, mods)

        def on_pos(window, x, y):
            x, y = gl_utils.window_coordinate_to_framebuffer_coordinate(
                window, x, y, cached_scale=None
            )
            g_pool.gui.update_mouse(x, y)
            pos = x, y
            pos = normalize(pos, g_pool.camera_render_size)
            # Position in img pixels
            pos = denormalize(pos, g_pool.capture.frame_size)
            for p in g_pool.plugins:
                p.on_pos(pos)

        def on_scroll(window, x, y):
            g_pool.gui.update_scroll(x, y * scroll_factor)

        def on_drop(window, paths):
            for path in paths:
                try:
                    assert_valid_recording_type(path)
                    _restart_with_recording(path)
                    return
                except InvalidRecordingException as err:
                    logger.debug(str(err))

            for plugin in g_pool.plugins:
                if plugin.on_drop(paths):
                    break

        def _restart_with_recording(rec_dir):
            logger.debug(f"Starting new session with '{rec_dir}'")
            ipc_pub.notify(
                {"subject": "player_drop_process.should_start", "rec_dir": rec_dir}
            )
            glfw.set_window_should_close(g_pool.main_window, True)

        tick = delta_t()

        def get_dt():
            return next(tick)

        recording = PupilRecording(rec_dir)
        meta_info = recording.meta_info

        # log info about Pupil Platform and Platform in player.log
        logger.info(f"Application Version: {app_version}")
        logger.info(f"System Info: {get_system_info()}")
        logger.debug(f"Debug flag: {debug}")

        icon_bar_width = 50
        window_size = None
        content_scale = 1.0

        # create container for globally scoped vars
        g_pool = SimpleNamespace()
        g_pool.app = "player"
        g_pool.process = "player"
        g_pool.zmq_ctx = zmq_ctx
        g_pool.ipc_pub = ipc_pub
        g_pool.ipc_pub_url = ipc_pub_url
        g_pool.ipc_sub_url = ipc_sub_url
        g_pool.ipc_push_url = ipc_push_url
        g_pool.plugin_by_name = {p.__name__: p for p in plugins}
        g_pool.camera_render_size = None

        video_path = recording.files().core().world().videos()[0].resolve()
        try:
            File_Source(
                g_pool,
                timing="external",
                source_path=video_path,
                buffered_decoding=True,
                fill_gaps=True,
            )
        except AttributeError:
            logger.warning(
                "Buffered decoder not available. Falling back to on demand decoder."
            )
            File_Source(
                g_pool,
                timing="external",
                source_path=video_path,
                buffered_decoding=False,
                fill_gaps=True,
            )

        # load session persistent settings
        session_settings = Persistent_Dict(
            os.path.join(user_dir, "user_settings_player")
        )
        if parse_version(session_settings.get("version", "0.0")) != app_version:
            logger.info(
                "Session setting are a different version of this app. I will not use those."
            )
            session_settings.clear()

        width, height = g_pool.capture.frame_size
        width += icon_bar_width
        width, height = session_settings.get("window_size", (width, height))

        window_name = f"Pupil Player: {meta_info.recording_name} - {rec_dir}"

        glfw.init()
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
        main_window = glfw.create_window(width, height, window_name, None, None)
        window_position_manager = gl_utils.WindowPositionManager()
        window_pos = window_position_manager.new_window_position(
            window=main_window,
            default_position=window_position_default,
            previous_position=session_settings.get("window_position", None),
        )
        glfw.set_window_pos(main_window, window_pos[0], window_pos[1])

        glfw.make_context_current(main_window)
        cygl.utils.init()
        g_pool.main_window = main_window

        g_pool.version = app_version
        g_pool.timestamps = g_pool.capture.timestamps
        g_pool.get_timestamp = lambda: 0.0
        g_pool.user_dir = user_dir
        g_pool.rec_dir = rec_dir
        g_pool.meta_info = meta_info
        g_pool.min_data_confidence = session_settings.get(
            "min_data_confidence", MIN_DATA_CONFIDENCE_DEFAULT
        )
        g_pool.min_calibration_confidence = session_settings.get(
            "min_calibration_confidence", MIN_CALIBRATION_CONFIDENCE_DEFAULT
        )

        # populated by producers
        g_pool.pupil_positions = pm.PupilDataBisector()
        g_pool.gaze_positions = pm.Bisector()
        g_pool.fixations = pm.Affiliator()
        g_pool.eye_movements = pm.Affiliator()

        def set_data_confidence(new_confidence):
            g_pool.min_data_confidence = new_confidence
            notification = {"subject": "min_data_confidence_changed"}
            notification["_notify_time_"] = time() + 0.8
            g_pool.ipc_pub.notify(notification)

        def do_export(_):
            left_idx = g_pool.seek_control.trim_left
            right_idx = g_pool.seek_control.trim_right
            export_range = left_idx, right_idx + 1  # exclusive range.stop
            export_ts_window = pm.exact_window(g_pool.timestamps, (left_idx, right_idx))

            export_dir = os.path.join(g_pool.rec_dir, "exports")
            export_dir = next_export_sub_dir(export_dir)

            os.makedirs(export_dir)
            logger.info(f'Created export dir at "{export_dir}"')

            export_info = {
                "Player Software Version": str(g_pool.version),
                "Data Format Version": meta_info.min_player_version,
                "Export Date": strftime("%d.%m.%Y", localtime()),
                "Export Time": strftime("%H:%M:%S", localtime()),
                "Frame Index Range:": g_pool.seek_control.get_frame_index_trim_range_string(),
                "Relative Time Range": g_pool.seek_control.get_rel_time_trim_range_string(),
                "Absolute Time Range": g_pool.seek_control.get_abs_time_trim_range_string(),
            }
            with open(os.path.join(export_dir, "export_info.csv"), "w") as csv:
                write_key_value_file(csv, export_info)

            notification = {
                "subject": "should_export",
                "range": export_range,
                "ts_window": export_ts_window,
                "export_dir": export_dir,
            }
            g_pool.ipc_pub.notify(notification)

        def reset_restart():
            logger.warning("Resetting all settings and restarting Player.")
            glfw.set_window_should_close(main_window, True)
            ipc_pub.notify({"subject": "clear_settings_process.should_start"})
            ipc_pub.notify(
                {
                    "subject": "player_process.should_start",
                    "rec_dir": rec_dir,
                    "delay": 2.0,
                }
            )

        def toggle_general_settings(collapsed):
            # this is the menu toggle logic.
            # Only one menu can be open.
            # If no menu is open the menubar should collapse.
            g_pool.menubar.collapsed = collapsed
            for m in g_pool.menubar.elements:
                m.collapsed = True
            general_settings.collapsed = collapsed

        g_pool.gui = ui.UI()
        g_pool.menubar = ui.Scrolling_Menu(
            "Settings", pos=(-500, 0), size=(-icon_bar_width, 0), header_pos="left"
        )
        g_pool.iconbar = ui.Scrolling_Menu(
            "Icons", pos=(-icon_bar_width, 0), size=(0, 0), header_pos="hidden"
        )
        g_pool.timelines = ui.Container((0, 0), (0, 0), (0, 0))
        g_pool.timelines.horizontal_constraint = g_pool.menubar
        g_pool.user_timelines = ui.Timeline_Menu(
            "User Timelines", pos=(0.0, -150.0), size=(0.0, 0.0), header_pos="headline"
        )
        g_pool.user_timelines.color = RGBA(a=0.0)
        g_pool.user_timelines.collapsed = True
        # add container that constaints itself to the seekbar height
        vert_constr = ui.Container((0, 0), (0, -50.0), (0, 0))
        vert_constr.append(g_pool.user_timelines)
        g_pool.timelines.append(vert_constr)

        def set_window_size():
            # Get current capture frame size
            f_width, f_height = g_pool.capture.frame_size

            # Get current display scale factor
            content_scale = gl_utils.get_content_scale(main_window)
            framebuffer_scale = gl_utils.get_framebuffer_scale(main_window)
            display_scale_factor = content_scale / framebuffer_scale

            # Scale the capture frame size by display scale factor
            f_width *= display_scale_factor
            f_height *= display_scale_factor

            # Increas the width to account for the added scaled icon bar width
            f_width += icon_bar_width * display_scale_factor

            # Set the newly calculated size (scaled capture frame size + scaled icon bar width)
            glfw.set_window_size(main_window, int(f_width), int(f_height))

        general_settings = ui.Growing_Menu("General", header_pos="headline")
        general_settings.append(ui.Button("Reset window size", set_window_size))
        general_settings.append(
            ui.Info_Text(f"Minimum Player Version: {meta_info.min_player_version}")
        )
        general_settings.append(ui.Info_Text(f"Player Version: {g_pool.version}"))
        general_settings.append(
            ui.Info_Text(f"Recording Software: {meta_info.recording_software_name}")
        )
        general_settings.append(
            ui.Info_Text(
                f"Recording Software Version: {meta_info.recording_software_version}"
            )
        )

        general_settings.append(
            ui.Info_Text(
                "High level data, e.g. fixations, or visualizations only consider gaze data that has an equal or higher confidence than the minimum data confidence."
            )
        )
        general_settings.append(
            ui.Slider(
                "min_data_confidence",
                g_pool,
                setter=set_data_confidence,
                step=0.05,
                min=0.0,
                max=1.0,
                label="Minimum data confidence",
            )
        )

        general_settings.append(
            ui.Button("Restart with default settings", reset_restart)
        )

        g_pool.menubar.append(general_settings)
        icon = ui.Icon(
            "collapsed",
            general_settings,
            label=chr(0xE8B8),
            on_val=False,
            off_val=True,
            setter=toggle_general_settings,
            label_font="pupil_icons",
        )
        icon.tooltip = "General Settings"
        g_pool.iconbar.append(icon)

        user_plugin_separator = ui.Separator()
        user_plugin_separator.order = 0.35
        g_pool.iconbar.append(user_plugin_separator)

        g_pool.quickbar = ui.Stretching_Menu("Quick Bar", (0, 100), (100, -100))
        g_pool.export_button = ui.Thumb(
            "export",
            label=chr(0xE2C5),
            getter=lambda: False,
            setter=do_export,
            hotkey=Hotkey.EXPORT_START_PLAYER_HOTKEY(),
            label_font="pupil_icons",
        )
        g_pool.quickbar.extend([g_pool.export_button])
        g_pool.gui.append(g_pool.menubar)
        g_pool.gui.append(g_pool.timelines)
        g_pool.gui.append(g_pool.iconbar)
        g_pool.gui.append(g_pool.quickbar)

        # we always load these plugins
        _pupil_producer_plugins = [
            # In priority order (first is default)
            ("Pupil_From_Recording", {}),
            ("Offline_Pupil_Detection", {}),
            ("DisabledPupilProducer", {}),
        ]
        _pupil_producer_plugins = list(reversed(_pupil_producer_plugins))
        _gaze_producer_plugins = [
            # In priority order (first is default)
            ("GazeFromRecording", {}),
            ("GazeFromOfflineCalibration", {}),
        ]
        _gaze_producer_plugins = list(reversed(_gaze_producer_plugins))
        default_plugins = [
            ("Plugin_Manager", {}),
            ("Seek_Control", {}),
            ("Log_Display", {}),
            ("Raw_Data_Exporter", {}),
            ("Vis_Polyline", {}),
            ("Vis_Circle", {}),
            ("System_Graphs", {}),
            ("System_Timelines", {}),
            ("World_Video_Exporter", {}),
            *_pupil_producer_plugins,
            *_gaze_producer_plugins,
            ("Audio_Playback", {}),
        ]
        _plugins_to_load = session_settings.get("loaded_plugins", None)
        if _plugins_to_load is None:
            # If no plugins are available from a previous session,
            # then use the default plugin list
            _plugins_to_load = default_plugins
        else:
            # If there are plugins available from a previous session,
            # then prepend plugins that are required, but might have not been available before
            _plugins_to_load = [
                *_pupil_producer_plugins,
                *_gaze_producer_plugins,
                *_plugins_to_load,
            ]

        g_pool.plugins = Plugin_List(g_pool, _plugins_to_load)

        # Manually add g_pool.capture to the plugin list
        g_pool.plugins._plugins.append(g_pool.capture)
        g_pool.plugins._plugins.sort(key=lambda p: p.order)
        g_pool.capture.init_ui()

        general_settings.insert(
            -1,
            ui.Text_Input(
                "rel_time_trim_section",
                getter=g_pool.seek_control.get_rel_time_trim_range_string,
                setter=g_pool.seek_control.set_rel_time_trim_range_string,
                label="Relative time range to export",
            ),
        )
        general_settings.insert(
            -1,
            ui.Text_Input(
                "frame_idx_trim_section",
                getter=g_pool.seek_control.get_frame_index_trim_range_string,
                setter=g_pool.seek_control.set_frame_index_trim_range_string,
                label="Frame index range to export",
            ),
        )

        on_focus = partial(gl_utils.window_focus_clipboard_callback, g_pool)

        # Register callbacks main_window
        glfw.set_framebuffer_size_callback(main_window, on_resize)
        glfw.set_key_callback(main_window, on_window_key)
        glfw.set_char_callback(main_window, on_window_char)
        glfw.set_mouse_button_callback(main_window, on_window_mouse_button)
        glfw.set_cursor_pos_callback(main_window, on_pos)
        glfw.set_scroll_callback(main_window, on_scroll)
        glfw.set_drop_callback(main_window, on_drop)
        glfw.set_window_focus_callback(main_window, on_focus)

        toggle_general_settings(True)

        g_pool.gui.configuration = session_settings.get("ui_config", {})
        # If previously selected plugin was not loaded this time, we will have an
        # expanded menubar without any menu selected. We need to ensure the menubar is
        # collapsed in this case.
        if all(submenu.collapsed for submenu in g_pool.menubar.elements):
            g_pool.menubar.collapsed = True

        # gl_state settings
        gl_utils.basic_gl_setup()
        g_pool.image_tex = Named_Texture()

        # trigger on_resize
        on_resize(main_window, *glfw.get_framebuffer_size(main_window))

        def handle_notifications(n):
            subject = n["subject"]
            if subject == "start_plugin":
                g_pool.plugins.add(
                    g_pool.plugin_by_name[n["name"]], args=n.get("args", {})
                )
            elif subject.startswith("meta.should_doc"):
                ipc_pub.notify(
                    {"subject": "meta.doc", "actor": g_pool.app, "doc": player.__doc__}
                )
                for p in g_pool.plugins:
                    if (
                        p.on_notify.__doc__
                        and p.__class__.on_notify != Plugin.on_notify
                    ):
                        ipc_pub.notify(
                            {
                                "subject": "meta.doc",
                                "actor": p.class_name,
                                "doc": p.on_notify.__doc__,
                            }
                        )

        while not glfw.window_should_close(main_window) and not process_was_interrupted:
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
            events["dt"] = get_dt()

            # pupil and gaze positions are added by their respective producer plugins
            events["pupil"] = []
            events["gaze"] = []

            # allow each Plugin to do its work.
            for p in g_pool.plugins:
                p.recent_events(events)

            # check if a plugin need to be destroyed
            g_pool.plugins.clean()

            glfw.make_context_current(main_window)
            glfw.poll_events()
            # render visual feedback from loaded plugins
            if gl_utils.is_window_visible(main_window):
                gl_utils.glViewport(0, 0, *g_pool.camera_render_size)
                g_pool.capture.gl_display()
                for p in g_pool.plugins:
                    p.gl_display()

                gl_utils.glViewport(0, 0, *window_size)

                user_input = g_pool.gui.update()

                for b in user_input.buttons:
                    button, action, mods = b
                    x, y = glfw.get_cursor_pos(main_window)
                    pos = gl_utils.window_coordinate_to_framebuffer_coordinate(
                        main_window, x, y, cached_scale=None
                    )
                    pos = normalize(pos, g_pool.camera_render_size)
                    pos = denormalize(pos, g_pool.capture.frame_size)

                    for plugin in g_pool.plugins:
                        if plugin.on_click(pos, button, action):
                            break

                for key, scancode, action, mods in user_input.keys:
                    for plugin in g_pool.plugins:
                        if plugin.on_key(key, scancode, action, mods):
                            break

                for char_ in user_input.chars:
                    for plugin in g_pool.plugins:
                        if plugin.on_char(char_):
                            break

                # present frames at appropriate speed
                g_pool.seek_control.wait(events["frame"].timestamp)
                glfw.swap_buffers(main_window)

        session_settings["loaded_plugins"] = g_pool.plugins.get_initializers()
        session_settings["min_data_confidence"] = g_pool.min_data_confidence
        session_settings[
            "min_calibration_confidence"
        ] = g_pool.min_calibration_confidence
        session_settings["ui_config"] = g_pool.gui.configuration
        session_settings["window_position"] = glfw.get_window_pos(main_window)
        session_settings["version"] = str(g_pool.version)

        session_window_size = glfw.get_window_size(main_window)
        if 0 not in session_window_size:
            f_width, f_height = session_window_size
            if platform.system() in ("Windows", "Linux"):
                f_width, f_height = (
                    f_width / content_scale,
                    f_height / content_scale,
                )
            session_settings["window_size"] = int(f_width), int(f_height)

        session_settings.close()

        # de-init all running plugins
        for p in g_pool.plugins:
            p.alive = False
        g_pool.plugins.clean()

        g_pool.gui.terminate()
        glfw.destroy_window(main_window)

    except Exception:
        logger.exception("Process Player crashed with trace:")
    finally:
        logger.info("Process shutting down.")
        ipc_pub.notify({"subject": "player_process.stopped"})
        sleep(1.0)


def player_drop(
    rec_dir, ipc_pub_url, ipc_sub_url, ipc_push_url, user_dir, app_version, debug
):
    # general imports
    import logging
    from time import sleep

    # networking
    import zmq
    import zmq_tools

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
        from plugin import import_runtime_plugins

        import_runtime_plugins(os.path.join(user_dir, "plugins", "recording-upgrades"))

        import glfw
        from gl_utils import GLFWErrorReporting

        GLFWErrorReporting.set_default()

        import gl_utils
        import player_methods as pm
        from file_methods import Persistent_Dict
        from OpenGL.GL import glClearColor
        from pupil_recording import (
            InvalidRecordingException,
            assert_valid_recording_type,
        )
        from pupil_recording.update import update_recording
        from pyglui.pyfontstash import fontstash
        from pyglui.ui import get_roboto_font_path
        from version_utils import parse_version

        process_was_interrupted = False

        def interrupt_handler(sig, frame):
            import traceback

            trace = traceback.format_stack(f=frame)
            logger.debug(f"Caught signal {sig} in:\n" + "".join(trace))
            nonlocal process_was_interrupted
            process_was_interrupted = True

        signal.signal(signal.SIGINT, interrupt_handler)

        def on_drop(window, paths):
            nonlocal rec_dir
            rec_dir = paths[0]

        if rec_dir:
            try:
                assert_valid_recording_type(rec_dir)
            except InvalidRecordingException as err:
                logger.exception(str(err))
                rec_dir = None
        # load session persistent settings
        session_settings = Persistent_Dict(
            os.path.join(user_dir, "user_settings_player")
        )
        if parse_version(session_settings.get("version", "0.0")) != app_version:
            logger.info(
                "Session setting are from a  different version of this app. I will not use those."
            )
            session_settings.clear()
        w, h = session_settings.get("window_size", (1280, 720))

        glfw.init()
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, 0)
        window = glfw.create_window(w, h, "Pupil Player", None, None)
        glfw.window_hint(glfw.RESIZABLE, 1)

        glfw.make_context_current(window)

        window_position_manager = gl_utils.WindowPositionManager()
        window_pos = window_position_manager.new_window_position(
            window=window,
            default_position=window_position_default,
            previous_position=session_settings.get("window_position", None),
        )
        glfw.set_window_pos(window, window_pos[0], window_pos[1])

        glfw.set_drop_callback(window, on_drop)

        glfont = fontstash.Context()
        glfont.add_font("roboto", get_roboto_font_path())
        glfont.set_align_string(v_align="center", h_align="middle")
        glfont.set_color_float((0.2, 0.2, 0.2, 0.9))
        gl_utils.basic_gl_setup()
        glClearColor(0.5, 0.5, 0.5, 0.0)
        text = "Drop a recording directory onto this window."
        tip = "(Tip: You can drop a recording directory onto the app icon.)"
        # text = "Please supply a Pupil recording directory as first arg when calling Pupil Player."

        def display_string(string, font_size, center_y):
            x = w / 2 * content_scale
            y = center_y * content_scale

            glfont.set_size(font_size * content_scale)

            glfont.set_blur(10.5)
            glfont.set_color_float((0.0, 0.0, 0.0, 1.0))
            glfont.draw_text(x, y, string)

            glfont.set_blur(0.96)
            glfont.set_color_float((1.0, 1.0, 1.0, 1.0))
            glfont.draw_text(x, y, string)

        def display_multiline_string(string, font_size, top_y, split_chr="\n"):
            for idx, line in enumerate(string.split(split_chr)):
                center_y = top_y + font_size * idx * 1.2
                display_string(line, font_size=font_size, center_y=center_y)
            bottom_y = top_y + font_size * (idx + 1) * 1.2
            return bottom_y

        while not glfw.window_should_close(window) and not process_was_interrupted:
            fb_size = glfw.get_framebuffer_size(window)
            content_scale = gl_utils.get_content_scale(window)
            gl_utils.adjust_gl_view(*fb_size)

            if rec_dir:
                try:
                    assert_valid_recording_type(rec_dir)
                    logger.info(f"Starting new session with '{rec_dir}'")
                    text = "Updating recording format\n."
                    tip = "This may take a while!"
                except InvalidRecordingException as err:
                    logger.exception(str(err))
                    if err.recovery:
                        text = err.reason
                        tip = err.recovery
                    else:
                        text = "Invalid recording"
                        tip = err.reason
                    rec_dir = None

            gl_utils.clear_gl_screen()

            top_y = display_multiline_string(text, font_size=51, top_y=216)
            display_multiline_string(tip, font_size=42, top_y=top_y + 50)

            glfw.swap_buffers(window)

            if rec_dir:
                try:

                    async def update_ui():
                        nonlocal text
                        indefinitive_progress = 0
                        base_text = text
                        while True:
                            text = base_text + "." * indefinitive_progress
                            indefinitive_progress += 1
                            indefinitive_progress %= 3

                            gl_utils.clear_gl_screen()
                            top_y = display_multiline_string(
                                text, font_size=51, top_y=216
                            )
                            display_multiline_string(
                                tip, font_size=42, top_y=top_y + 50
                            )
                            glfw.swap_buffers(window)
                            glfw.poll_events()
                            await asyncio.sleep(1.0)

                    async def update_recording_without_blocking_ui():
                        updating_ui = asyncio.create_task(update_ui())

                        loop = asyncio.get_running_loop()
                        update_this_recording = partial(update_recording, rec_dir)
                        await loop.run_in_executor(None, update_this_recording)

                    asyncio.run(update_recording_without_blocking_ui())

                except AssertionError as err:
                    logger.exception(str(err))
                    tip = "Oops! There was an error updating the recording."
                    rec_dir = None
                except InvalidRecordingException as err:
                    logger.exception(str(err))
                    if err.recovery:
                        text = err.reason
                        tip = err.recovery
                    else:
                        text = "Invalid recording"
                        tip = err.reason
                    rec_dir = None
                else:
                    glfw.set_window_should_close(window, True)

            glfw.poll_events()

        session_settings["window_position"] = glfw.get_window_pos(window)
        session_settings.close()
        glfw.destroy_window(window)
        if rec_dir:
            ipc_pub.notify(
                {"subject": "player_process.should_start", "rec_dir": rec_dir}
            )

    except Exception:
        logger.exception("Process player_drop crashed with trace:")
        raise

    finally:
        sleep(1.0)


def player_profiled(
    rec_dir, ipc_pub_url, ipc_sub_url, ipc_push_url, user_dir, app_version, debug
):
    import cProfile
    import os
    import subprocess

    from .player import player

    cProfile.runctx(
        "player(rec_dir, ipc_pub_url, ipc_sub_url, ipc_push_url, user_dir, app_version, debug)",
        {
            "rec_dir": rec_dir,
            "ipc_pub_url": ipc_pub_url,
            "ipc_sub_url": ipc_sub_url,
            "ipc_push_url": ipc_push_url,
            "user_dir": user_dir,
            "app_version": app_version,
            "debug": debug,
        },
        locals(),
        "player.pstats",
    )
    loc = os.path.abspath(__file__).rsplit("pupil_src", 1)
    gprof2dot_loc = os.path.join(loc[0], "pupil_src", "shared_modules", "gprof2dot.py")
    subprocess.call(
        "python "
        + gprof2dot_loc
        + " -f pstats player.pstats | dot -Tpng -o player_cpu_time.png",
        shell=True,
    )
    print(
        "created cpu time graph for world process. Please check out the png next to the player.py file"
    )
