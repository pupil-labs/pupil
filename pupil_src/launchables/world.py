"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
import platform
import signal
from types import SimpleNamespace


def world(
    timebase,
    eye_procs_alive,
    ipc_pub_url,
    ipc_sub_url,
    ipc_push_url,
    user_dir,
    version,
    preferred_remote_port,
    hide_ui,
    debug,
):
    """Reads world video and runs plugins.

    Creates a window, gl context.
    Grabs images from a capture.
    Maps pupil to gaze data
    Can run various plug-ins.

    Reacts to notifications:
        ``eye_process.started``
        ``start_plugin``
        ``should_stop``

    Emits notifications:
        ``eye_process.should_start``
        ``eye_process.should_stop``
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
    notify_sub = zmq_tools.Msg_Receiver(zmq_ctx, ipc_sub_url, topics=("notify",))

    # log setup
    logging.getLogger("OpenGL").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    def launch_eye_process(eye_id, delay=0):
        n = {
            "subject": "eye_process.should_start.{}".format(eye_id),
            "eye_id": eye_id,
            "delay": delay,
        }
        ipc_pub.notify(n)

    def stop_eye_process(eye_id):
        n = {
            "subject": "eye_process.should_stop.{}".format(eye_id),
            "eye_id": eye_id,
            "delay": 0.2,
        }
        ipc_pub.notify(n)

    def start_stop_eye(eye_id, make_alive):
        if make_alive:
            launch_eye_process(eye_id)
        else:
            stop_eye_process(eye_id)

    def detection_enabled_getter() -> bool:
        return g_pool.pupil_detection_enabled

    def detection_enabled_setter(is_on: bool):
        g_pool.pupil_detection_enabled = is_on
        n = {"subject": "set_pupil_detection_enabled", "value": is_on}
        ipc_pub.notify(n)

    try:
        from background_helper import IPC_Logging_Task_Proxy

        IPC_Logging_Task_Proxy.push_url = ipc_push_url

        from tasklib.background.patches import IPCLoggingPatch

        IPCLoggingPatch.ipc_push_url = ipc_push_url

        from OpenGL.GL import GL_COLOR_BUFFER_BIT

        # display
        import glfw
        from gl_utils import GLFWErrorReporting

        GLFWErrorReporting.set_default()

        from version_utils import parse_version
        from pyglui import ui, cygl, __version__ as pyglui_version

        assert parse_version(pyglui_version) >= parse_version(
            "1.27"
        ), "pyglui out of date, please upgrade to newest version"
        from pyglui.cygl.utils import Named_Texture
        import gl_utils

        # helpers/utils
        from file_methods import Persistent_Dict
        from methods import normalize, denormalize, delta_t, get_system_info, timer
        from uvc import get_time_monotonic

        logger.info("Application Version: {}".format(version))
        logger.info("System Info: {}".format(get_system_info()))
        logger.debug(f"Debug flag: {debug}")

        import audio

        # Plug-ins
        from plugin import (
            Plugin,
            System_Plugin_Base,
            Plugin_List,
            import_runtime_plugins,
        )
        from plugin_manager import Plugin_Manager
        from calibration_choreography import (
            available_calibration_choreography_plugins,
            CalibrationChoreographyPlugin,
            patch_loaded_plugins_with_choreography_plugin,
        )

        available_choreography_plugins = available_calibration_choreography_plugins()

        from gaze_mapping import registered_gazer_classes
        from gaze_mapping.gazer_base import GazerBase
        from pupil_detector_plugins.detector_base_plugin import PupilDetectorPlugin
        from fixation_detector import Fixation_Detector
        from recorder import Recorder
        from display_recent_gaze import Display_Recent_Gaze
        from time_sync import Time_Sync
        from network_api import NetworkApiPlugin
        from pupil_groups import Pupil_Groups
        from surface_tracker import Surface_Tracker_Online
        from log_display import Log_Display
        from annotations import Annotation_Capture
        from log_history import Log_History
        from blink_detection import Blink_Detection
        from video_capture import (
            source_classes,
            manager_classes,
            Base_Manager,
            Base_Source,
        )
        from pupil_data_relay import Pupil_Data_Relay
        from remote_recorder import Remote_Recorder
        from accuracy_visualizer import Accuracy_Visualizer

        from system_graphs import System_Graphs
        from camera_intrinsics_estimation import Camera_Intrinsics_Estimation
        from hololens_relay import Hololens_Relay
        from head_pose_tracker.online_head_pose_tracker import Online_Head_Pose_Tracker

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

        process_was_interrupted = False

        def interrupt_handler(sig, frame):
            import traceback

            trace = traceback.format_stack(f=frame)
            logger.debug(f"Caught signal {sig} in:\n" + "".join(trace))
            nonlocal process_was_interrupted
            process_was_interrupted = True

        signal.signal(signal.SIGINT, interrupt_handler)

        icon_bar_width = 50
        window_size = None
        camera_render_size = None
        content_scale = 1.0

        # g_pool holds variables for this process they are accessible to all plugins
        g_pool = SimpleNamespace()
        g_pool.debug = debug
        g_pool.app = "capture"
        g_pool.process = "world"
        g_pool.user_dir = user_dir
        g_pool.version = version
        g_pool.timebase = timebase
        g_pool.zmq_ctx = zmq_ctx
        g_pool.ipc_pub = ipc_pub
        g_pool.ipc_pub_url = ipc_pub_url
        g_pool.ipc_sub_url = ipc_sub_url
        g_pool.ipc_push_url = ipc_push_url
        g_pool.eye_procs_alive = eye_procs_alive
        g_pool.preferred_remote_port = preferred_remote_port

        def get_timestamp():
            return get_time_monotonic() - g_pool.timebase.value

        g_pool.get_timestamp = get_timestamp
        g_pool.get_now = get_time_monotonic

        # manage plugins
        runtime_plugins = import_runtime_plugins(
            os.path.join(g_pool.user_dir, "plugins")
        )
        runtime_plugins = [
            p for p in runtime_plugins if not issubclass(p, PupilDetectorPlugin)
        ]
        user_plugins = [
            Pupil_Groups,
            NetworkApiPlugin,
            Time_Sync,
            Surface_Tracker_Online,
            Annotation_Capture,
            Log_History,
            Fixation_Detector,
            Blink_Detection,
            Remote_Recorder,
            Accuracy_Visualizer,
            Camera_Intrinsics_Estimation,
            Hololens_Relay,
            Online_Head_Pose_Tracker,
        ]

        system_plugins = (
            [
                Log_Display,
                Display_Recent_Gaze,
                Recorder,
                Pupil_Data_Relay,
                Plugin_Manager,
                System_Graphs,
            ]
            + manager_classes
            + source_classes
        )
        plugins = (
            system_plugins
            + user_plugins
            + runtime_plugins
            + available_choreography_plugins
            + registered_gazer_classes()
        )
        user_plugins += [
            p
            for p in runtime_plugins
            if not isinstance(
                p,
                (
                    Base_Manager,
                    Base_Source,
                    System_Plugin_Base,
                    CalibrationChoreographyPlugin,
                    GazerBase,
                ),
            )
        ]
        g_pool.plugin_by_name = {p.__name__: p for p in plugins}

        default_capture_name = "UVC_Source"
        default_capture_settings = {
            "preferred_names": [
                "Pupil Cam1 ID2",
                "Logitech Camera",
                "(046d:081d)",
                "C510",
                "B525",
                "C525",
                "C615",
                "C920",
                "C930e",
            ],
            "frame_size": (1280, 720),
            "frame_rate": 30,
        }

        default_plugins = [
            (default_capture_name, default_capture_settings),
            ("Pupil_Data_Relay", {}),
            ("UVC_Manager", {}),
            ("NDSI_Manager", {}),
            ("HMD_Streaming_Manager", {}),
            ("File_Manager", {}),
            ("Log_Display", {}),
            ("Dummy_Gaze_Mapper", {}),
            ("Display_Recent_Gaze", {}),
            # Calibration choreography plugin is added below by calling
            # patch_loaded_plugins_with_choreography_plugin
            ("Recorder", {}),
            ("NetworkApiPlugin", {}),
            ("Fixation_Detector", {}),
            ("Blink_Detection", {}),
            ("Accuracy_Visualizer", {}),
            ("Plugin_Manager", {}),
            ("System_Graphs", {}),
        ]

        def consume_events_and_render_buffer():
            gl_utils.glViewport(0, 0, *camera_render_size)
            for p in g_pool.plugins:
                p.gl_display()

            gl_utils.glViewport(0, 0, *window_size)
            try:
                clipboard = glfw.get_clipboard_string(main_window).decode()
            except (AttributeError, glfw.GLFWError):
                # clipboard is None, might happen on startup
                clipboard = ""
            g_pool.gui.update_clipboard(clipboard)
            user_input = g_pool.gui.update()
            if user_input.clipboard != clipboard:
                # only write to clipboard if content changed
                glfw.set_clipboard_string(main_window, user_input.clipboard)

            for button, action, mods in user_input.buttons:
                x, y = glfw.get_cursor_pos(main_window)
                pos = gl_utils.window_coordinate_to_framebuffer_coordinate(
                    main_window, x, y, cached_scale=None
                )
                pos = normalize(pos, camera_render_size)
                # Position in img pixels
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
            nonlocal camera_render_size
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
            camera_render_size = w - int(icon_bar_width * g_pool.gui.scale), h
            g_pool.gui.update_window(*window_size)
            g_pool.gui.collect_menus()

            for p in g_pool.plugins:
                p.on_window_resize(window, *camera_render_size)

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
            pos = normalize(pos, camera_render_size)
            # Position in img pixels
            pos = denormalize(pos, g_pool.capture.frame_size)
            for p in g_pool.plugins:
                p.on_pos(pos)

        def on_scroll(window, x, y):
            g_pool.gui.update_scroll(x, y * scroll_factor)

        def on_drop(window, paths):
            for plugin in g_pool.plugins:
                if plugin.on_drop(paths):
                    break

        tick = delta_t()

        def get_dt():
            return next(tick)

        # load session persistent settings
        session_settings = Persistent_Dict(
            os.path.join(g_pool.user_dir, "user_settings_world")
        )
        if parse_version(session_settings.get("version", "0.0")) != g_pool.version:
            logger.info(
                "Session setting are from a different version of this app. I will not use those."
            )
            session_settings.clear()

        g_pool.min_data_confidence = 0.6
        g_pool.min_calibration_confidence = session_settings.get(
            "min_calibration_confidence", 0.8
        )
        g_pool.pupil_detection_enabled = session_settings.get(
            "pupil_detection_enabled", True
        )
        g_pool.active_gaze_mapping_plugin = None
        g_pool.capture = None

        audio.set_audio_mode(
            session_settings.get("audio_mode", audio.get_default_audio_mode())
        )

        def handle_notifications(noti):
            subject = noti["subject"]
            if subject == "set_pupil_detection_enabled":
                g_pool.pupil_detection_enabled = noti["value"]
            elif subject == "start_plugin":
                try:
                    g_pool.plugins.add(
                        g_pool.plugin_by_name[noti["name"]], args=noti.get("args", {})
                    )
                except KeyError as err:
                    logger.error(f"Attempt to load unknown plugin: {err}")
            elif subject == "stop_plugin":
                for p in g_pool.plugins:
                    if p.class_name == noti["name"]:
                        p.alive = False
                        g_pool.plugins.clean()
            elif subject == "eye_process.started":
                noti = {
                    "subject": "set_pupil_detection_enabled",
                    "value": g_pool.pupil_detection_enabled,
                }
                ipc_pub.notify(noti)
            elif subject == "set_min_calibration_confidence":
                g_pool.min_calibration_confidence = noti["value"]
            elif subject.startswith("meta.should_doc"):
                ipc_pub.notify(
                    {"subject": "meta.doc", "actor": g_pool.app, "doc": world.__doc__}
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
            elif subject == "world_process.adapt_window_size":
                set_window_size()
            elif subject == "world_process.should_stop":
                glfw.set_window_should_close(main_window, True)

        width, height = session_settings.get(
            "window_size", (1280 + icon_bar_width, 720)
        )

        # window and gl setup
        glfw.init()
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
        if hide_ui:
            glfw.window_hint(glfw.VISIBLE, 0)  # hide window
        main_window = glfw.create_window(
            width, height, "Pupil Capture - World", None, None
        )

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

        def reset_restart():
            logger.warning("Resetting all settings and restarting Capture.")
            glfw.set_window_should_close(main_window, True)
            ipc_pub.notify({"subject": "clear_settings_process.should_start"})
            ipc_pub.notify({"subject": "world_process.should_start", "delay": 2.0})

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
        g_pool.menubar = ui.Scrolling_Menu(
            "Settings", pos=(-400, 0), size=(-icon_bar_width, 0), header_pos="left"
        )
        g_pool.iconbar = ui.Scrolling_Menu(
            "Icons", pos=(-icon_bar_width, 0), size=(0, 0), header_pos="hidden"
        )
        g_pool.quickbar = ui.Stretching_Menu("Quick Bar", (0, 100), (120, -100))
        g_pool.gui.append(g_pool.menubar)
        g_pool.gui.append(g_pool.iconbar)
        g_pool.gui.append(g_pool.quickbar)

        general_settings = ui.Growing_Menu("General", header_pos="headline")

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

        general_settings.append(ui.Button("Reset window size", set_window_size))
        general_settings.append(
            ui.Selector(
                "Audio mode",
                None,
                getter=audio.get_audio_mode,
                setter=audio.set_audio_mode,
                selection=audio.get_audio_mode_list(),
            )
        )

        general_settings.append(
            ui.Switch(
                "pupil_detection_enabled",
                label="Pupil detection",
                getter=detection_enabled_getter,
                setter=detection_enabled_setter,
            )
        )
        general_settings.append(
            ui.Switch(
                "eye0_process",
                label="Detect eye 0",
                setter=lambda alive: start_stop_eye(0, alive),
                getter=lambda: eye_procs_alive[0].value,
            )
        )
        general_settings.append(
            ui.Switch(
                "eye1_process",
                label="Detect eye 1",
                setter=lambda alive: start_stop_eye(1, alive),
                getter=lambda: eye_procs_alive[1].value,
            )
        )

        general_settings.append(
            ui.Info_Text("Capture Version: {}".format(g_pool.version))
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

        loaded_plugins = session_settings.get("loaded_plugins", default_plugins)

        # Resolve the active calibration choreography plugin
        loaded_plugins = patch_loaded_plugins_with_choreography_plugin(
            loaded_plugins, app=g_pool.app
        )
        session_settings["loaded_plugins"] = loaded_plugins

        # plugins that are loaded based on user settings from previous session
        g_pool.plugins = Plugin_List(g_pool, loaded_plugins)

        if not g_pool.capture:
            # Make sure we always have a capture running. Important if there was no
            # capture stored in session settings.
            g_pool.plugins.add(
                g_pool.plugin_by_name[default_capture_name], default_capture_settings
            )

        # Register callbacks main_window
        glfw.set_framebuffer_size_callback(main_window, on_resize)
        glfw.set_key_callback(main_window, on_window_key)
        glfw.set_char_callback(main_window, on_window_char)
        glfw.set_mouse_button_callback(main_window, on_window_mouse_button)
        glfw.set_cursor_pos_callback(main_window, on_pos)
        glfw.set_scroll_callback(main_window, on_scroll)
        glfw.set_drop_callback(main_window, on_drop)

        # gl_state settings
        gl_utils.basic_gl_setup()
        g_pool.image_tex = Named_Texture()

        toggle_general_settings(True)

        # now that we have a proper window we can load the last gui configuration
        g_pool.gui.configuration = session_settings.get("ui_config", {})
        # If previously selected plugin was not loaded this time, we will have an
        # expanded menubar without any menu selected. We need to ensure the menubar is
        # collapsed in this case.
        if all(submenu.collapsed for submenu in g_pool.menubar.elements):
            g_pool.menubar.collapsed = True

        # create a timer to control window update frequency
        window_update_timer = timer(1 / 60)

        def window_should_update():
            return next(window_update_timer)

        # trigger setup of window and gl sizes
        on_resize(main_window, *glfw.get_framebuffer_size(main_window))

        if session_settings.get("eye1_process_alive", True):
            launch_eye_process(1, delay=0.6)
        if session_settings.get("eye0_process_alive", True):
            launch_eye_process(0, delay=0.3)

        ipc_pub.notify({"subject": "world_process.started"})
        logger.warning("Process started.")

        if platform.system() == "Darwin":
            # On macOS, calls to glfw.swap_buffers() deliberately take longer in case of
            # occluded windows, based on the swap interval value. This causes an FPS drop
            # and leads to problems when recording. To side-step this behaviour, the swap
            # interval is set to zero.
            #
            # Read more about window occlusion on macOS here:
            # https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/WorkWhenVisible.html
            glfw.swap_interval(0)

        # Event loop
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

            # a dictionary that allows plugins to post and read events
            events = {}
            # report time between now and the last loop interation
            events["dt"] = get_dt()

            # allow each Plugin to do its work.
            for p in g_pool.plugins:
                p.recent_events(events)

            # check if a plugin need to be destroyed
            g_pool.plugins.clean()

            # "blacklisted" events that were already sent
            del events["pupil"]
            del events["gaze"]
            # delete if exists. More expensive than del, so only use it when key might not exist
            events.pop("annotation", None)

            # send new events to ipc:
            if "frame" in events:
                del events["frame"]  # send explicitly with frame publisher
            if "depth_frame" in events:
                del events["depth_frame"]
            if "audio_packets" in events:
                del events["audio_packets"]
            del events["dt"]  # no need to send this
            for data in events.values():
                assert isinstance(data, (list, tuple))
                for d in data:
                    ipc_pub.send(d)

            glfw.make_context_current(main_window)
            # render visual feedback from loaded plugins
            glfw.poll_events()
            if window_should_update() and gl_utils.is_window_visible(main_window):

                gl_utils.glViewport(0, 0, *camera_render_size)
                for p in g_pool.plugins:
                    p.gl_display()

                gl_utils.glViewport(0, 0, *window_size)
                try:
                    clipboard = glfw.get_clipboard_string(main_window).decode()
                except (AttributeError, glfw.GLFWError):
                    # clipboard is None, might happen on startup
                    clipboard = ""
                g_pool.gui.update_clipboard(clipboard)
                user_input = g_pool.gui.update()
                if user_input.clipboard != clipboard:
                    # only write to clipboard if content changed
                    glfw.set_clipboard_string(main_window, user_input.clipboard)

                for button, action, mods in user_input.buttons:
                    x, y = glfw.get_cursor_pos(main_window)
                    pos = gl_utils.window_coordinate_to_framebuffer_coordinate(
                        main_window, x, y, cached_scale=None
                    )
                    pos = normalize(pos, camera_render_size)
                    # Position in img pixels
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

        session_settings["loaded_plugins"] = g_pool.plugins.get_initializers()
        session_settings["ui_config"] = g_pool.gui.configuration
        session_settings["version"] = str(g_pool.version)
        session_settings["eye0_process_alive"] = eye_procs_alive[0].value
        session_settings["eye1_process_alive"] = eye_procs_alive[1].value
        session_settings[
            "min_calibration_confidence"
        ] = g_pool.min_calibration_confidence
        session_settings["pupil_detection_enabled"] = g_pool.pupil_detection_enabled
        session_settings["audio_mode"] = audio.get_audio_mode()

        if not hide_ui:
            glfw.restore_window(main_window)  # need to do this for windows os
            session_settings["window_position"] = glfw.get_window_pos(main_window)
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
        glfw.terminate()

    except Exception:
        import traceback

        trace = traceback.format_exc()
        logger.error("Process Capture crashed with trace:\n{}".format(trace))

    finally:
        # shut down eye processes:
        stop_eye_process(0)
        stop_eye_process(1)

        logger.info("Process shutting down.")
        ipc_pub.notify({"subject": "world_process.stopped"})
        sleep(1.0)


def world_profiled(
    timebase,
    eye_procs_alive,
    ipc_pub_url,
    ipc_sub_url,
    ipc_push_url,
    user_dir,
    version,
    preferred_remote_port,
    hide_ui,
    debug,
):
    import cProfile
    import subprocess
    import os
    from .world import world

    cProfile.runctx(
        "world(timebase, eye_procs_alive, ipc_pub_url,ipc_sub_url,ipc_push_url,user_dir,version,preferred_remote_port, hide_ui, debug)",
        {
            "timebase": timebase,
            "eye_procs_alive": eye_procs_alive,
            "ipc_pub_url": ipc_pub_url,
            "ipc_sub_url": ipc_sub_url,
            "ipc_push_url": ipc_push_url,
            "user_dir": user_dir,
            "version": version,
            "preferred_remote_port": preferred_remote_port,
            "hide_ui": hide_ui,
            "debug": debug,
        },
        locals(),
        "world.pstats",
    )
    loc = os.path.abspath(__file__).rsplit("pupil_src", 1)
    gprof2dot_loc = os.path.join(loc[0], "pupil_src", "shared_modules", "gprof2dot.py")
    subprocess.call(
        "python "
        + gprof2dot_loc
        + " -f pstats world.pstats | dot -Tpng -o world_cpu_time.png",
        shell=True,
    )
    print(
        "created cpu time graph for world process. Please check out the png next to the world.py file"
    )
