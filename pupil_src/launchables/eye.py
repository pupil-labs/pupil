"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
import platform
import signal
import time
from types import SimpleNamespace


class Is_Alive_Manager:
    """
    A context manager to wrap the is_alive flag.
    Is alive will stay true as long is the eye process is running.
    """

    def __init__(self, is_alive, ipc_socket, eye_id, logger):
        self.is_alive = is_alive
        self.ipc_socket = ipc_socket
        self.eye_id = eye_id
        self.logger = logger

    def __enter__(self):
        self.is_alive.value = True
        self.ipc_socket.notify(
            {"subject": "eye_process.started", "eye_id": self.eye_id}
        )
        return self

    def __exit__(self, etype, value, traceback):
        if etype is not None:
            import traceback as tb

            self.logger.error(
                f"Process Eye{self.eye_id} crashed with trace:\n"
                + "".join(tb.format_exception(etype, value, traceback))
            )

        self.is_alive.value = False
        self.ipc_socket.notify(
            {"subject": "eye_process.stopped", "eye_id": self.eye_id}
        )
        time.sleep(1.0)
        return True  # do not propagate exception


def eye(
    timebase,
    is_alive_flag,
    ipc_pub_url,
    ipc_sub_url,
    ipc_push_url,
    user_dir,
    version,
    eye_id,
    overwrite_cap_settings=None,
    hide_ui=False,
    debug=False,
    pub_socket_hwm=None,
    parent_application="capture",
    skip_driver_installation=False,
):
    """reads eye video and detects the pupil.

    Creates a window, gl context.
    Grabs images from a capture.
    Streams Pupil coordinates.

    Reacts to notifications:
        ``eye_process.should_stop``: Stops the eye process
        ``recording.started``: Starts recording eye video
        ``recording.stopped``: Stops recording eye video
        ``frame_publishing.started``: Starts frame publishing
        ``frame_publishing.stopped``: Stops frame publishing
        ``start_eye_plugin``: Start plugins in eye process

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
    pupil_socket = zmq_tools.Msg_Streamer(zmq_ctx, ipc_pub_url, pub_socket_hwm)
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
        logger.warning("Aborting redundant eye process startup")
        return

    with Is_Alive_Manager(is_alive_flag, ipc_socket, eye_id, logger):
        # general imports
        import traceback

        import cv2

        # display
        import glfw
        import numpy as np
        from gl_utils import GLFWErrorReporting
        from OpenGL.GL import GL_COLOR_BUFFER_BIT

        GLFWErrorReporting.set_default()

        import gl_utils

        # monitoring
        import psutil
        from av_writer import JPEG_Writer, MPEG_Writer, NonMonotonicTimestampError
        from background_helper import IPC_Logging_Task_Proxy
        from file_methods import Persistent_Dict
        from gl_utils import (
            adjust_gl_view,
            basic_gl_setup,
            clear_gl_screen,
            glViewport,
            is_window_visible,
            make_coord_system_norm_based,
            make_coord_system_pixel_based,
        )
        from methods import denormalize, normalize, timer
        from ndsi import H264Writer

        # Plug-ins
        from plugin import Plugin_List
        from pupil_detector_plugins import EVENT_KEY, available_detector_plugins
        from pyglui import cygl, graph, ui
        from pyglui.cygl.utils import Named_Texture
        from roi import Roi

        # helpers/utils
        from uvc import get_time_monotonic
        from version_utils import parse_version
        from video_capture import manager_classes, source_classes

        IPC_Logging_Task_Proxy.push_url = ipc_push_url

        def interrupt_handler(sig, frame):
            import traceback

            trace = traceback.format_stack(f=frame)
            logger.debug(f"Caught signal {sig} in:\n" + "".join(trace))
            # NOTE: Interrupt is handled in world/service/player which are responsible for
            # shutting down the eye process properly

        signal.signal(signal.SIGINT, interrupt_handler)

        # UI Platform tweaks
        if platform.system() == "Linux":
            scroll_factor = 10.0
            window_position_default = (600, 300 * eye_id + 30)
        elif platform.system() == "Windows":
            scroll_factor = 10.0
            window_position_default = (600, 90 + 300 * eye_id)
        else:
            scroll_factor = 1.0
            window_position_default = (600, 300 * eye_id)

        icon_bar_width = 50
        window_size = None
        content_scale = 1.0

        # g_pool holds variables for this process
        g_pool = SimpleNamespace()

        # make some constants avaiable
        g_pool.debug = debug
        g_pool.user_dir = user_dir
        g_pool.version = version
        g_pool.app = parent_application
        g_pool.eye_id = eye_id
        g_pool.process = f"eye{eye_id}"
        g_pool.timebase = timebase
        g_pool.camera_render_size = None
        g_pool.skip_driver_installation = skip_driver_installation

        g_pool.zmq_ctx = zmq_ctx
        g_pool.ipc_pub = ipc_socket
        g_pool.ipc_pub_url = ipc_pub_url
        g_pool.ipc_sub_url = ipc_sub_url
        g_pool.ipc_push_url = ipc_push_url

        def get_timestamp():
            return get_time_monotonic() - g_pool.timebase.value

        g_pool.get_timestamp = get_timestamp
        g_pool.get_now = get_time_monotonic

        def load_runtime_pupil_detection_plugins():
            from plugin import import_runtime_plugins
            from pupil_detector_plugins.detector_base_plugin import PupilDetectorPlugin

            plugins_path = os.path.join(g_pool.user_dir, "plugins")

            for plugin in import_runtime_plugins(plugins_path):
                if not isinstance(plugin, type):
                    continue
                if not issubclass(plugin, PupilDetectorPlugin):
                    continue
                if plugin is PupilDetectorPlugin:
                    continue
                yield plugin

        available_detectors = available_detector_plugins()
        runtime_detectors = list(load_runtime_pupil_detection_plugins())
        plugins = (
            manager_classes
            + source_classes
            + available_detectors
            + runtime_detectors
            + [Roi]
        )
        g_pool.plugin_by_name = {p.__name__: p for p in plugins}

        preferred_names = [
            f"Pupil Cam3 ID{eye_id}",
            f"Pupil Cam2 ID{eye_id}",
            f"Pupil Cam1 ID{eye_id}",
        ]
        if eye_id == 0:
            preferred_names += ["HD-6000"]

        default_capture_name = "UVC_Source"
        default_capture_settings = {
            "preferred_names": preferred_names,
            "frame_size": (192, 192),
            "frame_rate": 120,
        }

        default_plugins = [
            # TODO: extend with plugins
            (default_capture_name, default_capture_settings),
            ("UVC_Manager", {}),
            *[(p.__name__, {}) for p in available_detectors],
            ("NDSI_Manager", {}),
            ("HMD_Streaming_Manager", {}),
            ("File_Manager", {}),
            ("Roi", {}),
        ]

        def consume_events_and_render_buffer():
            glfw.make_context_current(main_window)
            clear_gl_screen()

            if all(c > 0 for c in g_pool.camera_render_size):
                glViewport(0, 0, *g_pool.camera_render_size)
                for p in g_pool.plugins:
                    p.gl_display()

            glViewport(0, 0, *window_size)
            # render graphs
            fps_graph.draw()
            cpu_graph.draw()

            # render GUI
            try:
                clipboard = glfw.get_clipboard_string(None).decode()
            except (AttributeError, glfw.GLFWError):
                # clipboard is None, might happen on startup
                clipboard = ""
            g_pool.gui.update_clipboard(clipboard)
            user_input = g_pool.gui.update()
            if user_input.clipboard != clipboard:
                # only write to clipboard if content changed
                glfw.set_clipboard_string(None, user_input.clipboard)

            for button, action, mods in user_input.buttons:
                x, y = glfw.get_cursor_pos(main_window)
                pos = gl_utils.window_coordinate_to_framebuffer_coordinate(
                    main_window, x, y, cached_scale=None
                )
                pos = normalize(pos, g_pool.camera_render_size)
                if g_pool.flip:
                    pos = 1 - pos[0], 1 - pos[1]
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

            # update screen
            glfw.swap_buffers(main_window)

        # Callback functions
        def on_resize(window, w, h):
            nonlocal window_size
            nonlocal content_scale

            is_minimized = bool(glfw.get_window_attrib(window, glfw.ICONIFIED))

            if is_minimized:
                return

            # Always clear buffers on resize to make sure that there are no overlapping
            # artifacts from previous frames.
            gl_utils.glClear(GL_COLOR_BUFFER_BIT)
            gl_utils.glClearColor(0, 0, 0, 1)

            active_window = glfw.get_current_context()
            glfw.make_context_current(window)
            content_scale = gl_utils.get_content_scale(window)
            framebuffer_scale = gl_utils.get_framebuffer_scale(window)
            g_pool.gui.scale = content_scale
            window_size = w, h
            g_pool.camera_render_size = w - int(icon_bar_width * g_pool.gui.scale), h
            g_pool.gui.update_window(w, h)
            g_pool.gui.collect_menus()
            for g in g_pool.graphs:
                g.scale = content_scale
                g.adjust_window_size(w, h)
            adjust_gl_view(w, h)
            glfw.make_context_current(active_window)

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

        def on_iconify(window, iconified):
            g_pool.iconified = iconified

        def on_window_mouse_button(window, button, action, mods):
            g_pool.gui.update_button(button, action, mods)

        def on_pos(window, x, y):
            x, y = gl_utils.window_coordinate_to_framebuffer_coordinate(
                window, x, y, cached_scale=None
            )
            g_pool.gui.update_mouse(x, y)

            pos = x, y
            pos = normalize(pos, g_pool.camera_render_size)
            if g_pool.flip:
                pos = 1 - pos[0], 1 - pos[1]
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

        # load session persistent settings
        session_settings = Persistent_Dict(
            os.path.join(g_pool.user_dir, f"user_settings_eye{eye_id}")
        )
        if parse_version(session_settings.get("version", "0.0")) != g_pool.version:
            logger.debug(
                "Session setting are from a different version of this app. I will not use those."
            )
            session_settings.clear()

        camera_is_physically_flipped = eye_id == 0
        g_pool.iconified = False
        g_pool.capture = None
        g_pool.flip = session_settings.get("flip", camera_is_physically_flipped)
        g_pool.display_mode = session_settings.get("display_mode", "camera_image")
        g_pool.display_mode_info_text = {
            "camera_image": "Raw eye camera image. This uses the least amount of CPU power",
            "roi": "Click and drag on the blue circles to adjust the region of interest. The region should be as small as possible, but large enough to capture all pupil movements.",
            "algorithm": "Algorithm display mode overlays a visualization of the pupil detection parameters on top of the eye video. Adjust parameters within the Pupil Detection menu below.",
        }

        def set_display_mode_info(val):
            g_pool.display_mode = val
            g_pool.display_mode_info.text = g_pool.display_mode_info_text[val]

        def toggle_general_settings(collapsed):
            # this is the menu toggle logic.
            # Only one menu can be open.
            # If no menu is open the menubar should collapse.
            g_pool.menubar.collapsed = collapsed
            for m in g_pool.menubar.elements:
                m.collapsed = True
            general_settings.collapsed = collapsed

        # Initialize glfw
        glfw.init()
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
        if hide_ui:
            glfw.window_hint(glfw.VISIBLE, 0)  # hide window
        title = f"Pupil Capture - eye {eye_id}"

        # Pupil Cam1 uses 4:3 resolutions. Pupil Cam2 and Cam3 use 1:1 resolutions.
        # As all Pupil Core and VR/AR add-ons are shipped with Pupil Cam2 and Cam3
        # cameras, we adjust the default eye window size to a 1:1 content aspect ratio.
        # The size of 500 was chosen s.t. the menu still fits.
        default_window_size = 500 + icon_bar_width, 500
        width, height = session_settings.get("window_size", default_window_size)

        main_window = glfw.create_window(width, height, title, None, None)

        window_position_manager = gl_utils.WindowPositionManager()
        window_pos = window_position_manager.new_window_position(
            window=main_window,
            default_position=window_position_default,
            previous_position=session_settings.get("window_position", None),
        )
        glfw.set_window_pos(main_window, window_pos[0], window_pos[1])

        glfw.make_context_current(main_window)
        cygl.utils.init()

        # gl_state settings
        basic_gl_setup()
        g_pool.image_tex = Named_Texture()
        g_pool.image_tex.update_from_ndarray(np.ones((1, 1), dtype=np.uint8) + 125)

        # setup GUI
        g_pool.gui = ui.UI()
        g_pool.menubar = ui.Scrolling_Menu(
            "Settings", pos=(-500, 0), size=(-icon_bar_width, 0), header_pos="left"
        )
        g_pool.iconbar = ui.Scrolling_Menu(
            "Icons", pos=(-icon_bar_width, 0), size=(0, 0), header_pos="hidden"
        )
        g_pool.gui.append(g_pool.menubar)
        g_pool.gui.append(g_pool.iconbar)

        general_settings = ui.Growing_Menu("General", header_pos="headline")

        def set_window_size():
            # Get current capture frame size
            f_width, f_height = g_pool.capture.frame_size
            # Eye camera resolutions are too small to be used as default window sizes.
            # We use double their size instead.
            frame_scale_factor = 2
            f_width *= frame_scale_factor
            f_height *= frame_scale_factor

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
        general_settings.append(ui.Switch("flip", g_pool, label="Flip image display"))
        general_settings.append(
            ui.Selector(
                "display_mode",
                g_pool,
                setter=set_display_mode_info,
                selection=["camera_image", "roi", "algorithm"],
                labels=["Camera Image", "ROI", "Algorithm"],
                label="Mode",
            )
        )
        g_pool.display_mode_info = ui.Info_Text(
            g_pool.display_mode_info_text[g_pool.display_mode]
        )

        general_settings.append(g_pool.display_mode_info)

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

        plugins_to_load = session_settings.get("loaded_plugins", default_plugins)
        if overwrite_cap_settings:
            # Ensure that overwrite_cap_settings takes preference over source plugins
            # with incorrect settings that were loaded from session settings.
            plugins_to_load.append(overwrite_cap_settings)

        # Add runtime plugins to the list of plugins to load with default arguments,
        # if not already restored from session settings
        plugins_to_load_names = {name for name, _ in plugins_to_load}
        for runtime_detector in runtime_detectors:
            runtime_name = runtime_detector.__name__
            if runtime_name not in plugins_to_load_names:
                plugins_to_load.append((runtime_name, {}))

        g_pool.plugins = Plugin_List(g_pool, plugins_to_load)

        if not g_pool.capture:
            # Make sure we always have a capture running. Important if there was no
            # capture stored in session settings.
            g_pool.plugins.add(
                g_pool.plugin_by_name[default_capture_name], default_capture_settings
            )

        toggle_general_settings(True)

        g_pool.writer = None
        g_pool.rec_path = None

        # Register callbacks main_window
        glfw.set_framebuffer_size_callback(main_window, on_resize)
        glfw.set_window_iconify_callback(main_window, on_iconify)
        glfw.set_key_callback(main_window, on_window_key)
        glfw.set_char_callback(main_window, on_window_char)
        glfw.set_mouse_button_callback(main_window, on_window_mouse_button)
        glfw.set_cursor_pos_callback(main_window, on_pos)
        glfw.set_scroll_callback(main_window, on_scroll)
        glfw.set_drop_callback(main_window, on_drop)

        # load last gui configuration
        g_pool.gui.configuration = session_settings.get("ui_config", {})
        # If previously selected plugin was not loaded this time, we will have an
        # expanded menubar without any menu selected. We need to ensure the menubar is
        # collapsed in this case.
        if all(submenu.collapsed for submenu in g_pool.menubar.elements):
            g_pool.menubar.collapsed = True

        # set up performance graphs
        pid = os.getpid()
        ps = psutil.Process(pid)
        ts = g_pool.get_timestamp()

        cpu_graph = graph.Bar_Graph()
        cpu_graph.pos = (20, 50)
        cpu_graph.update_fn = ps.cpu_percent
        cpu_graph.update_rate = 5
        cpu_graph.label = "CPU %0.1f"

        fps_graph = graph.Bar_Graph()
        fps_graph.pos = (140, 50)
        fps_graph.update_rate = 5
        fps_graph.label = "%0.0f FPS"
        g_pool.graphs = [cpu_graph, fps_graph]

        # set the last saved window size
        on_resize(main_window, *glfw.get_framebuffer_size(main_window))

        should_publish_frames = False
        frame_publish_format = "jpeg"
        frame_publish_format_recent_warning = False

        # create a timer to control window update frequency
        window_update_timer = timer(1 / 60)

        def window_should_update():
            return next(window_update_timer)

        logger.debug("Process started.")

        frame = None

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
        window_should_close = False
        while not window_should_close:

            if notify_sub.new_data:
                t, notification = notify_sub.recv()
                subject = notification["subject"]
                if subject.startswith("eye_process.should_stop"):
                    if notification["eye_id"] == eye_id:
                        break
                elif subject == "recording.started":
                    if notification["record_eye"] and g_pool.capture.online:
                        g_pool.rec_path = notification["rec_path"]
                        raw_mode = notification["compression"]
                        start_time_synced = notification["start_time_synced"]
                        logger.debug(f"Saving eye video to: {g_pool.rec_path}")
                        video_path = os.path.join(g_pool.rec_path, f"eye{eye_id}.mp4")
                        if raw_mode and frame and g_pool.capture.jpeg_support:
                            g_pool.writer = JPEG_Writer(video_path, start_time_synced)
                        elif hasattr(g_pool.capture._recent_frame, "h264_buffer"):
                            g_pool.writer = H264Writer(
                                video_path,
                                g_pool.capture.frame_size[0],
                                g_pool.capture.frame_size[1],
                                g_pool.capture.frame_rate,
                            )
                        else:
                            g_pool.writer = MPEG_Writer(video_path, start_time_synced)
                elif subject == "recording.stopped":
                    if g_pool.writer:
                        logger.debug("Done recording.")
                        try:
                            g_pool.writer.release()
                        except RuntimeError:
                            logger.info("No eye video recorded")
                        else:
                            # TODO: wrap recording logic into plugin
                            g_pool.capture.intrinsics.save(
                                g_pool.rec_path, custom_name=f"eye{eye_id}"
                            )
                        finally:
                            g_pool.writer = None
                elif subject.startswith("meta.should_doc"):
                    ipc_socket.notify(
                        {
                            "subject": "meta.doc",
                            "actor": f"eye{eye_id}",
                            "doc": eye.__doc__,
                        }
                    )
                elif subject.startswith("frame_publishing.started"):
                    should_publish_frames = True
                    frame_publish_format = notification.get("format", "jpeg")
                elif subject.startswith("frame_publishing.stopped"):
                    should_publish_frames = False
                    frame_publish_format = "jpeg"
                elif (
                    subject.startswith("start_eye_plugin")
                    and notification["target"] == g_pool.process
                ):
                    try:
                        g_pool.plugins.add(
                            g_pool.plugin_by_name[notification["name"]],
                            notification.get("args", {}),
                        )
                    except KeyError as err:
                        logger.error(f"Attempt to load unknown plugin: {err}")
                elif (
                    subject.startswith("stop_eye_plugin")
                    and notification["target"] == g_pool.process
                ):
                    try:
                        plugin_to_stop = g_pool.plugin_by_name[notification["name"]]
                    except KeyError as err:
                        logger.error(f"Attempt to load unknown plugin: {err}")
                    else:
                        plugin_to_stop.alive = False
                        g_pool.plugins.clean()

                for plugin in g_pool.plugins:
                    plugin.on_notify(notification)

            event = {}
            for plugin in g_pool.plugins:
                plugin.recent_events(event)

            frame = event.get("frame")
            if frame:
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
                            logger.warning(
                                '{}s are not compatible with format "{}"'.format(
                                    type(frame), frame_publish_format
                                )
                            )
                    else:
                        frame_publish_format_recent_warning = False
                        pupil_socket.send(
                            {
                                "topic": f"frame.eye.{eye_id}",
                                "width": frame.width,
                                "height": frame.height,
                                "index": frame.index,
                                "timestamp": frame.timestamp,
                                "format": frame_publish_format,
                                "__raw_data__": [data],
                            }
                        )

                t = frame.timestamp
                dt, ts = t - ts, t
                try:
                    fps_graph.add(1.0 / dt)
                except ZeroDivisionError:
                    pass

                if g_pool.writer:
                    try:
                        g_pool.writer.write_video_frame(frame)
                    except NonMonotonicTimestampError as e:
                        logger.error(
                            "Recorder received non-monotonic timestamp!"
                            " Stopping the recording!"
                        )
                        logger.debug(str(e))
                        ipc_socket.notify({"subject": "recording.should_stop"})
                        ipc_socket.notify(
                            {"subject": "recording.should_stop", "remote_notify": "all"}
                        )

                for result in event.get(EVENT_KEY, ()):
                    pupil_socket.send(result)

            # GL drawing
            if window_should_update():
                cpu_graph.update()
                if is_window_visible(main_window):
                    consume_events_and_render_buffer()
                glfw.poll_events()
                window_should_close = glfw.window_should_close(main_window)

        # END while running

        # in case eye recording was still runnnig: Save&close
        if g_pool.writer:
            logger.debug("Done recording eye.")
            g_pool.writer.release()
            g_pool.writer = None

        session_settings["loaded_plugins"] = g_pool.plugins.get_initializers()
        # save session persistent settings
        session_settings["flip"] = g_pool.flip
        session_settings["display_mode"] = g_pool.display_mode
        session_settings["ui_config"] = g_pool.gui.configuration
        session_settings["version"] = str(g_pool.version)

        if not hide_ui:
            glfw.restore_window(main_window)  # need to do this for windows os
            session_settings["window_position"] = glfw.get_window_pos(main_window)
            session_window_size = glfw.get_window_size(main_window)
            if 0 not in session_window_size:
                f_width, f_height = session_window_size
                if platform.system() in ("Windows", "Linux"):
                    # Store unscaled window size as the operating system will scale the
                    # windows appropriately during launch on Windows and Linux.
                    f_width, f_height = (
                        f_width / content_scale,
                        f_height / content_scale,
                    )
                session_settings["window_size"] = int(f_width), int(f_height)

        session_settings.close()

    logger.debug("Process shutting down.")
    for plugin in g_pool.plugins:
        plugin.alive = False
    g_pool.plugins.clean()

    glfw.destroy_window(main_window)
    g_pool.gui.terminate()
    glfw.terminate()
    logger.debug("Process shut down.")


def eye_profiled(
    timebase,
    is_alive_flag,
    ipc_pub_url,
    ipc_sub_url,
    ipc_push_url,
    user_dir,
    version,
    eye_id,
    overwrite_cap_settings=None,
    hide_ui=False,
    debug=False,
    pub_socket_hwm=None,
    parent_application="capture",
    skip_driver_installation=False,
):
    import cProfile
    import os
    import subprocess
    from textwrap import dedent

    from .eye import eye

    cProfile.runctx(
        dedent(
            """
            eye(
                timebase,
                is_alive_flag,
                ipc_pub_url,
                ipc_sub_url,
                ipc_push_url,
                user_dir,
                version,
                eye_id,
                overwrite_cap_settings,
                hide_ui,
                debug,
                pub_socket_hwm,
                parent_application,
                skip_driver_installation
            )
            """
        ),
        {
            "timebase": timebase,
            "is_alive_flag": is_alive_flag,
            "ipc_pub_url": ipc_pub_url,
            "ipc_sub_url": ipc_sub_url,
            "ipc_push_url": ipc_push_url,
            "user_dir": user_dir,
            "version": version,
            "eye_id": eye_id,
            "overwrite_cap_settings": overwrite_cap_settings,
            "hide_ui": hide_ui,
            "debug": debug,
            "pub_socket_hwm": pub_socket_hwm,
            "parent_application": parent_application,
            "skip_driver_installation": skip_driver_installation,
        },
        locals(),
        f"eye{eye_id}.pstats",
    )
    loc = os.path.abspath(__file__).rsplit("pupil_src", 1)
    gprof2dot_loc = os.path.join(loc[0], "pupil_src", "shared_modules", "gprof2dot.py")
    subprocess.call(
        "python "
        + gprof2dot_loc
        + " -f pstats eye{0}.pstats | dot -Tpng -o eye{0}_cpu_time.png".format(eye_id),
        shell=True,
    )
    print(
        "created cpu time graph for eye{} process. Please check out the png next to the eye.py file".format(
            eye_id
        )
    )
