"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import platform
import socket

import gl_utils
import glfw
import numpy as np
from gl_utils import GLFWErrorReporting
from OpenGL.GL import GL_COLOR_BUFFER_BIT

GLFWErrorReporting.set_default()

from plugin import System_Plugin_Base
from pyglui import cygl, ui

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

window_size_default = (400, 400)
logger = logging.getLogger(__name__)


class Service_UI(System_Plugin_Base):
    def __init__(
        self,
        g_pool,
        window_size=window_size_default,
        window_position=None,
        gui_scale=1.0,
        ui_config={},
    ):
        super().__init__(g_pool)

        self.texture = np.zeros((1, 1, 3), dtype=np.uint8) + 128

        glfw.init()
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
        if g_pool.hide_ui:
            glfw.window_hint(glfw.VISIBLE, 0)  # hide window

        main_window = glfw.create_window(*window_size, "Pupil Service", None, None)

        window_position_manager = gl_utils.WindowPositionManager()
        window_position = window_position_manager.new_window_position(
            window=main_window,
            default_position=window_position_default,
            previous_position=window_position,
        )
        glfw.set_window_pos(main_window, *window_position)

        glfw.make_context_current(main_window)
        cygl.utils.init()
        g_pool.main_window = main_window

        g_pool.gui = ui.UI()
        g_pool.menubar = ui.Scrolling_Menu(
            "Settings", pos=(0, 0), size=(0, 0), header_pos="headline"
        )
        g_pool.gui.append(g_pool.menubar)

        # Callback functions
        def on_resize(window, w, h):
            # Always clear buffers on resize to make sure that there are no overlapping
            # artifacts from previous frames.
            gl_utils.glClear(GL_COLOR_BUFFER_BIT)
            gl_utils.glClearColor(0, 0, 0, 1)

            self.window_size = w, h
            self.content_scale = gl_utils.get_content_scale(window)
            g_pool.gui.scale = self.content_scale
            g_pool.gui.update_window(w, h)
            g_pool.gui.collect_menus()

            # Needed, to update the window buffer while resizing
            self.update_ui()

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

        def on_scroll(window, x, y):
            g_pool.gui.update_scroll(x, y * scroll_factor)

        def set_window_size():
            # Get default window size
            f_width, f_height = window_size_default

            # Get current display scale factor
            content_scale = gl_utils.get_content_scale(main_window)
            framebuffer_scale = gl_utils.get_framebuffer_scale(main_window)
            display_scale_factor = content_scale / framebuffer_scale

            # Scale the capture frame size by display scale factor
            f_width *= display_scale_factor
            f_height *= display_scale_factor

            # Set the newly calculated size (scaled capture frame size + scaled icon bar width)
            glfw.set_window_size(main_window, int(f_width), int(f_height))

        def reset_restart():
            logger.warning("Resetting all settings and restarting Capture.")
            glfw.set_window_should_close(main_window, True)
            self.notify_all({"subject": "clear_settings_process.should_start"})
            self.notify_all({"subject": "service_process.should_start", "delay": 2.0})

        g_pool.menubar.append(ui.Button("Reset window size", set_window_size))

        for *_, (ip, port, *_) in socket.getaddrinfo(
            "localhost",
            g_pool.preferred_remote_port,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        ):
            pupil_remote_addr = f"{ip}:{port}"
            break
        else:
            pupil_remote_addr = "unknown"

        g_pool.menubar.append(
            ui.Text_Input(
                "pupil_remote_addr",
                getter=lambda: pupil_remote_addr,
                setter=lambda x: None,
                label="Pupil Remote address",
            )
        )
        g_pool.menubar.append(
            ui.Switch(
                "eye0_process",
                label="Detect eye 0",
                setter=lambda alive: self.start_stop_eye(0, alive),
                getter=lambda: g_pool.eye_procs_alive[0].value,
            )
        )
        g_pool.menubar.append(
            ui.Switch(
                "eye1_process",
                label="Detect eye 1",
                setter=lambda alive: self.start_stop_eye(1, alive),
                getter=lambda: g_pool.eye_procs_alive[1].value,
            )
        )

        g_pool.menubar.append(ui.Info_Text(f"Service Version: {g_pool.version}"))

        g_pool.menubar.append(ui.Button("Restart with default settings", reset_restart))

        # Register callbacks main_window
        glfw.set_framebuffer_size_callback(main_window, on_resize)
        glfw.set_key_callback(main_window, on_window_key)
        glfw.set_char_callback(main_window, on_window_char)
        glfw.set_mouse_button_callback(main_window, on_window_mouse_button)
        glfw.set_cursor_pos_callback(main_window, on_pos)
        glfw.set_scroll_callback(main_window, on_scroll)
        g_pool.gui.configuration = ui_config
        gl_utils.basic_gl_setup()

        on_resize(g_pool.main_window, *glfw.get_framebuffer_size(main_window))

    def on_notify(self, notification):
        if notification["subject"] == "service_process.ui.should_update":
            # resend delayed notification, keep ui loop running:
            notification["delay"] = notification["initial_delay"]
            self.notify_all(notification)
            self.update_ui()

    def update_ui(self):
        if not glfw.window_should_close(self.g_pool.main_window):
            gl_utils.glViewport(0, 0, *self.window_size)
            glfw.poll_events()
            self.gl_display()
            try:
                clipboard = glfw.get_clipboard_string(None).decode()
            except (AttributeError, glfw.GLFWError):
                # clipbaord is None, might happen on startup
                clipboard = ""
            self.g_pool.gui.update_clipboard(clipboard)
            user_input = self.g_pool.gui.update()
            if user_input.clipboard and user_input.clipboard != clipboard:
                # only write to clipboard if content changed
                glfw.set_clipboard_string(None, user_input.clipboard)

            glfw.swap_buffers(self.g_pool.main_window)
        else:
            self.notify_all({"subject": "service_process.should_stop"})

    def gl_display(self):
        gl_utils.make_coord_system_norm_based()
        cygl.utils.draw_gl_texture(self.texture)
        gl_utils.make_coord_system_pixel_based(
            (self.window_size[1], self.window_size[0], 3)
        )

    def cleanup(self):
        if not self.g_pool.hide_ui:
            glfw.restore_window(self.g_pool.main_window)

        del self.g_pool.menubar[:]
        self.g_pool.gui.remove(self.g_pool.menubar)

        self.g_pool.gui.terminate()
        glfw.destroy_window(self.g_pool.main_window)
        glfw.terminate()

        del self.g_pool.gui
        del self.g_pool.main_window
        del self.texture

    def get_init_dict(self):
        sess = {
            "window_position": glfw.get_window_pos(self.g_pool.main_window),
            "ui_config": self.g_pool.gui.configuration,
        }

        session_window_size = glfw.get_window_size(self.g_pool.main_window)
        if 0 not in session_window_size:
            f_width, f_height = session_window_size
            if platform.system() in ("Windows", "Linux"):
                content_scale = gl_utils.get_content_scale(self.g_pool.main_window)
                f_width, f_height = (
                    f_width / content_scale,
                    f_height / content_scale,
                )
            sess["window_size"] = int(f_width), int(f_height)

        return sess

    def start_stop_eye(self, eye_id, make_alive):
        if make_alive:
            n = {
                "subject": f"eye_process.should_start.{eye_id}",
                "eye_id": eye_id,
            }
        else:
            n = {
                "subject": f"eye_process.should_stop.{eye_id}",
                "eye_id": eye_id,
                "delay": 0.2,
            }
        self.notify_all(n)
