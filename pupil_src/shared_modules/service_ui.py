'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import platform
import logging
import socket

import numpy as np

import glfw
import gl_utils
from pyglui import ui, cygl
from plugin import System_Plugin_Base

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

window_size_default = (400, 400)
logger = logging.getLogger(__name__)


class Service_UI(System_Plugin_Base):
    def __init__(self, g_pool, window_size=window_size_default,
                 window_position=window_position_default,
                 gui_scale=1., ui_config={}):
        super().__init__(g_pool)

        self.texture = np.zeros((1, 1, 3), dtype=np.uint8) + 128

        glfw.glfwInit()
        main_window = glfw.glfwCreateWindow(*window_size, "Pupil Service")
        glfw.glfwSetWindowPos(main_window, *window_position)
        glfw.glfwMakeContextCurrent(main_window)
        cygl.utils.init()
        g_pool.main_window = main_window

        g_pool.gui = ui.UI()
        g_pool.gui_user_scale = gui_scale
        g_pool.menubar = ui.Scrolling_Menu("Settings", pos=(0, 0), size=(0, 0),
                                           header_pos='headline')
        g_pool.gui.append(g_pool.menubar)

        # Callback functions
        def on_resize(window, w, h):
            self.window_size = w, h
            self.hdpi_factor = glfw.getHDPIFactor(window)
            g_pool.gui.scale = g_pool.gui_user_scale * self.hdpi_factor
            g_pool.gui.update_window(w, h)
            g_pool.gui.collect_menus()

        def on_window_key(window, key, scancode, action, mods):
            g_pool.gui.update_key(key, scancode, action, mods)

        def on_window_char(window, char):
            g_pool.gui.update_char(char)

        def on_window_mouse_button(window, button, action, mods):
            g_pool.gui.update_button(button, action, mods)

        def on_pos(window, x, y):
            x, y = x * self.hdpi_factor, y * self.hdpi_factor
            g_pool.gui.update_mouse(x, y)

        def on_scroll(window, x, y):
            g_pool.gui.update_scroll(x, y * scroll_factor)

        def set_scale(new_scale):
            g_pool.gui_user_scale = new_scale
            on_resize(main_window, *self.window_size)

        def set_window_size():
            glfw.glfwSetWindowSize(main_window, *window_size_default)

        def reset_restart():
            logger.warning("Resetting all settings and restarting Capture.")
            glfw.glfwSetWindowShouldClose(main_window, True)
            self.notify_all({'subject': 'clear_settings_process.should_start'})
            self.notify_all({'subject': 'service_process.should_start', 'delay': 2.})

        g_pool.menubar.append(ui.Selector('gui_user_scale', g_pool,
                                          setter=set_scale,
                                          selection=[.6, .8, 1., 1.2, 1.4],
                                          label='Interface size'))

        g_pool.menubar.append(ui.Button('Reset window size', set_window_size))

        pupil_remote_addr = '{}:50020'.format(socket.gethostbyname(socket.gethostname()))
        g_pool.menubar.append(ui.Text_Input('pupil_remote_addr',
                                            getter=lambda: pupil_remote_addr,
                                            setter=lambda x: None,
                                            label='Pupil Remote address'))

        g_pool.menubar.append(ui.Selector('detection_mapping_mode',
                                          g_pool,
                                          label='Detection & mapping mode',
                                          setter=self.set_detection_mapping_mode,
                                          selection=['disabled', '2d', '3d']))
        g_pool.menubar.append(ui.Switch('eye0_process',
                                        label='Detect eye 0',
                                        setter=lambda alive: self.start_stop_eye(0, alive),
                                        getter=lambda: g_pool.eyes_are_alive[0].value))
        g_pool.menubar.append(ui.Switch('eye1_process',
                                        label='Detect eye 1',
                                        setter=lambda alive: self.start_stop_eye(1, alive),
                                        getter=lambda: g_pool.eyes_are_alive[1].value))

        g_pool.menubar.append(ui.Info_Text('Service Version: {}'.format(g_pool.version)))

        g_pool.menubar.append(ui.Button('Restart with default settings', reset_restart))

        # Register callbacks main_window
        glfw.glfwSetFramebufferSizeCallback(main_window, on_resize)
        glfw.glfwSetKeyCallback(main_window, on_window_key)
        glfw.glfwSetCharCallback(main_window, on_window_char)
        glfw.glfwSetMouseButtonCallback(main_window, on_window_mouse_button)
        glfw.glfwSetCursorPosCallback(main_window, on_pos)
        glfw.glfwSetScrollCallback(main_window, on_scroll)
        g_pool.gui.configuration = ui_config
        gl_utils.basic_gl_setup()

        on_resize(g_pool.main_window, *glfw.glfwGetFramebufferSize(main_window))

    def on_notify(self, notification):
        if notification['subject'] == 'service_process.ui.should_update':
            # resend delayed notification, keep ui loop running:
            notification['delay'] = notification['initial_delay']
            self.notify_all(notification)
            self.update_ui()

    def update_ui(self):
        if not glfw.glfwWindowShouldClose(self.g_pool.main_window):
            gl_utils.glViewport(0, 0, *self.window_size)
            self.gl_display()
            try:
                clipboard = glfw.glfwGetClipboardString(self.g_pool.main_window).decode()
            except AttributeError:  # clipbaord is None, might happen on startup
                clipboard = ''
            self.g_pool.gui.update_clipboard(clipboard)
            user_input = self.g_pool.gui.update()
            if user_input.clipboard and user_input.clipboard != clipboard:
                # only write to clipboard if content changed
                glfw.glfwSetClipboardString(self.g_pool.main_window, user_input.clipboard.encode())

            glfw.glfwSwapBuffers(self.g_pool.main_window)
            glfw.glfwPollEvents()
        else:
            self.notify_all({'subject': 'service_process.should_stop'})

    def gl_display(self):
        gl_utils.make_coord_system_norm_based()
        cygl.utils.draw_gl_texture(self.texture)
        gl_utils.make_coord_system_pixel_based((self.window_size[1], self.window_size[0], 3))

    def cleanup(self):
        glfw.glfwRestoreWindow(self.g_pool.main_window)

        del self.g_pool.menubar[:]
        self.g_pool.gui.remove(self.g_pool.menubar)

        self.g_pool.gui.terminate()
        glfw.glfwDestroyWindow(self.g_pool.main_window)
        glfw.glfwTerminate()

        del self.g_pool.gui
        del self.g_pool.main_window
        del self.texture

    def get_init_dict(self):
        sess = {'window_position': glfw.glfwGetWindowPos(self.g_pool.main_window),
                'gui_scale': self.g_pool.gui_user_scale,
                'ui_config': self.g_pool.gui.configuration}

        session_window_size = glfw.glfwGetWindowSize(self.g_pool.main_window)
        if 0 not in session_window_size:
            sess['window_size'] = session_window_size

        return sess

    def start_stop_eye(self, eye_id, make_alive):
        if make_alive:
            n = {'subject': 'eye_process.should_start.{}'.format(eye_id), 'eye_id': eye_id}
        else:
            n = {'subject': 'eye_process.should_stop.{}'.format(eye_id), 'eye_id': eye_id, 'delay': 0.2}
        self.notify_all(n)

    def set_detection_mapping_mode(self, new_mode):
        self.notify_all({'subject': 'set_detection_mapping_mode', 'mode': new_mode})
