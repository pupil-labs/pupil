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
import glfw
from pyglui import ui, cygl
from plugin import System_Plugin

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


class Service_UI(System_Plugin):
    def __init__(self, g_pool, window_size=(400, 300), window_position=window_position_default, gui_scale=1., ui_config={}):
        super().__init__(g_pool)

        glfw.glfwInit()
        main_window = glfw.glfwCreateWindow(*window_size, "Pupil Service")
        glfw.glfwSetWindowPos(main_window, *window_position)
        glfw.glfwMakeContextCurrent(main_window)
        cygl.utils.init()
        g_pool.main_window = main_window

        g_pool.gui = ui.UI()
        g_pool.gui_user_scale = gui_scale
        g_pool.menubar = ui.Scrolling_Menu("Settings", header_pos='headline')
        g_pool.gui.append(g_pool.menubar)

        g_pool.menubar.append(ui.Selector('detection_mapping_mode',
                                          g_pool,
                                          label='Detection & mapping mode',
                                          setter=self.set_detection_mapping_mode,
                                          selection=['2d', '3d']))
        g_pool.menubar.append(ui.Switch('eye0_process',
                                        label='Detect eye 0',
                                        setter=lambda alive: self.start_stop_eye(0, alive),
                                        getter=lambda: g_pool.eyes_are_alive[0].value))
        g_pool.menubar.append(ui.Switch('eye1_process',
                                        label='Detect eye 1',
                                        setter=lambda alive: self.start_stop_eye(1, alive),
                                        getter=lambda: g_pool.eyes_are_alive[1].value))

        g_pool.menubar.append(ui.Info_Text('Service Version: {}'.format(g_pool.version)))

        # Callback functions
        def on_resize(window, w, h):
            self.hdpi_factor = float(glfw.glfwGetFramebufferSize(window)[0] / glfw.glfwGetWindowSize(window)[0])
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

        # Register callbacks main_window
        glfw.glfwSetFramebufferSizeCallback(main_window, on_resize)
        glfw.glfwSetKeyCallback(main_window, on_window_key)
        glfw.glfwSetCharCallback(main_window, on_window_char)
        glfw.glfwSetMouseButtonCallback(main_window, on_window_mouse_button)
        glfw.glfwSetCursorPosCallback(main_window, on_pos)
        glfw.glfwSetScrollCallback(main_window, on_scroll)
        g_pool.gui.configuration = ui_config

    def cleanup(self):
        del self.g_pool.menubar[:]
        self.g_pool.gui.remove(self.g_pool.menubar)

    def start_stop_eye(self, eye_id, make_alive):
        if make_alive:
            n = {'subject': 'eye_process.should_start.{}'.format(eye_id), 'eye_id': eye_id}
        else:
            n = {'subject': 'eye_process.should_stop.{}'.format(eye_id), 'eye_id': eye_id, 'delay': 0.2}
        self.notify_all(n)

    def set_detection_mapping_mode(self, new_mode):
        self.notify_all({'subject': 'set_detection_mapping_mode', 'mode': new_mode})
