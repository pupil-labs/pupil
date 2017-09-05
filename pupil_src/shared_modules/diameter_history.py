'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import logging
import glfw
from collections import deque
from plugin import Plugin
from pyglui import ui, graph

import gl_utils

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Diameter_History(Plugin):
    """Pupil dilation visualization

    This plugin uses the 3d model's pupil diameter
    and displays it in a graph for each eye.
    """
    order = .9

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.graphs = ()
        self.menu = None

    def init_gui(self):
        eye0_graph = graph.Bar_Graph(min_val=.0, max_val=5.)
        eye0_graph.pos = (260, 230)
        eye0_graph.update_rate = 5
        eye0_graph.label = "id0 dia: %0.2f"

        eye1_graph = graph.Bar_Graph(min_val=.0, max_val=5.)
        eye1_graph.pos = (380, 230)
        eye1_graph.update_rate = 5
        eye1_graph.label = "id0 dia: %0.2f"

        self.graphs = eye0_graph, eye1_graph
        self.on_window_resize(self.g_pool.main_window, *glfw.glfwGetFramebufferSize(self.g_pool.main_window))

        def close():
            self.alive = False

        self.menu = ui.Growing_Menu('Pupil Diameter History')
        self.menu.collapsed = True
        self.menu.append(ui.Button('Close', close))
        self.menu.append(ui.Info_Text('Displays the recent pupil diameter in millimeters for each eye.'))
        self.g_pool.sidebar.append(self.menu)

    def recent_events(self, events):
        for p in events['pupil_positions']:
            diam = p.get('diameter_3d', 0.)
            if diam > 0. and p['confidence'] > 0.6:
                self.graphs[p['id']].add(diam)

    def gl_display(self):
        for g in self.graphs:
            g.draw()

    def on_window_resize(self, window, w, h):
        if gl_utils.is_window_visible(window):
            hdpi_factor = float(glfw.glfwGetFramebufferSize(window)[0] / glfw.glfwGetWindowSize(window)[0])
            for g in self.graphs:
                g.scale = hdpi_factor
                g.adjust_window_size(w, h)

    def get_init_dict(self):
        return {}

    def deinit_ui(self):
        self.graphs = ()
        self.g_pool.sidebar.remove(self.menu)
        self.menu = None

    def cleanup(self):
        if self.menu:
            self.deinit_ui()
