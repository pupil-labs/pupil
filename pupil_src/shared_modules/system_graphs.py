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
import psutil
import glfw
from pyglui import ui, graph
from pyglui.cygl.utils import RGBA, mix_smooth
from plugin import System_Plugin_Base


class System_Graphs(System_Plugin_Base):
    icon_chr = chr(0xe01d)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool, show_cpu=True, show_fps=True, show_conf0=True,
                 show_conf1=True, show_dia0=False, show_dia1=False,
                 dia_min=0., dia_max=8.):
        super().__init__(g_pool)
        self.show_cpu = show_cpu
        self.show_fps = show_fps
        self.show_conf0 = show_conf0
        self.show_conf1 = show_conf1
        self.show_dia0 = show_dia0
        self.show_dia1 = show_dia1
        self.dia_min = dia_min
        self.dia_max = dia_max
        self.conf_grad_limits = .0, 1.
        self.ts = None
        self.idx = None

    def init_ui(self):
        self.add_menu()
        self.menu_icon.order = 0.01
        self.menu.label = 'System Graphs'
        self.menu.append(ui.Switch('show_cpu', self, label='Display CPU usage'))
        self.menu.append(ui.Switch('show_fps', self, label='Display frames per second'))
        self.menu.append(ui.Switch('show_conf0', self, label='Display confidence for eye 0'))
        self.menu.append(ui.Switch('show_conf1', self, label='Display confidence for eye 1'))
        self.menu.append(ui.Switch('show_dia0', self, label='Display pupil diameter for eye 0'))
        self.menu.append(ui.Switch('show_dia1', self, label='Display pupil diameter for eye 1'))

        # set up performace graphs:
        pid = os.getpid()
        ps = psutil.Process(pid)

        self.cpu_graph = graph.Bar_Graph()
        self.cpu_graph.pos = (20, 50)
        self.cpu_graph.update_fn = ps.cpu_percent
        self.cpu_graph.update_rate = 5
        self.cpu_graph.label = 'CPU %0.1f'

        self.fps_graph = graph.Bar_Graph()
        self.fps_graph.pos = (140, 50)
        self.fps_graph.update_rate = 5
        self.fps_graph.label = "%0.0f FPS"

        self.conf0_graph = graph.Bar_Graph(max_val=1.0)
        self.conf0_graph.pos = (260, 50)
        self.conf0_graph.update_rate = 5
        self.conf0_graph.label = "id0 conf: %0.2f"
        self.conf1_graph = graph.Bar_Graph(max_val=1.0)
        self.conf1_graph.pos = (380, 50)
        self.conf1_graph.update_rate = 5
        self.conf1_graph.label = "id1 conf: %0.2f"

        self.dia0_graph = graph.Bar_Graph(min_val=self.dia_min, max_val=self.dia_max)
        self.dia0_graph.pos = (260, 100)
        self.dia0_graph.update_rate = 5
        self.dia0_graph.label = "id0 dia: %0.2f"

        self.dia1_graph = graph.Bar_Graph(min_val=self.dia_min, max_val=self.dia_max)
        self.dia1_graph.pos = (380, 100)
        self.dia1_graph.update_rate = 5
        self.dia1_graph.label = "id1 dia: %0.2f"

        self.conf_grad = RGBA(1., .0, .0, self.conf0_graph.color[3]), self.conf0_graph.color

        def set_dia_min(val):
            self.dia0_graph.min_val = val
            self.dia1_graph.min_val = val

        def set_dia_max(val):
            self.dia0_graph.max_val = val
            self.dia1_graph.max_val = val

        self.menu.append(ui.Slider('min_val', self.dia0_graph, label='Minimum pupil diameter',
                                   setter=set_dia_min, min=0., max=15., step=0.1))
        self.menu.append(ui.Slider('max_val', self.dia0_graph, label='Maximum pupil diameter',
                                   setter=set_dia_max, min=1., max=15., step=0.1))

        self.on_window_resize(self.g_pool.main_window)

    def on_window_resize(self, window, *args):
        fb_size = glfw.glfwGetFramebufferSize(window)
        hdpi_factor = glfw.getHDPIFactor(window)

        self.cpu_graph.scale = hdpi_factor
        self.fps_graph.scale = hdpi_factor
        self.conf0_graph.scale = hdpi_factor
        self.conf1_graph.scale = hdpi_factor
        self.dia0_graph.scale = hdpi_factor
        self.dia1_graph.scale = hdpi_factor

        self.cpu_graph.adjust_window_size(*fb_size)
        self.fps_graph.adjust_window_size(*fb_size)
        self.conf0_graph.adjust_window_size(*fb_size)
        self.conf1_graph.adjust_window_size(*fb_size)
        self.dia0_graph.adjust_window_size(*fb_size)
        self.dia1_graph.adjust_window_size(*fb_size)

    def gl_display(self):
        if self.show_cpu:
            self.cpu_graph.draw()
        if self.show_fps:
            self.fps_graph.draw()
        if self.show_conf0:
            self.conf0_graph.color = mix_smooth(self.conf_grad[0], self.conf_grad[1],
                                                self.conf0_graph.avg, self.conf_grad_limits[0],
                                                self.conf_grad_limits[1])
            self.conf0_graph.draw()
        if self.show_conf1:
            self.conf1_graph.color = mix_smooth(self.conf_grad[0], self.conf_grad[1],
                                                self.conf1_graph.avg, self.conf_grad_limits[0],
                                                self.conf_grad_limits[1])
            self.conf1_graph.draw()
        if self.show_dia0:
            self.dia0_graph.draw()
        if self.show_dia1:
            self.dia1_graph.draw()

    def recent_events(self, events):
        # update cpu graph
        self.cpu_graph.update()

        # update pupil graphs
        if 'frame' not in events or self.idx != events["frame"].index:
            for p in events["pupil_positions"]:
                # update confidence graph
                cg = self.conf0_graph if p['id'] == 0 else self.conf1_graph
                cg.add(p['confidence'])
                # update diameter graph
                dg = self.dia0_graph if p['id'] == 0 else self.dia1_graph
                dg.add(p.get('diameter_3d', 0.))

        # update wprld fps graph
        if 'frame' in events:
            t = events["frame"].timestamp
            if self.ts and t != self.ts:
                dt, self.ts = t-self.ts, t
                try:
                    self.fps_graph.add(1./dt)
                except ZeroDivisionError:
                    pass
            else:
                self.ts = t
            self.idx = events["frame"].index  # required for eye graph logic in player

    def deinit_ui(self):
        self.remove_menu()
        self.cpu_graph = None
        self.fps_graph = None
        self.conf0_graph = None
        self.conf1_graph = None
        self.dia0_graph = None
        self.dia1_graph = None

    def get_init_dict(self):
        return {'show_cpu': self.show_cpu, 'show_fps': self.show_fps,
                'show_conf0': self.show_conf0, 'show_conf1': self.show_conf1,
                'show_dia0': self.show_dia0, 'show_dia1': self.show_dia1}
