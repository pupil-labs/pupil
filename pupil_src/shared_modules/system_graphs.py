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

import gl_utils
import glfw
import psutil
from gl_utils import GLFWErrorReporting

GLFWErrorReporting.set_default()

from plugin import System_Plugin_Base
from pyglui import graph, ui
from pyglui.cygl.utils import RGBA, mix_smooth


class System_Graphs(System_Plugin_Base):
    icon_chr = chr(0xE01D)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        show_cpu=True,
        show_fps=True,
        show_conf0=True,
        show_conf1=True,
        **kwargs,
    ):
        super().__init__(g_pool)
        self.show_cpu = show_cpu
        self.show_fps = show_fps
        self.show_conf0 = show_conf0
        self.show_conf1 = show_conf1
        self.conf_grad_limits = 0.0, 1.0
        self.ts = None
        self.idx = None

    def init_ui(self):
        # set up performace graphs:
        pid = os.getpid()
        ps = psutil.Process(pid)

        self.cpu_graph = graph.Bar_Graph()
        self.cpu_graph.pos = (20, 50)
        self.cpu_graph.update_fn = ps.cpu_percent
        self.cpu_graph.update_rate = 5
        self.cpu_graph.label = "CPU %0.1f"

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

        self.conf_grad = (
            RGBA(1.0, 0.0, 0.0, self.conf0_graph.color[3]),
            self.conf0_graph.color,
        )

        self.on_window_resize(self.g_pool.main_window)

    def on_window_resize(self, window, *args):
        fb_size = glfw.get_framebuffer_size(window)
        content_scale = gl_utils.get_content_scale(window)

        self.cpu_graph.scale = content_scale
        self.fps_graph.scale = content_scale
        self.conf0_graph.scale = content_scale
        self.conf1_graph.scale = content_scale

        self.cpu_graph.adjust_window_size(*fb_size)
        self.fps_graph.adjust_window_size(*fb_size)
        self.conf0_graph.adjust_window_size(*fb_size)
        self.conf1_graph.adjust_window_size(*fb_size)

    def gl_display(self):
        if self.show_cpu:
            self.cpu_graph.draw()
        if self.show_fps:
            self.fps_graph.draw()
        if self.show_conf0:
            self.conf0_graph.color = mix_smooth(
                self.conf_grad[0],
                self.conf_grad[1],
                self.conf0_graph.avg,
                self.conf_grad_limits[0],
                self.conf_grad_limits[1],
            )
            self.conf0_graph.draw()
        if self.show_conf1:
            self.conf1_graph.color = mix_smooth(
                self.conf_grad[0],
                self.conf_grad[1],
                self.conf1_graph.avg,
                self.conf_grad_limits[0],
                self.conf_grad_limits[1],
            )
            self.conf1_graph.draw()

    def recent_events(self, events):
        # update cpu graph
        self.cpu_graph.update()

        # update pupil graphs
        if "frame" not in events or self.idx != events["frame"].index:
            for p in events["pupil"]:
                if p["topic"] == "pupil.0.2d":
                    assert p["id"] == 0  # sanity check
                    self.conf0_graph.add(p["confidence"])
                elif p["topic"] == "pupil.1.2d":
                    assert p["id"] == 1  # sanity check
                    self.conf1_graph.add(p["confidence"])
                # pre-2.0 recordings:
                elif p["topic"] == "pupil.0":
                    assert p["id"] == 0  # sanity check
                    self.conf0_graph.add(p["confidence"])
                elif p["topic"] == "pupil.1":
                    assert p["id"] == 1  # sanity check
                    self.conf1_graph.add(p["confidence"])

        # update wprld fps graph
        if "frame" in events:
            t = events["frame"].timestamp
            if self.ts and t != self.ts:
                dt, self.ts = t - self.ts, t
                try:
                    self.fps_graph.add(1.0 / dt)
                except ZeroDivisionError:
                    pass
            else:
                self.ts = t
            self.idx = events["frame"].index  # required for eye graph logic in player

    def deinit_ui(self):
        self.cpu_graph = None
        self.fps_graph = None
        self.conf0_graph = None
        self.conf1_graph = None

    def get_init_dict(self):
        return {
            "show_cpu": self.show_cpu,
            "show_fps": self.show_fps,
            "show_conf0": self.show_conf0,
            "show_conf1": self.show_conf1,
        }
