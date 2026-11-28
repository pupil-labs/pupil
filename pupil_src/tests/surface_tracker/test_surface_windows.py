"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _install_missing_dep_stubs(monkeypatch):
    def _stub_module(name, attrs=None):
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    np_stub = _stub_module(
        "numpy",
        {
            "__version__": "1.26.0",
            "float32": float,
            "int64": int,
            "ndarray": list,
            "array": lambda obj, dtype=None: obj,
            "asarray": lambda obj, dtype=None: obj,
        },
    )
    _stub_module("numpy.typing", {"NDArray": object})
    np_stub.typing = sys.modules["numpy.typing"]

    _stub_module("scipy")
    _stub_module("scipy.interpolate", {"interp1d": MagicMock()})
    _stub_module("scipy.spatial")
    _stub_module("scipy.spatial.distance", {"pdist": MagicMock()})
    _stub_module(
        "cv2",
        {
            "getPerspectiveTransform": MagicMock(),
            "TERM_CRITERIA_EPS": 2,
            "TERM_CRITERIA_COUNT": 1,
        },
    )
    _stub_module("rich")
    _stub_module("rich.progress", {"track": lambda it, **kw: it})
    _stub_module("pupil_apriltags", {"Detection": object, "Detector": MagicMock})
    _stub_module(
        "msgpack",
        {
            "version": (1, 0, 0),
            "pack": MagicMock(),
            "unpack": MagicMock(return_value={}),
        },
    )
    pyglui_mod = _stub_module("pyglui")
    pyglui_mod.ui = _stub_module("pyglui.ui", {"get_opensans_font_path": lambda: ""})
    pyglui_cygl = _stub_module("pyglui.cygl")
    pyglui_cygl.utils = _stub_module(
        "pyglui.cygl.utils",
        {"Named_Texture": MagicMock, "RGBA": MagicMock(), "draw_polyline": MagicMock()},
    )
    pyglui_mod.cygl = pyglui_cygl
    pyglui_fontstash = _stub_module("pyglui.pyfontstash")
    pyglui_fontstash.fontstash = _stub_module(
        "pyglui.pyfontstash.fontstash", {"Context": MagicMock}
    )
    pyglui_mod.pyfontstash = pyglui_fontstash


@pytest.fixture
def surface_tracker_modules(monkeypatch):
    saved_modules = dict(sys.modules)
    try:
        _install_missing_dep_stubs(monkeypatch)
        for mod_name in list(sys.modules):
            if mod_name == "surface_tracker" or mod_name.startswith("surface_tracker."):
                sys.modules.pop(mod_name, None)
            if mod_name == "gl_utils" or mod_name.startswith("gl_utils."):
                sys.modules.pop(mod_name, None)

        gui_mod = importlib.import_module("surface_tracker.gui")
        tracker_mod = importlib.import_module("surface_tracker.surface_tracker")
        yield gui_mod, tracker_mod
    finally:
        for mod_name in list(sys.modules):
            if mod_name not in saved_modules:
                sys.modules.pop(mod_name, None)
        sys.modules.update(saved_modules)


class _DummySurface:
    def __init__(self, name: str, defined: bool = False):
        self.name = name
        self.defined = defined


def test_remove_surface_closes_surface_window(surface_tracker_modules, monkeypatch):
    gui_mod, _ = surface_tracker_modules
    destroy_window = MagicMock()
    monkeypatch.setattr(gui_mod.glfw, "destroy_window", destroy_window)

    tracker = SimpleNamespace()
    gui = gui_mod.GUI(tracker)
    surface = _DummySurface(name="Surface 1")
    gui.add_surface(surface)

    fake_handle = object()
    gui.surface_windows[surface]._window = fake_handle

    gui.remove_surface(surface)

    destroy_window.assert_called_once_with(fake_handle)
    assert surface not in gui.surface_windows


def test_cleanup_closes_all_surface_windows(
    surface_tracker_modules, monkeypatch, tmp_path
):
    gui_mod, tracker_mod = surface_tracker_modules
    destroy_window = MagicMock()
    monkeypatch.setattr(gui_mod.glfw, "destroy_window", destroy_window)

    class _DummySurfaceTracker(tracker_mod.Surface_Tracker):
        @property
        def Surface_Class(self):
            return _DummySurface

        @property
        def _save_dir(self):
            return str(tmp_path)

        @property
        def has_freeze_feature(self):
            return False

        @property
        def ui_info_text(self):
            return ""

        @property
        def supported_heatmap_modes(self):
            return []

        def _update_markers(self, frame):
            pass

        def _update_surface_locations(self, frame_index):
            pass

        def _update_surface_corners(self):
            pass

        def _update_surface_heatmaps(self):
            pass

    g_pool = SimpleNamespace(debug=False)
    tracker = _DummySurfaceTracker(g_pool)
    s1 = _DummySurface(name="S1", defined=False)
    s2 = _DummySurface(name="S2", defined=False)
    tracker.add_surface(s1)
    tracker.add_surface(s2)

    handle_1 = object()
    handle_2 = object()
    tracker.gui.surface_windows[s1]._window = handle_1
    tracker.gui.surface_windows[s2]._window = handle_2

    tracker.cleanup()

    assert destroy_window.call_count == 2
    destroy_window.assert_any_call(handle_1)
    destroy_window.assert_any_call(handle_2)
    assert tracker.gui.surface_windows[s1]._window is None
    assert tracker.gui.surface_windows[s2]._window is None
