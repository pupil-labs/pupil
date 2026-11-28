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
import multiprocessing
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest


def test_world_cleanup_does_not_mask_early_import_error(monkeypatch, tmp_path):
    zmq_stub = types.ModuleType("zmq")
    zmq_stub.Context = MagicMock(return_value=MagicMock())

    zmq_tools_stub = types.ModuleType("zmq_tools")
    dispatcher = MagicMock()
    zmq_tools_stub.Msg_Dispatcher = MagicMock(return_value=dispatcher)
    zmq_tools_stub.Msg_Receiver = MagicMock(return_value=MagicMock())
    zmq_tools_stub.ZMQ_handler = lambda *args, **kwargs: logging.NullHandler()

    monkeypatch.setitem(sys.modules, "zmq", zmq_stub)
    monkeypatch.setitem(sys.modules, "zmq_tools", zmq_tools_stub)
    # Halt an early deferred import inside world()'s try block before g_pool is set.
    monkeypatch.setitem(sys.modules, "background_helper", None)

    from launchables.world import world

    eye_procs_alive = [
        multiprocessing.Value("b", False),
        multiprocessing.Value("b", False),
    ]
    with pytest.raises(ModuleNotFoundError, match="background_helper"):
        world(
            timebase=multiprocessing.Value("d", 0.0),
            eye_procs_alive=eye_procs_alive,
            ipc_pub_url="tcp://127.0.0.1:1",
            ipc_sub_url="tcp://127.0.0.1:2",
            ipc_push_url="tcp://127.0.0.1:3",
            user_dir=str(tmp_path),
            version="3.6",
            preferred_remote_port=50020,
            hide_ui=True,
            debug=False,
            skip_driver_installation=True,
        )

    dispatcher.notify.assert_any_call({"subject": "world_process.stopped"})


def test_pupil_core_spec_collects_scipy_special_cdflib():
    spec_path = (
        pathlib.Path(__file__).resolve().parents[2] / "deployment" / "pupil_core.spec"
    )
    spec_source = spec_path.read_text(encoding="utf-8")
    assert "scipy.special._cdflib" in spec_source
