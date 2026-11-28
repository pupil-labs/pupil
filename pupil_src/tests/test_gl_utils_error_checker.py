"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import importlib.util
from pathlib import Path

import OpenGL.error

_GL_UTILS_PATH = (
    Path(__file__).resolve().parents[1] / "shared_modules" / "gl_utils" / "utils.py"
)


def _load_gl_utils_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _GL_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_gl_utils_with_immutable_error_checker(monkeypatch):
    class _ImmutableMeta(type):
        def __setattr__(cls, name, value):
            raise TypeError(
                "can't set attributes of built-in/extension type "
                "'OpenGL_accelerate.errorchecker._ErrorChecker'"
            )

    class _FakeAcceleratedErrorChecker(metaclass=_ImmutableMeta):
        def __init__(self, *args, **kwargs):
            pass

        def glCheckError(self, result, baseOperation=None, cArguments=None, *args):
            return result

    monkeypatch.setattr(OpenGL.error, "_ErrorChecker", _FakeAcceleratedErrorChecker)
    module = _load_gl_utils_module("gl_utils_utils_accelerated_test")
    assert callable(module.custom_gl_error_handling)


def test_import_gl_utils_with_disabled_error_checker(monkeypatch):
    monkeypatch.setattr(OpenGL.error, "_ErrorChecker", None)
    module = _load_gl_utils_module("gl_utils_utils_disabled_test")
    assert module._original_gl_error_check is None


def test_import_gl_utils_with_mutable_error_checker(monkeypatch):
    class _MutableErrorChecker:
        def glCheckError(self, result, baseOperation=None, cArguments=None, *args):
            return result

    monkeypatch.setattr(OpenGL.error, "_ErrorChecker", _MutableErrorChecker)
    module = _load_gl_utils_module("gl_utils_utils_mutable_test")
    assert _MutableErrorChecker.glCheckError is module.custom_gl_error_handling
