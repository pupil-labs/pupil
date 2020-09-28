import glfw
import glfw.GLFW  # TODO: Remove when switching to pyglfw API
import logging


logger = logging.getLogger(__name__)


__windows__ = []
__c_callbacks__ = {}
__py_callbacks__ = {}


def legacy_glfw_get_video_mode(monitor):
    mode_struct = glfw.get_video_mode(monitor)
    return (
        mode_struct.width,
        mode_struct.height,
        mode_struct.red_bits,
        mode_struct.blue_bits,
        mode_struct.green_bits,
        mode_struct.refresh_rate,
    )


def legacy_glfw_get_error():
    code, msg = glfw.get_error()
    if code:
        return msg.value.decode()


def legacy_glfw_init():
    res = glfw.init()
    if res < 0:
        raise Exception(f"GLFW could not be initialized: {legacy_glfw_get_error()}")


def legacy_glfw_create_window(
    width=640, height=480, title="GLFW Window", monitor=None, share=None
):

    window = glfw.create_window(width, height, title, monitor, share)

    if window:
        __windows__.append(window)
        index = __windows__.index(window)
        __c_callbacks__[index] = {}
        __py_callbacks__[index] = {
            "window_pos_fun": None,
            "window_size_fun": None,
            "window_close_fun": None,
            "window_iconify_fun": None,
            "framebuffer_size_fun": None,
            "key_fun": None,
            "char_fun": None,
            "mouse_button_fun": None,
            "cursor_pos_fun": None,
            "scroll_fun": None,
            "drop_fun": None,
        }
        return window
    else:
        raise Exception(f"GLFW window failed to create: {legacy_glfw_get_error()}")


def legacy_glfw_destroy_window(window):
    index = __windows__.index(window)
    try:
        __c_callbacks__[index]
    except KeyError:
        logger.error("Window already destroyed.")
    else:
        glfw.destroy_window(window)
        # We do not delete window from the list (or it would impact windows numbering)
        # del __windows__[index]
        del __c_callbacks__[index]
        del __py_callbacks__[index]


### CALLBACKS

import ctypes


def __set_window_callback(
    window, callback, key: str, c_func_type, glfw_callback_setter
):
    index = __windows__.index(window)
    old_callback = __py_callbacks__[index][key]
    __py_callbacks__[index][key] = callback

    if callback:
        callback = c_func_type(callback)

    __c_callbacks__[index][key] = callback
    glfw_callback_setter(window, callback)
    return old_callback


def legacy_glfw_set_window_pos_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="window_pos_fun",
        c_func_type=ctypes.CFUNCTYPE(
            None, ctypes.POINTER(glfw._GLFWwindow), ctypes.c_int, ctypes.c_int
        ),
        glfw_callback_setter=glfw.set_window_posCallback,
    )


def legacy_glfw_set_window_size_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="window_size_fun",
        c_func_type=ctypes.CFUNCTYPE(
            None, ctypes.POINTER(glfw._GLFWwindow), ctypes.c_int, ctypes.c_int
        ),
        glfw_callback_setter=glfw.set_window_sizeCallback,
    )


def legacy_glfw_set_window_close_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="window_close_fun",
        c_func_type=ctypes.CFUNCTYPE(None, ctypes.POINTER(glfw._GLFWwindow)),
        glfw_callback_setter=glfw.set_window_close_callback,
    )


def legacy_glfw_set_window_iconify_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="window_iconify_fun",
        c_func_type=ctypes.CFUNCTYPE(
            None, ctypes.POINTER(glfw._GLFWwindow), ctypes.c_int
        ),
        glfw_callback_setter=glfw.set_window_iconify_callback,
    )


def legacy_glfw_set_framebuffer_size_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="framebuffer_size_fun",
        c_func_type=ctypes.CFUNCTYPE(
            None, ctypes.POINTER(glfw._GLFWwindow), ctypes.c_int, ctypes.c_int
        ),
        glfw_callback_setter=glfw.set_framebuffer_size_callback,
    )


def legacy_glfw_set_key_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="key_fun",
        c_func_type=ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(glfw._GLFWwindow),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ),
        glfw_callback_setter=glfw.set_key_callback,
    )


def legacy_glfw_set_char_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="char_fun",
        c_func_type=ctypes.CFUNCTYPE(
            None, ctypes.POINTER(glfw._GLFWwindow), ctypes.c_int
        ),
        glfw_callback_setter=glfw.set_char_callback,
    )


def legacy_glfw_set_mouse_button_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="mouse_button_fun",
        c_func_type=ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(glfw._GLFWwindow),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ),
        glfw_callback_setter=glfw.set_mouse_button_callback,
    )


def legacy_glfw_set_cursor_pos_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="cursor_pos_fun",
        c_func_type=ctypes.CFUNCTYPE(
            None, ctypes.POINTER(glfw._GLFWwindow), ctypes.c_double, ctypes.c_double
        ),
        glfw_callback_setter=glfw.set_cursor_pos_callback,
    )


def legacy_glfw_set_scroll_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="scroll_fun",
        c_func_type=ctypes.CFUNCTYPE(
            None, ctypes.POINTER(glfw._GLFWwindow), ctypes.c_double, ctypes.c_double
        ),
        glfw_callback_setter=glfw.set_scroll_callback,
    )


def legacy_glfw_set_drop_callback(window, callback=None):
    __set_window_callback(
        window=window,
        callback=callback,
        key="drop_fun",
        c_func_type=ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(glfw._GLFWwindow),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ),
        glfw_callback_setter=glfw.set_drop_callback,
    )
