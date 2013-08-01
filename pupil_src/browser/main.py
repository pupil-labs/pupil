# make shared modules available across pupil_src
import sys, os
loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
sys.path.append(os.path.join(loc[0], 'pupil_src', 'shared_modules'))

import numpy as np
from time import time
from ctypes import  c_int,c_bool,c_float,create_string_buffer
from glfw import *
import atb

from uvc_capture import autoCreateCapture
from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen, draw_gl_point_norm


# Callback functions
def on_resize(w, h):
    atb.TwWindowSize(w, h);
    adjust_gl_view(w,h)

def on_key(key, pressed):
    if not atb.TwEventKeyboardGLFW(key,pressed):
        if pressed:
            if key == GLFW_KEY_ESC:
                on_close()

def on_char(char, pressed):
    if not atb.TwEventCharGLFW(char,pressed):
        pass

def on_button(button, pressed):
    if not atb.TwEventMouseButtonGLFW(button,pressed):
        if pressed:
            pos = glfwGetMousePos()
            pos = normalize(pos,glfwGetWindowSize())
            pos = denormalize(pos,(img.shape[1],img.shape[0]) ) # Position in img pixels
            for p in g.plugins:
                p.on_click(pos)

def on_pos(x, y):
    if atb.TwMouseMotion(x,y):
        pass

def on_scroll(pos):
    if not atb.TwMouseWheel(pos):
        pass

def on_close():
    quit.value = True
    print "WORLD Process closing from window"

# helpers called by the main atb bar
def update_fps():
    old_time, bar.timestamp = bar.timestamp, time()
    dt = bar.timestamp - old_time
    if dt:
        bar.fps.value += .05 * (1 / dt - bar.fps.value)

def set_window_size(mode,data):
    height,width = img.shape[:2]
    ratio = (1,.75,.5,.25)[mode]
    w,h = int(width*ratio),int(height*ratio)
    glfwSetWindowSize(w,h)
    data.value=mode # update the bar.value

def get_from_data(data):
    """
    helper for atb getter and setter use
    """
    return data.value

# Initialize ant tweak bar - inherits from atb.Bar
atb.init()
bar = atb.Bar(name = "Browser", label="Controls",
        help="Scene controls", color=(50, 50, 50), alpha=100,valueswidth=150,
        text='light', position=(10, 10),refresh=.3, size=(300, 200))
bar.next_atb_pos = (10,220)
bar.fps = c_float(0.0)
bar.timestamp = time()

bar.window_size = c_int(0)
window_size_enum = atb.enum("Display Size",{"Full":0, "Medium":1,"Half":2,"Mini":3})


bar.add_var("fps", bar.fps, step=1., readonly=True)
bar.add_var("display size", vtype=window_size_enum,setter=set_window_size,getter=get_from_data,data=bar.window_size)
# bar.add_var("exit", g_pool.quit)


# Initialize glfw
glfwInit()
width, height = 640,480
glfwOpenWindow(width, height, 0, 0, 0, 8, 0, 0, GLFW_WINDOW)
glfwSetWindowTitle("Browser")
glfwSetWindowPos(0,0)

# Register callbacks
glfwSetWindowSizeCallback(on_resize)
glfwSetWindowCloseCallback(on_close)
glfwSetKeyCallback(on_key)
glfwSetCharCallback(on_char)
glfwSetMouseButtonCallback(on_button)
glfwSetMousePosCallback(on_pos)
glfwSetMouseWheelCallback(on_scroll)

# gl_state settings
import OpenGL.GL as gl
gl.glEnable(gl.GL_POINT_SMOOTH)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
gl.glEnable(gl.GL_BLEND)
del gl

# Event loop
while glfwGetWindowParam(GLFW_OPENED) and not quit.value:
    update_fps()

    # Get an image from the grabber
    # s, img = cap.read()
    img = np.zeros((480,640,3))

    for p in g.plugins:
        p.update(img)

    g.plugins = [p for p in g.plugins if p.alive]

    # render the screen
    clear_gl_screen()
    draw_gl_texture(img)

    # render visual feedback from loaded plugins
    for p in g.plugins:
        p.gl_display()


    atb.draw()
    glfwSwapBuffers()

# end while running and clean-up
print "Browser closed"
glfwCloseWindow()
glfwTerminate()


if __name__ == '__main__':
    pass