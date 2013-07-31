# make shared modules available across pupil_src
import sys, os
loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
sys.path.append(os.path.join(loc[0], 'pupil_src', 'shared_modules'))

from glfw import *
import atb
from methods import normalize, denormalize, chessboard, circle_grid, gen_pattern_grid, calibrate_camera,Temp
from uvc_capture import autoCreateCapture



# Initialize glfw
glfwInit()
height,width = img.shape[:2]
glfwOpenWindow(width, height, 0, 0, 0, 8, 0, 0, GLFW_WINDOW)
glfwSetWindowTitle("World")
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
while glfwGetWindowParam(GLFW_OPENED) and not g_pool.quit.value:
    update_fps()

    # Get input characters entered in player
    if g_pool.player_input.value:
        player_input = g_pool.player_input.value
        g_pool.player_input.value = 0
        on_char(player_input,True)

    # Get an image from the grabber
    s, img = cap.read()

    for p in g.plugins:
        p.update(img)

    g.plugins = [p for p in g.plugins if p.alive]

    g_pool.player_refresh.set()

    # render the screen
    clear_gl_screen()
    draw_gl_texture(img)

    # render visual feedback from loaded plugins
    for p in g.plugins:
        p.gl_display()


    # update gaze point from shared variable pool and draw on screen. If both coords are 0: no pupil pos was detected.
    if not g_pool.gaze[:] == [0.,0.]:
        draw_gl_point_norm(g_pool.gaze[:],color=(1.,0.,0.,0.5))

    atb.draw()
    glfwSwapBuffers()

# end while running and clean-up
print "Browser closed"
glfwCloseWindow()
glfwTerminate()