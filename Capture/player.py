import os, sys
import OpenGL.GL as gl
from glfw import *
import numpy as np
import cv2
from methods import Temp
from uvc_capture import Capture
from time import sleep
from glob import glob
from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen


def make_grid(dim=(11,4)):
    """
    this function generates the structure for an asymetrical circle grid
    It returns a Vertext Buffer Object that is used by glumpy to draw it in
    the opengl Window.
    """
    x,y = range(dim[0]),range(dim[1])
    p = np.array([[[s,i] for s in x] for i in y], dtype=np.float32)
    p[:,1::2,1] += 0.5

    # width  (on sceen this is the height) of pattern is 1
    # height is scaled accordingly to preserve aspect ratio
    p[:,:,0] /= (dim[1]+1)*2
    p[:,:,1] /= (dim[1]+1)
    p = np.reshape(p, (-1,2), 'F')
    return p


def player(g_pool,size):
    """player
        - Shows 9 point calibration pattern
        - Plays a source video synchronized with world process
        - Get src videos from directory (glob)
        - Iterate through videos on each record event
    """

    grid = make_grid()

    # player object
    player = Temp()
    player.play_list = glob('src_video/*')
    path_parent = os.path.dirname( os.path.abspath(sys.argv[0]))
    player.playlist = [os.path.join(path_parent, path) for path in player.play_list]
    player.captures = [Capture(src) for src in player.playlist]
    print "Player found %i videos in src_video"%len(player.captures)
    player.captures =  [c for c in player.captures if c is not None]
    print "Player sucessfully loaded %i videos in src_video"%len(player.captures)
    # for c in player.captures: c.auto_rewind = False
    player.current_video = 0

    # Callbacks
    def on_resize(w, h):
        adjust_gl_view(w,h)

    def on_key(key, pressed):
        if key == GLFW_KEY_ESC:
                on_close()
    def on_char(char, pressed):
        if char  == ord('9'):
            g_pool.cal9.value = True


    def on_close():
        g_pool.quit.value = True
        print "Player Process closing from window"


    # initialize glfw
    glfwInit()
    glfwOpenWindow(size[0], size[1], 0, 0, 0, 8, 0, 0, GLFW_WINDOW)
    glfwSetWindowTitle("Player")
    glfwSetWindowPos(100,0)
    glfwDisable(GLFW_AUTO_POLL_EVENTS)


    #Callbacks
    glfwSetWindowSizeCallback(on_resize)
    glfwSetWindowCloseCallback(on_close)
    glfwSetKeyCallback(on_key)
    glfwSetCharCallback(on_char)


    #gl state settings
    gl.glEnable( gl.GL_BLEND )
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glClearColor(1.,1.,1.,0.)


    while glfwGetWindowParam(GLFW_OPENED) and not g_pool.quit.value:

        glfwPollEvents()

        if g_pool.player_refresh.wait(0.01):
            g_pool.player_refresh.clear()

            clear_gl_screen()
            if g_pool.cal9.value:
                circle_id,step = g_pool.cal9_circle_id.value,g_pool.cal9_step.value
                gl.glPushMatrix()
                gl.glTranslatef(-1+.1,-8/11.+.1,0.) # center the pattern on the screen adjust_gl_view help us too
                gl.glScalef(1.9,1.9,1.0)
                gl.glColor4f(0.0,0.0,0.0,1.0)
                gl.glPointSize(40)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                gl.glBegin(gl.GL_POINTS)
                for p in grid:
                    gl.glVertex3f(p[0],p[1],0.0)
                gl.glEnd()

                gl.glPointSize((40)*(1.01-(step+1)/80.0))
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ZERO)
                gl.glColor4f(1.0,0.0,0.0,1.0)
                gl.glBegin(gl.GL_POINTS)
                gl.glVertex3f(grid[circle_id][0],grid[circle_id][1],0.0)
                gl.glEnd()
                gl.glPopMatrix()

            elif g_pool.play.value:
                s, img = player.captures[player.current_video].read_RGB()
                if s:
                    draw_gl_texture(image)
                else:
                    player.captures[player.current_video].rewind()
                    player.current_video +=1
                    if player.current_video >= len(player.captures):
                        player.current_video = 0
                    g_pool.play.value = False
            glfwSwapBuffers()

    glfwCloseWindow()
    glfwTerminate()
    print "PLAYER Process closed"

