'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from glfw import *
from OpenGL.GL import *
from platform import system

from pyglui.cygl.utils import RGBA
from pyglui.cygl import utils as glutils
from pyglui.pyfontstash import fontstash as fs
from pyglui.ui import get_opensans_font_path
import math
import numpy as np

#UI Platform tweaks
if system() == 'Linux':
    window_position_default = (0,0)
elif system() == 'Windows':
    window_position_default = (8,31)
else:
    window_position_default = (0,0)

class Visualizer(object):
    """docstring for Visualizer
    Visualizer is a base class for all visualizations in new windows
    """


    def __init__(self, g_pool, name = "Visualizer", run_independently = False):

        self.name = name
        self.window_size = (640,480)
        self.window = None
        self.input = None
        self.run_independently = run_independently
        self.sphere = None
        self.other_window = None
        self.g_pool = g_pool

    def begin_update_window(self ):
        if self.window:
            if glfwWindowShouldClose(self.window):
                self.close_window()
                return

            self.other_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self.window)


    def update_window(self):
        pass

    def end_update_window(self ):
        if self.window:
            glfwSwapBuffers(self.window)
            glfwPollEvents()
        glfwMakeContextCurrent(self.other_window)


    ############## DRAWING FUNCTIONS ##############################

    def draw_frustum(self, width, height , length):

        W = width/2.0
        H = height/2.0
        Z = length
        # draw it
        glLineWidth(1)
        glColor4f( 1, 0.5, 0, 0.5 )
        glBegin( GL_LINE_LOOP )
        glVertex3f( 0, 0, 0 )
        glVertex3f( -W, H, Z )
        glVertex3f( W, H, Z )
        glVertex3f( 0, 0, 0 )
        glVertex3f( W, H, Z )
        glVertex3f( W, -H, Z )
        glVertex3f( 0, 0, 0 )
        glVertex3f( W, -H, Z )
        glVertex3f( -W, -H, Z )
        glVertex3f( 0, 0, 0 )
        glVertex3f( -W, -H, Z )
        glVertex3f( -W, H, Z )
        glEnd( )

    def draw_coordinate_system(self,l=1):
        # Draw x-axis line. RED
        glLineWidth(2)
        glColor3f( 1, 0, 0 )
        glBegin( GL_LINES )
        glVertex3f( 0, 0, 0 )
        glVertex3f( l, 0, 0 )
        glEnd( )

        # Draw y-axis line. GREEN.
        glColor3f( 0, 1, 0 )
        glBegin( GL_LINES )
        glVertex3f( 0, 0, 0 )
        glVertex3f( 0, l, 0 )
        glEnd( )

        # Draw z-axis line. BLUE
        glColor3f( 0, 0, 1 )
        glBegin( GL_LINES )
        glVertex3f( 0, 0, 0 )
        glVertex3f( 0, 0, l )
        glEnd( )

    def draw_sphere(self,sphere_position, sphere_radius,contours = 45, color =RGBA(.2,.5,0.5,.5) ):

        glPushMatrix()
        glTranslatef(sphere_position[0],sphere_position[1],sphere_position[2])
        glScale(sphere_radius,sphere_radius,sphere_radius)
        self.sphere.draw(color, primitive_type = GL_LINE_STRIP)
        glPopMatrix()

    def basic_gl_setup(self):
        glEnable(GL_POINT_SPRITE )
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE) # overwrite pointsize
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glClearColor(.8,.8,.8,1.)
        glEnable(GL_LINE_SMOOTH)
        # glEnable(GL_POINT_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

    def adjust_gl_view(self,w,h):
        """
        adjust view onto our scene.
        """
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def clear_gl_screen(self):
        glClearColor(0.9,0.9,0.9,1.)
        glClear(GL_COLOR_BUFFER_BIT)

    def close_window(self):
        if self.window:
            glfwDestroyWindow(self.window)
            self.window = None

    def open_window(self):
        if not self.window:
            self.input = {'button':None, 'mouse':(0,0)}


            # get glfw started
            if self.run_independently:
                glfwInit()
                self.window = glfwCreateWindow(self.window_size[0], self.window_size[1], self.name, None  )
            else:
                self.window = glfwCreateWindow(self.window_size[0], self.window_size[1], self.name, None, share= glfwGetCurrentContext() )

            self.other_window = glfwGetCurrentContext();

            glfwMakeContextCurrent(self.window)

            glfwSetWindowPos(self.window,window_position_default[0],window_position_default[1])
            # Register callbacks window
            glfwSetFramebufferSizeCallback(self.window,self.on_resize)
            glfwSetWindowIconifyCallback(self.window,self.on_iconify)
            glfwSetKeyCallback(self.window,self.on_key)
            glfwSetCharCallback(self.window,self.on_char)
            glfwSetMouseButtonCallback(self.window,self.on_button)
            glfwSetCursorPosCallback(self.window,self.on_pos)
            glfwSetScrollCallback(self.window,self.on_scroll)

            # get glfw started
            if self.run_independently:
                glutils.init()
            self.basic_gl_setup()

            self.sphere = glutils.Sphere(20)


            self.glfont = fs.Context()
            self.glfont.add_font('opensans',get_opensans_font_path())
            self.glfont.set_size(18)
            self.glfont.set_color_float((0.2,0.5,0.9,1.0))
            self.on_resize(self.window,*glfwGetFramebufferSize(self.window))
            glfwMakeContextCurrent(self.other_window)



    ############ window callbacks #################
    def on_resize(self,window,w, h):
        h = max(h,1)
        w = max(w,1)

        self.window_size = (w,h)
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        self.adjust_gl_view(w,h)
        glfwMakeContextCurrent(active_window)

    def on_button(self,window,button, action, mods):
        # self.gui.update_button(button,action,mods)
        if action == GLFW_PRESS:
            self.input['button'] = button
            self.input['mouse'] = glfwGetCursorPos(window)
        if action == GLFW_RELEASE:
            self.input['button'] = None

    def on_pos(self,window,x, y):
        hdpi_factor = float(glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0])
        x,y = x*hdpi_factor,y*hdpi_factor
        # self.gui.update_mouse(x,y)
        if self.input['button']==GLFW_MOUSE_BUTTON_RIGHT:
            old_x,old_y = self.input['mouse']
            self.trackball.drag_to(x-old_x,y-old_y)
            self.input['mouse'] = x,y
        if self.input['button']==GLFW_MOUSE_BUTTON_LEFT:
            old_x,old_y = self.input['mouse']
            self.trackball.pan_to(x-old_x,y-old_y)
            self.input['mouse'] = x,y

    def on_char(self,window,char):
        pass

    def on_scroll(self,window,x,y):
        pass

    def on_iconify(self,window,iconified):
        pass

    def on_key(self,window, key, scancode, action, mods):
        pass
