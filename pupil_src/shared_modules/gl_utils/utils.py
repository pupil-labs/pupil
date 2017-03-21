'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import OpenGL
# OpenGL.FULL_LOGGING = True
OpenGL.ERROR_LOGGING = False
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective

import numpy as np
import math
import glfw

__all__ =  ['make_coord_system_norm_based',
            'make_coord_system_pixel_based',
            'make_coord_system_eye_camera_based',
            'adjust_gl_view',
            'clear_gl_screen',
            'basic_gl_setup',
            'cvmat_to_glmat',
            'is_window_visible'
]

def is_window_visible(window):
    visible = glfw.glfwGetWindowAttrib(window, glfw.GLFW_VISIBLE)
    iconified = glfw.glfwGetWindowAttrib(window, glfw.GLFW_ICONIFIED)
    return visible and not iconified

def cvmat_to_glmat(m):
    mat = np.eye(4,dtype=np.float32)
    mat = mat.flatten()
    # convert to OpenGL matrix
    mat[0]  = m[0,0]
    mat[4]  = m[0,1]
    mat[12] = m[0,2]
    mat[1]  = m[1,0]
    mat[5]  = m[1,1]
    mat[13] = m[1,2]
    mat[3]  = m[2,0]
    mat[7]  = m[2,1]
    mat[15] = m[2,2]
    return mat


def basic_gl_setup():
    glEnable( GL_POINT_SPRITE )
    # glEnable(GL_POINT_SMOOTH)
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE) # overwrite pointsize
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)
    glClearColor(1.,1.,1.,0.)
    glEnable(GL_LINE_SMOOTH)
    # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    # glEnable(GL_POLYGON_SMOOTH)
    # glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)


def clear_gl_screen():
    glClear(GL_COLOR_BUFFER_BIT)

def adjust_gl_view(w,h):
    """
    adjust view onto our scene.
    """
    h = max(h,1)
    w = max(w,1)
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, w, h, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()



def make_coord_system_pixel_based(img_shape,flip=False):
    height,width,channels = img_shape
    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if flip:
        glOrtho(width,0, 0, height,-1,1) # origin in the top left corner just like the img np-array
    else:
        glOrtho(0, width, height, 0,-1,1) # origin in the top left corner just like the img np-array

    # Switch back to Model View Matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def make_coord_system_eye_camera_based(window_size, focal_length ):
    camera_fov = math.degrees(2.0 * math.atan( window_size[1] / (2.0 * focal_length)))
    glMatrixMode( GL_PROJECTION )
    glLoadIdentity( )
    gluPerspective( camera_fov, float(window_size[0])/window_size[1], 0.1, 2000.0 )
    glMatrixMode( GL_MODELVIEW )
    glLoadIdentity( )


def make_coord_system_norm_based(flip=False):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if flip:
        glOrtho(1, 0, 1, 0,-1,1) # gl coord convention
    else:
        glOrtho(0, 1, 0, 1,-1,1) # gl coord convention

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

