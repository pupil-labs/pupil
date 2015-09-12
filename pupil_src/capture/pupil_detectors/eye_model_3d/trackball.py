'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from OpenGL.GL import *
from OpenGL.GLU import gluPerspective

class Trackball(object):
    """docstring for Trackball"""
    def __init__(self):
        super(Trackball, self).__init__()

        self.distance = [0,0,-40]
        self.pitch = 0
        self.roll = 0
        self.aspect = 1.
        self.window = 1,1


    def push(self):
        glMatrixMode( GL_PROJECTION )
        glPushMatrix()
        glLoadIdentity( )
        gluPerspective( 30.0, self.aspect, 0.1, 10000.0 )
        glTranslatef(*self.distance)
        glRotatef(0,1,0,0)
        glRotatef(self.pitch,1,0,0)
        glRotatef(self.roll,0,1,0)
        glMatrixMode( GL_MODELVIEW )
        glPushMatrix()

    def pop(self):
        glMatrixMode( GL_PROJECTION )
        glPopMatrix()
        glMatrixMode( GL_MODELVIEW )
        glPopMatrix()

    def drag_to(self,dx,dy):
        self.pitch += dy*(360./self.window[1])
        self.roll -= dx*(360./self.window[0])

    def pan_to(self,dx,dy):
        self.distance[0] +=dx/10.
        self.distance[1] -=dy/10.

    def zoom_to(self,dy):
        self.distance[2] += dy

    def set_window_size(self,w,h):
        self.aspect = float(w)/h
        self.window = w,h

    def __repr__(self):
        return "Trackball: viewing distance: %s, roll: %2.0fdeg, pitch %2.0fdeg"%(self.distance,self.roll,self.pitch)
