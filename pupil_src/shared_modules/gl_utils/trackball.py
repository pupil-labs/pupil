"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from OpenGL.GL import (
    GL_MODELVIEW,
    GL_PROJECTION,
    glLoadIdentity,
    glMatrixMode,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTranslatef,
)
from OpenGL.GLU import gluPerspective


class Trackball:
    def __init__(self, fov=30):
        super().__init__()

        self.distance = [0, 0, 0.1]
        self.pitch = 0
        self.roll = 0
        self.aspect = 1.0
        self.fov = fov
        self.window = 1, 1

    def push(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(self.fov, self.aspect, 0.1, 200000.0)
        glTranslatef(*self.distance)
        glRotatef(0, 1, 0, 0)
        glRotatef(self.pitch, 1, 0, 0)
        glRotatef(self.roll, 0, 1, 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

    def pop(self):
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

    def drag_to(self, dx, dy):
        self.pitch += dy * (360.0 / self.window[1])
        self.roll -= dx * (360.0 / self.window[0])

    def pan_to(self, dx, dy):
        self.distance[0] += dx / 10.0
        self.distance[1] -= dy / 10.0

    def zoom_to(self, dy):
        self.distance[2] += dy

    def set_window_size(self, w, h):
        h = max(h, 1)
        w = max(w, 1)

        self.aspect = float(w) / h
        self.window = w, h

    def __repr__(self):
        return "Trackball: viewing distance: {}, roll: {:2.0f}deg, pitch {:2.0f}deg".format(
            self.distance,
            self.roll,
            self.pitch,
        )
