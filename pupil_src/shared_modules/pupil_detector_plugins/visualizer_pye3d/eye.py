"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import cv2
import numpy as np
from OpenGL.GL import *

from .pose import PosedObject
from .utilities import (
    normalize,
    rotate_v1_on_v2,
    sph2cart,
    transform_as_homogeneous_point,
    transform_as_homogeneous_vector,
)


class BasicEye(PosedObject):
    def __init__(self):
        super().__init__(pose=np.eye(4), extrinsics=None, children=())

        self._gaze_vector = PosedObject()
        self.eyeball_center = [0.0, 0.0, 0.0]

    def update_from_gaze_point(self, gaze_point):
        new_gaze_vector = transform_as_homogeneous_vector(
            normalize(gaze_point - self.eyeball_center), self.extrinsics
        )
        self.update_from_gaze_vector(new_gaze_vector)

    def update_from_spherical(self, phi, theta):
        new_gaze_vector = sph2cart(phi, theta)
        self.update_from_gaze_vector(new_gaze_vector)

    def update_from_gaze_vector(self, new_gaze_vector):
        rotation = rotate_v1_on_v2([0.0, 0.0, 1.0], new_gaze_vector)
        self._gaze_vector.rmat = rotation

    def move_to_point(self, point):
        self.translate(point - self.eyeball_center)

    @property
    def eyeball_center(self):
        return self.tvec

    @eyeball_center.setter
    def eyeball_center(self, point):
        self.tvec = np.asarray(point)

    @property
    def gaze_vector(self):
        return (self._gaze_vector.pose @ self.pose)[:3, 2]

    def __str__(self):
        return "\n".join(f"{k}:{v}" for k, v in self.__dict__.items())


class LeGrandEye(BasicEye):
    def __init__(
        self,
        eyeball_radius=12.0,
        cornea_radius=7.8,
        iris_radius=6.0,
        n_refraction=1.3375,
        camera=None,
    ):

        super().__init__()

        self.model_type = "LeGrand"

        # PUPIL
        distance_eyeball_pupil = np.sqrt(eyeball_radius**2 - iris_radius**2)
        self.__pupil_center = [0.0, 0.0, distance_eyeball_pupil]
        self.pupil_radius = 2.0
        self.pupil_normal = np.asarray([0.0, 0.0, 1.0])

        # IRIS
        self.__iris_center = self.__pupil_center
        self.iris_radius = iris_radius
        self.iris_normal = np.asarray([0.0, 0.0, 1.0])
        self.iris_color = [46 / 255.0, 220 / 255.0, 255.0 / 255.0]

        # CORNEA
        h = np.sqrt(cornea_radius**2 - iris_radius**2)
        distance_eyeball_cornea = distance_eyeball_pupil - h
        self.__cornea_center = np.asarray([0, 0, distance_eyeball_cornea])
        self.cornea_radius = cornea_radius

        # EYEBALL
        self.eyeball_radius = eyeball_radius

        # self.translate(np.asarray([0., 0., 35.]))
        # self.update_from_gaze_vector(np.asarray([0., 0., -1]))

        # PHYSICAL CONSTANTS
        self.n_refraction = n_refraction

        # CAMERA POINTED AT EYE
        self.camera = camera

        # GL SETUP
        self.eyeball_alpha = np.arccos(distance_eyeball_pupil / self.eyeball_radius)
        self.cornea_alpha = np.arccos(4.0 / self.cornea_radius) / 1.0
        self.set_up_gl_vertices()

    @property
    def cornea_center(self):
        cornea_center = transform_as_homogeneous_point(
            self.__cornea_center, self.pose @ self._gaze_vector.pose
        )
        return cornea_center

    @property
    def iris_center(self):
        iris_center = transform_as_homogeneous_point(
            self.__iris_center, self.pose @ self._gaze_vector.pose
        )
        return iris_center

    @property
    def pupil_center(self):
        pupil_center = transform_as_homogeneous_point(
            self.__pupil_center, self.pose @ self._gaze_vector.pose
        )
        return pupil_center

    def set_up_gl_vertices(self):

        # EYEBALL
        self.central_ring_eyeball = [
            [self.eyeball_radius * np.sin(phi), 0, self.eyeball_radius * np.cos(phi)]
            for phi in np.linspace(
                self.eyeball_alpha, 2 * np.pi - self.eyeball_alpha, 30
            )
        ]
        self.rings_eyeball = [self.central_ring_eyeball]
        for phi in np.linspace(0, np.pi, 20):
            central_ring_rotated = [
                cv2.Rodrigues(np.asarray([0.0, 0.0, phi]))[0] @ v
                for v in self.central_ring_eyeball
            ]
            self.rings_eyeball.append(central_ring_rotated)

        # IRIS
        angles = [phi for phi in np.linspace(0, 2 * np.pi, 40)]
        self.iris_quads = []
        for i in range(len(angles) - 1):
            self.iris_quads.append(
                [
                    np.array((np.cos(angles[i]), np.sin(angles[i]), 0)),
                    np.array((np.cos(angles[i + 1]), np.sin(angles[i + 1]), 0)),
                    np.array((np.cos(angles[i + 1]), np.sin(angles[i + 1]), 0)),
                    np.array((np.cos(angles[i]), np.sin(angles[i]), 0)),
                ]
            )

        # CORNEA
        self.central_ring_cornea = [
            [self.cornea_radius * np.sin(phi), 0, self.cornea_radius * np.cos(phi)]
            for phi in np.linspace(-self.cornea_alpha, self.cornea_alpha, 20)
        ]

        self.rings_cornea = [self.central_ring_cornea]
        for phi in np.linspace(0, np.pi, 10):
            central_ring_rotated = [
                cv2.Rodrigues(np.asarray([0.0, 0.0, phi]))[0] @ v
                for v in self.central_ring_cornea
            ]
            self.rings_cornea.append(central_ring_rotated)

    def draw_gl(
        self,
        draw_eyeball=True,
        draw_iris=True,
        draw_cornea=True,
        draw_gaze=True,
        alpha=1.0,
        color_gaze=(1.0, 1.0, 1.0),
        color_eyeball=(0.6, 0.6, 1.0),
        color_cornea=(1.0, 1.0, 1.0),
    ):

        glPushMatrix()

        glLoadIdentity()
        # if self.camera is not None:
        #     glMultMatrixf(self.camera.pose.T)
        glMultMatrixf(self.pose.T)
        glMultMatrixf(self._gaze_vector.pose.T)

        glPushMatrix()

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if draw_gaze:
            glLineWidth(2.0)
            glColor4f(*color_gaze, 1.0 * alpha)
            glBegin(GL_LINES)
            glVertex3f(*[0, 0, 0])
            glVertex3f(*[0, 0, 600])
            glEnd()

        # DRAW EYEBALL
        if draw_eyeball:

            # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glColor4f(*color_eyeball, 1.0 * alpha)
            glLineWidth(1.0)

            glPushMatrix()
            for i in range(len(self.rings_eyeball) - 1):
                for j in range(len(self.rings_eyeball[i]) - 1):
                    glBegin(GL_QUADS)
                    glVertex3f(
                        self.rings_eyeball[i][j][0],
                        self.rings_eyeball[i][j][1],
                        self.rings_eyeball[i][j][2],
                    )
                    glVertex3f(
                        self.rings_eyeball[i][j + 1][0],
                        self.rings_eyeball[i][j + 1][1],
                        self.rings_eyeball[i][j + 1][2],
                    )
                    glVertex3f(
                        self.rings_eyeball[i + 1][j + 1][0],
                        self.rings_eyeball[i + 1][j + 1][1],
                        self.rings_eyeball[i + 1][j + 1][2],
                    )
                    glVertex3f(
                        self.rings_eyeball[i + 1][j][0],
                        self.rings_eyeball[i + 1][j][1],
                        self.rings_eyeball[i + 1][j][2],
                    )
                    glEnd()
            glPopMatrix()

        # DRAW IRIS
        if draw_iris:

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glColor4f(
                self.iris_color[0], self.iris_color[1], self.iris_color[2], 0.4 * alpha
            )

            glPushMatrix()
            glTranslate(0, 0, self.__pupil_center[2])
            for quad in self.iris_quads:
                glBegin(GL_QUADS)
                glVertex3f(*(quad[0] * self.pupil_radius))
                glVertex3f(*(quad[1] * self.pupil_radius))
                glVertex3f(*(quad[2] * self.iris_radius))
                glVertex3f(*(quad[3] * self.iris_radius))
                glEnd()
            glPopMatrix()

        # DRAW CORNEA
        if draw_cornea:

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor4f(*color_cornea, 0.3 * alpha)
            glLineWidth(1.0)

            glPushMatrix()
            glTranslate(0, 0, self.__cornea_center[2])
            for i in range(len(self.rings_cornea) - 1):
                for j in range(len(self.rings_cornea[i]) - 1):
                    glBegin(GL_QUADS)
                    glVertex3f(
                        self.rings_cornea[i][j][0],
                        self.rings_cornea[i][j][1],
                        self.rings_cornea[i][j][2],
                    )
                    glVertex3f(
                        self.rings_cornea[i][j + 1][0],
                        self.rings_cornea[i][j + 1][1],
                        self.rings_cornea[i][j + 1][2],
                    )
                    glVertex3f(
                        self.rings_cornea[i + 1][j + 1][0],
                        self.rings_cornea[i + 1][j + 1][1],
                        self.rings_cornea[i + 1][j + 1][2],
                    )
                    glVertex3f(
                        self.rings_cornea[i + 1][j][0],
                        self.rings_cornea[i + 1][j][1],
                        self.rings_cornea[i + 1][j][2],
                    )
                    glEnd()
            glPopMatrix()

        glPopMatrix()

        glPopMatrix()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
