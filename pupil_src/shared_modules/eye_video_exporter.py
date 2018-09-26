"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

import cv2
import numpy as np
from pyglui import ui

from video_exporter import VideoExporter

logger = logging.getLogger(__name__)


def getEllipsePts(e, num_pts=10):
    c1 = e[0][0]
    c2 = e[0][1]
    a = e[1][0]
    b = e[1][1]
    angle = e[2]

    steps = np.linspace(0, 2 * np.pi, num=num_pts, endpoint=False)
    rot = cv2.getRotationMatrix2D((0, 0), -angle, 1)

    pts1 = a / 2.0 * np.cos(steps)
    pts2 = b / 2.0 * np.sin(steps)
    pts = np.column_stack((pts1, pts2, np.ones(pts1.shape[0])))

    pts_rot = np.matmul(rot, pts.T)
    pts_rot = pts_rot.T

    pts_rot[:, 0] += c1
    pts_rot[:, 1] += c2

    return pts_rot


def _no_change(capture, frame):
    return frame.img


def _add_pupil_ellipse(eyeid, pupil_positions):
    def add_pupil_ellipse(capture, frame):
        try:
            pp = next(
                (
                    pp
                    for pp in pupil_positions
                    if pp["id"] == eyeid and pp["timestamp"] == frame.timestamp
                )
            )
        except StopIteration:
            return frame.img
        else:
            eye_image = frame.img
            el = pp["ellipse"]
            conf = int(pp.get("model_confidence", pp.get("confidence", 0.1)) * 255)
            el_points = getEllipsePts((el["center"], el["axes"], el["angle"]))
            cv2.polylines(
                eye_image,
                [np.asarray(el_points, dtype="i")],
                True,
                (0, 0, 255, conf),
                thickness=1,
            )
            cv2.circle(
                eye_image,
                (int(el["center"][0]), int(el["center"][1])),
                5,
                (0, 0, 255, conf),
                thickness=-1,
            )
            return eye_image

    return add_pupil_ellipse


class Eye_Video_Exporter(VideoExporter):
    # TODO: docstring
    icon_chr = "EV"

    def __init__(self, g_pool, render_pupil=True):
        super().__init__(g_pool)
        self.render_pupil = render_pupil
        logger.info("Eye Video Exporter has been launched.")

    def customize_menu(self):
        self.menu.label = "Eye Video Exporter"
        self.menu.append(ui.Switch('render_pupil', self, label='Render detected pupil'))

    def _export_eye_video(self, export_range, export_dir, eye_id):
        if self.render_pupil:
            process_frame = _add_pupil_ellipse(eye_id, self.g_pool.pupil_positions)
        else:
            process_frame = _no_change
        eye_name = "eye" + str(eye_id)
        try:
            self.add_export_job(
                export_range, export_dir, "EyeVideo", eye_name, eye_name, process_frame
            )
        except FileNotFoundError:
            # happens if there is no such eye video
            pass

    def export_data(self, export_range, export_dir):
        self._export_eye_video(export_range, export_dir, eye_id=0)
        self._export_eye_video(export_range, export_dir, eye_id=1)
