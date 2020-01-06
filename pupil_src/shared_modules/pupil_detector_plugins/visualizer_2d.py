"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import cv2

from pyglui.cygl.utils import RGBA, draw_polyline, draw_points


def draw_eyeball_outline(pupil_detection_result_3d):
    eye_ball = pupil_detection_result_3d["projected_sphere"]
    try:
        pts = cv2.ellipse2Poly(
            (int(eye_ball["center"][0]), int(eye_ball["center"][1])),
            (int(eye_ball["axes"][0] / 2), int(eye_ball["axes"][1] / 2)),
            int(eye_ball["angle"]),
            0,
            360,
            8,
        )
    except ValueError:
        # Happens when converting 'nan' to int
        # TODO: Investigate why results are sometimes 'nan'
        return
    draw_polyline(
        pts, 2, RGBA(0.0, 0.9, 0.1, pupil_detection_result_3d["model_confidence"])
    )


def draw_pupil_outline(pupil_detection_result_2d):
    """Requires `"ellipse" in pupil_detection_result_2d`"""
    if pupil_detection_result_2d["confidence"] <= 0.0:
        return

    try:
        pts = cv2.ellipse2Poly(
            (
                int(pupil_detection_result_2d["ellipse"]["center"][0]),
                int(pupil_detection_result_2d["ellipse"]["center"][1]),
            ),
            (
                int(pupil_detection_result_2d["ellipse"]["axes"][0] / 2),
                int(pupil_detection_result_2d["ellipse"]["axes"][1] / 2),
            ),
            int(pupil_detection_result_2d["ellipse"]["angle"]),
            0,
            360,
            15,
        )
    except ValueError:
        # Happens when converting 'nan' to int
        # TODO: Investigate why results are sometimes 'nan'
        return

    confidence = pupil_detection_result_2d["confidence"] * 0.7
    draw_polyline(pts, 1, RGBA(1.0, 0, 0, confidence))
    draw_points(
        [pupil_detection_result_2d["ellipse"]["center"]],
        size=20,
        color=RGBA(1.0, 0.0, 0.0, confidence),
        sharpness=1.0,
    )
