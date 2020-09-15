"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
from typing import Dict, Tuple

import cv2
from pyglui.cygl.utils import RGBA, draw_points, draw_polyline

logger = logging.getLogger(__name__)


def draw_ellipse(
    ellipse: Dict, rgba: Tuple, thickness: float, draw_center: bool = False
):
    try:
        pts = cv2.ellipse2Poly(
            center=(int(ellipse["center"][0]), int(ellipse["center"][1])),
            axes=(int(ellipse["axes"][0] / 2), int(ellipse["axes"][1] / 2)),
            angle=int(ellipse["angle"]),
            arcStart=0,
            arcEnd=360,
            delta=8,
        )
    except Exception as e:
        # Known issues:
        #   - There are reports of negative eye_ball axes when drawing the 3D eyeball
        #     outline, which will raise cv2.error. TODO: Investigate cause in detectors.
        #   - There was a case where all values in the ellipse where 'NaN', which raises
        #     ValueError: cannot convert float NaN to integer. TODO: Investigate how we
        #     even got here, since calls to this function are confidence-gated!
        logger.debug(
            "Error drawing ellipse! Skipping...\n"
            f"Ellipse: {ellipse}\n"
            f"Color: {rgba}\n"
            f"Error: {type(e)}: {e}"
        )
        return

    draw_polyline(pts, thickness, RGBA(*rgba))
    if draw_center:
        draw_points(
            [ellipse["center"]],
            size=20,
            color=RGBA(*rgba),
            sharpness=1.0,
        )


def draw_eyeball_outline(pupil_detection_result_3d):
    if pupil_detection_result_3d["model_confidence"] <= 0.0:
        # NOTE: if 'model_confidence' == 0, some values of the 'projected_sphere' might
        # be 'nan', which will cause cv2.ellipse to crash.
        # TODO: Fix in detectors.
        return

    draw_ellipse(
        ellipse=pupil_detection_result_3d["projected_sphere"],
        rgba=(0, 0.9, 0.1, pupil_detection_result_3d["model_confidence"]),
        thickness=2,
    )


def draw_pupil_outline(pupil_detection_result_2d, color_rgb=(1.0, 0.0, 0.0)):
    if pupil_detection_result_2d["confidence"] <= 0.0:
        return

    draw_ellipse(
        ellipse=pupil_detection_result_2d["ellipse"],
        rgba=(*color_rgb, pupil_detection_result_2d["confidence"]),
        thickness=1,
        draw_center=True,
    )
