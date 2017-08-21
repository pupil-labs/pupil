'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import numpy as np
import scipy.spatial as sp

import OpenGL.GL as gl
from glfw import *

from pyglui import ui
from pyglui.cygl.utils import draw_points_norm, draw_polyline_norm, RGBA

from plugin import Plugin
from calibration_routines.calibrate import closest_matches_monocular

# logging
import logging
logger = logging.getLogger(__name__)


class Accuracy_Visualizer(Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites not between
    """
    order = .8

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.accuracy = None
        self.precision = None
        self.error_lines = None

    def init_gui(self):
        self.menu = ui.Growing_Menu('Accuracy Test')
        self.g_pool.sidebar.append(self.menu)

        def close():
            self.alive = False
        self.menu.append(ui.Button('Close', close))

        general_help = '''Measure gaze mapping accuracy and precision using a screen
                          based animation: After having calibrated on the screen run
                          this test. To compute results set your world cam FOV and
                          click 'calculate results'.'''.replace("\n", " ").replace("  ", '')
        self.menu.append(ui.Info_Text(general_help))

        accuracy_help = '''Accuracy is calculated as the average angular
                        offset (distance) (in degrees of visual angle)
                        between fixations locations and the corresponding
                        locations of the fixation targets.'''.replace("\n", " ").replace("  ", '')

        precision_help = '''Precision is calculated as the Root Mean Square (RMS)
                            of the angular distance (in degrees of visual angle)
                            between successive samples during a fixation.'''.replace("\n", " ").replace("  ", '')

        def ignore():
            pass

        self.menu.append(ui.Info_Text(accuracy_help))
        self.menu.append(ui.Text_Input('accuracy', self, 'Angular Accuracy', setter=ignore,
                         getter=lambda: self.accuracy if self.accuracy is not None else 'Not available'))
        self.menu.append(ui.Info_Text(precision_help))
        self.menu.append(ui.Text_Input('precision', self, 'Angluar Precision', setter=ignore,
                         getter=lambda: self.precision if self.precision is not None else 'Not available'))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def cleanup(self):
        self.deinit_gui()

    def on_notify(self, notification):
        if notification['subject'] == 'calibration.calibration_data':
            input_ = notification['pupil_list']
            labels = notification['ref_list']

            width, height = self.g_pool.capture.frame_size
            prediction = self.g_pool.active_gaze_mapping_plugin.map_batch(input_)

            # reuse closest_matches_monocular to correlate one label to each prediction
            # correlated['ref']: prediction, correlated['pupil']: label location
            correlated = closest_matches_monocular(prediction, labels)
            # [[pred.x, pred.y, label.x, label.y], ...], shape: n x 4
            locations = np.array([(*e['ref']['norm_pos'], *e['pupil']['norm_pos']) for e in correlated])
            self.error_lines = locations.reshape(-1, 2).copy()  # 2n x 2
            locations[:, ::2] *= width
            locations[:, 1::2] = (1. - locations[:, 1::2]) * height

            # Accuracy is calculated as the average angular
            # offset (distance) (in degrees of visual angle)
            # between fixations locations and the corresponding
            # locations of the fixation targets.
            undistorted = self.g_pool.capture.intrinsics.undistortPoints(locations)
            undistorted.shape = -1, 2
            # append column with z=1
            # using idea from https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-an-numpy-array
            undistorted_3d = np.ones((undistorted.shape[0], 3))  # shape: 2n x 3
            undistorted_3d[:, :-1] = undistorted
            # normalize vectors:
            undistorted_3d /= np.linalg.norm(undistorted_3d, axis=1)[:, np.newaxis]

            # vectors already normalized, therefore this is equivalent to the cosinus of the dot product
            # np.einsum('ij,ij->i', X, X) equivalent to np.diagonal(X @ X.T) but faster
            angular_err = np.einsum('ij,ij->i', undistorted_3d[::2, :], undistorted_3d[1::2, :])
            self.accuracy = np.rad2deg(np.arccos(angular_err.mean()))
            logger.info('Angular accuracy: {}'.format(self.accuracy))

            # lets calculate precision:  (RMS of distance of succesive samples.)
            # This is a little rough as we do not compensate headmovements in this test.

            # Precision is calculated as the Root Mean Square (RMS)
            # of the angular distance (in degrees of visual angle)
            # between successive samples during a fixation
            undistorted_3d.shape = -1, 6  # shape: n x 6
            succesive_distances_gaze = sp.distance.pdist(undistorted_3d[:-1, :3] - undistorted_3d[1:, :3])
            succesive_distances_ref = sp.distance.pdist(undistorted_3d[:-1, 3:] - undistorted_3d[1:, 3:])
            # if the ref distance is to big we must have moved to a new fixation or there is headmovement,
            # if the gaze dis is to big we can assume human error
            # both times gaze data is not valid for this mesurement
            succesive_distances = succesive_distances_gaze[np.logical_and(succesive_distances_gaze < 1., succesive_distances_ref < .1)]
            self.precision = np.sqrt(np.mean(succesive_distances ** 2))
            logger.info("Angular precision: {}".format(self.precision))

    def gl_display(self):
        if self.error_lines is not None:
            draw_polyline_norm(self.error_lines, color=RGBA(1., 0.5, 0., .5), line_type=gl.GL_LINES)
            draw_points_norm(self.error_lines[1::2], color=RGBA(.0, 0.5, 0.5, .5), size=3)
            draw_points_norm(self.error_lines[0::2], color=RGBA(.5, 0.0, 0.0, .5), size=3)

    def get_init_dict(self):
        return {}
