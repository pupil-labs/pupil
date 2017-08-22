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

    def __init__(self, g_pool, outlier_threshold=5.):
        super().__init__(g_pool)
        self.accuracy = None
        self.precision = None
        self.error_lines = None
        # .5 degrees, used to remove outliers from precision calculation
        self.succession_threshold = np.cos(np.deg2rad(.5))
        self._outlier_threshold = outlier_threshold  # in degrees

    def init_gui(self):
        self.menu = ui.Growing_Menu('Accuracy Test')
        self.g_pool.sidebar.append(self.menu)

        def close():
            self.alive = False
        self.menu.append(ui.Button('Close', close))

        general_help = '''Measure gaze mapping accuracy and precision using samples
                          that were collected during calibration. The outlier threshold
                          discards samples with high angular errors.'''.replace("\n", " ").replace("  ", '')
        self.menu.append(ui.Info_Text(general_help))

        # self.menu.append(ui.Info_Text(''))
        self.menu.append(ui.Text_Input('outlier_threshold', self, label='Outlier Threshold [degrees]'))

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

    @property
    def outlier_threshold(self):
        return self._outlier_threshold

    @outlier_threshold.setter
    def outlier_threshold(self, value):
        self._outlier_threshold = value
        self.notify_all({'subject': 'accuracy_visualizer.outlier_threshold_changed',
                         'delay': .5})

    def on_notify(self, notification):
        if notification['subject'] in ('calibration.calibration_data', 'accuracy_test.data'):
            self.recent_input = notification['pupil_list']
            self.recent_labels = notification['ref_list']
            self.recalculate()
        elif notification['subject'] == 'accuracy_visualizer.outlier_threshold_changed':
            self.recalculate()

    def recalculate(self):
        width, height = self.g_pool.capture.frame_size
        prediction = self.g_pool.active_gaze_mapping_plugin.map_batch(self.recent_input)

        # reuse closest_matches_monocular to correlate one label to each prediction
        # correlated['ref']: prediction, correlated['pupil']: label location
        correlated = closest_matches_monocular(prediction, self.recent_labels)
        # [[pred.x, pred.y, label.x, label.y], ...], shape: n x 4
        locations = np.array([(*e['ref']['norm_pos'], *e['pupil']['norm_pos']) for e in correlated])
        self.error_lines = locations.copy()  # n x 4
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

        # Cosine distance of A and B: (A @ B) / (||A|| * ||B||)
        # No need to calculate norms, since A and B are normalized in our case.
        # np.einsum('ij,ij->i', A, B) equivalent to np.diagonal(A @ B.T) but faster.
        angular_err = np.einsum('ij,ij->i', undistorted_3d[::2, :], undistorted_3d[1::2, :])

        # Good values are close to 1. since cos(0) == 1.
        # Therefore we look for values greater than cos(outlier_threshold)
        selected_indices = angular_err > np.cos(np.deg2rad(self.outlier_threshold))
        selected_samples = angular_err[selected_indices]
        num_used, num_total = selected_samples.shape[0], angular_err.shape[0]

        self.error_lines = self.error_lines[selected_indices].reshape(-1, 2)  # shape: num_used x 2
        self.accuracy = np.rad2deg(np.arccos(selected_samples.mean()))
        logger.info('Angular accuracy: {}. Used {} of {} samples.'.format(self.accuracy, num_used, num_total))

        # lets calculate precision:  (RMS of distance of succesive samples.)
        # This is a little rough as we do not compensate headmovements in this test.

        # Precision is calculated as the Root Mean Square (RMS)
        # of the angular distance (in degrees of visual angle)
        # between successive samples during a fixation
        undistorted_3d.shape = -1, 6  # shape: n x 6
        succesive_distances_gaze = np.einsum('ij,ij->i', undistorted_3d[:-1, :3], undistorted_3d[1:, :3])
        succesive_distances_ref = np.einsum('ij,ij->i', undistorted_3d[:-1, 3:], undistorted_3d[1:, 3:])

        # if the ref distance is to big we must have moved to a new fixation or there is headmovement,
        # if the gaze dis is to big we can assume human error
        # both times gaze data is not valid for this mesurement
        selected_indices = np.logical_and(succesive_distances_gaze > self.succession_threshold,
                                          succesive_distances_ref > self.succession_threshold)
        succesive_distances = succesive_distances_gaze[selected_indices]
        num_used, num_total = succesive_distances.shape[0], succesive_distances_gaze.shape[0]
        self.precision = np.sqrt(np.mean(np.arccos(succesive_distances) ** 2))
        logger.info("Angular precision: {}. Used {} of {} samples.".format(self.precision, num_used, num_total))

    def gl_display(self):
        if self.error_lines is not None:
            draw_polyline_norm(self.error_lines, color=RGBA(1., 0.5, 0., .5), line_type=gl.GL_LINES)
            draw_points_norm(self.error_lines[1::2], color=RGBA(.0, 0.5, 0.5, .5), size=3)
            draw_points_norm(self.error_lines[0::2], color=RGBA(.5, 0.0, 0.0, .5), size=3)

    def get_init_dict(self):
        return {'outlier_threshold': self.outlier_threshold}
