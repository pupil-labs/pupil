'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import numpy as np
from scipy.spatial import ConvexHull

import OpenGL.GL as gl
from glfw import *

from pyglui import ui
from pyglui.cygl.utils import draw_points_norm, draw_polyline_norm, RGBA

from plugin import Plugin
from calibration_routines.calibrate import closest_matches_monocular

from collections import namedtuple

# logging
import logging
logger = logging.getLogger(__name__)

Calculation_Result = namedtuple('Calculation_Result', ['result', 'num_used', 'num_total'])


class Accuracy_Visualizer(Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites not between
    """
    order = .8
    icon_chr = chr(0xec11)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool, outlier_threshold=5.,
                 vis_mapping_error=True, vis_calibration_area=True):
        super().__init__(g_pool)
        self.vis_mapping_error = vis_mapping_error
        self.vis_calibration_area = vis_calibration_area
        self.calibration_area = None
        self.accuracy = None
        self.precision = None
        self.error_lines = None
        self.recent_input = None
        self.recent_labels = None
        # .5 degrees, used to remove outliers from precision calculation
        self.succession_threshold = np.cos(np.deg2rad(.5))
        self._outlier_threshold = outlier_threshold  # in degrees

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Accuracy Visualizer'

        mapping_error_help = '''The mapping error (orange line) is the angular
                             distance between mapped pupil positions (red) and
                             their corresponding reference points (blue).
                             '''.replace("\n", " ").replace("  ", '')

        calib_area_help = '''The calibration area (green) is defined as the
                          convex hull of the reference points that were used
                          for calibration. 2D mapping looses accuracy outside
                          of this area. It is recommended to calibrate a big
                          portion of the subject's field of view.
                          '''.replace("\n", " ").replace("  ", '')
        self.menu.append(ui.Info_Text(calib_area_help))
        self.menu.append(ui.Switch('vis_mapping_error', self,
                                   label='Visualize mapping error'))


        self.menu.append(ui.Info_Text(mapping_error_help))
        self.menu.append(ui.Switch('vis_calibration_area', self,
                                   label='Visualize calibration area'))

        general_help = '''Measure gaze mapping accuracy and precision using samples
                          that were collected during calibration. The outlier threshold
                          discards samples with high angular errors.'''.replace("\n", " ").replace("  ", '')
        self.menu.append(ui.Info_Text(general_help))

        # self.menu.append(ui.Info_Text(''))
        self.menu.append(ui.Text_Input('outlier_threshold', self, label='Outlier Threshold [degrees]'))

        accuracy_help = '''Accuracy is calculated as the average angular
                        offset (distance) (in degrees of visual angle)
                        between fixation locations and the corresponding
                        locations of the fixation targets.'''.replace("\n", " ").replace("  ", '')

        precision_help = '''Precision is calculated as the Root Mean Square (RMS)
                            of the angular distance (in degrees of visual angle)
                            between successive samples during a fixation.'''.replace("\n", " ").replace("  ", '')

        def ignore(_):
            pass

        self.menu.append(ui.Info_Text(accuracy_help))
        self.menu.append(ui.Text_Input('accuracy', self,
                                       'Angular Accuracy',
                                       setter=ignore,
                                       getter=lambda: self.accuracy if self.accuracy is not None else 'Not available'))
        self.menu.append(ui.Info_Text(precision_help))
        self.menu.append(ui.Text_Input('precision', self,
                                       'Angular Precision',
                                       setter=ignore,
                                       getter=lambda: self.precision if self.precision is not None else 'Not available'))

    def deinit_ui(self):
        self.remove_menu()

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
            if 'hmd' in notification.get('calibration_method', ''):
                logger.error('Accuracy visualization is disabled for 3d hmd calibration')
                return
            self.recent_input = notification['pupil_list']
            self.recent_labels = notification['ref_list']
            if self.recent_input and self.recent_labels:
                self.recalculate()
            else:
                logger.error('Did not collect enough data to estimate gaze mapping accuracy.')

        elif notification['subject'] == 'accuracy_visualizer.outlier_threshold_changed':
            if self.recent_input and self.recent_labels:
                self.recalculate()
            else:
                pass

    def recalculate(self):
        assert self.recent_input and self.recent_labels
        prediction = self.g_pool.active_gaze_mapping_plugin.map_batch(self.recent_input)
        results = self.calc_acc_prec_errlines(prediction, self.recent_labels,
                                              self.g_pool.capture.intrinsics)
        logger.info('Angular accuracy: {}. Used {} of {} samples.'.format(*results[0]))
        logger.info("Angular precision: {}. Used {} of {} samples.".format(*results[1]))
        self.accuracy = results[0].result
        self.precision = results[1].result
        self.error_lines = results[2]

        hull = ConvexHull([loc['norm_pos'] for loc in self.recent_labels])
        self.calibration_area = hull.points[hull.vertices, :]

    def calc_acc_prec_errlines(self, gaze_pos, ref_pos, intrinsics):
        width, height = intrinsics.resolution

        # reuse closest_matches_monocular to correlate one label to each prediction
        # correlated['ref']: prediction, correlated['pupil']: label location
        correlated = closest_matches_monocular(gaze_pos, ref_pos)
        # [[pred.x, pred.y, label.x, label.y], ...], shape: n x 4
        locations = np.array([(*e['ref']['norm_pos'], *e['pupil']['norm_pos']) for e in correlated])
        error_lines = locations.copy()  # n x 4
        locations[:, ::2] *= width
        locations[:, 1::2] = (1. - locations[:, 1::2]) * height
        locations.shape = -1, 2

        # Accuracy is calculated as the average angular
        # offset (distance) (in degrees of visual angle)
        # between fixations locations and the corresponding
        # locations of the fixation targets.
        undistorted_3d = intrinsics.unprojectPoints(locations, normalize=True)

        # Cosine distance of A and B: (A @ B) / (||A|| * ||B||)
        # No need to calculate norms, since A and B are normalized in our case.
        # np.einsum('ij,ij->i', A, B) equivalent to np.diagonal(A @ B.T) but faster.
        angular_err = np.einsum('ij,ij->i', undistorted_3d[::2, :], undistorted_3d[1::2, :])

        # Good values are close to 1. since cos(0) == 1.
        # Therefore we look for values greater than cos(outlier_threshold)
        selected_indices = angular_err > np.cos(np.deg2rad(self.outlier_threshold))
        selected_samples = angular_err[selected_indices]
        num_used, num_total = selected_samples.shape[0], angular_err.shape[0]

        error_lines = error_lines[selected_indices].reshape(-1, 2)  # shape: num_used x 2
        accuracy = np.rad2deg(np.arccos(selected_samples.clip(-1., 1.).mean()))
        accuracy_result = Calculation_Result(accuracy, num_used, num_total)

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
        precision = np.sqrt(np.mean(np.rad2deg(np.arccos(succesive_distances.clip(-1., 1.))) ** 2))
        precision_result = Calculation_Result(precision, num_used, num_total)

        return accuracy_result, precision_result, error_lines

    def gl_display(self):
        if self.vis_mapping_error and self.error_lines is not None:
            draw_polyline_norm(self.error_lines,
                               color=RGBA(1., 0.5, 0., .5),
                               line_type=gl.GL_LINES)
            draw_points_norm(self.error_lines[1::2], size=3,
                             color=RGBA(.0, 0.5, 0.5, .5))
            draw_points_norm(self.error_lines[0::2], size=3,
                             color=RGBA(.5, 0.0, 0.0, .5))
        if self.vis_calibration_area and self.calibration_area is not None:
            draw_polyline_norm(self.calibration_area, thickness=2.,
                               color=RGBA(.663, .863, .463, .8),
                               line_type=gl.GL_LINE_LOOP)

    def get_init_dict(self):
        return {'outlier_threshold': self.outlier_threshold,
                'vis_mapping_error': self.vis_mapping_error,
                'vis_calibration_area': self.vis_calibration_area}
