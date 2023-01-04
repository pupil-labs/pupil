"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import traceback
import typing as T

import numpy as np
import scipy.spatial
from calibration_choreography import (
    ChoreographyAction,
    ChoreographyMode,
    ChoreographyNotification,
)
from gaze_mapping import gazer_classes_by_class_name, registered_gazer_classes
from gaze_mapping.notifications import (
    CalibrationResultNotification,
    CalibrationSetupNotification,
)
from gaze_mapping.utils import closest_matches_monocular
from plugin import Plugin

logger = logging.getLogger(__name__)


class CalculationResult(T.NamedTuple):
    result: float
    num_used: int
    num_total: int


class CorrelatedAndCoordinateTransformedResult(T.NamedTuple):
    """Holds result from correlating reference and gaze data and their respective
    transformations into norm, image, and camera coordinate systems.
    """

    norm_space: np.ndarray  # shape: 2*n, 2
    image_space: np.ndarray  # shape: 2*n, 2
    camera_space: np.ndarray  # shape: 2*n, 3

    @staticmethod
    def empty() -> "CorrelatedAndCoordinateTransformedResult":
        return CorrelatedAndCoordinateTransformedResult(
            norm_space=np.ndarray([]),
            image_space=np.ndarray([]),
            camera_space=np.ndarray([]),
        )

    @property
    def is_valid(self) -> bool:
        if len(self.norm_space.shape) != 2:
            return False
        # TODO: Make validity check exhaustive
        return True


class CorrelationError(ValueError):
    pass


class AccuracyPrecisionResult(T.NamedTuple):
    accuracy: CalculationResult
    precision: CalculationResult
    error_lines: np.ndarray
    correlation: CorrelatedAndCoordinateTransformedResult

    @staticmethod
    def failed() -> "AccuracyPrecisionResult":
        return AccuracyPrecisionResult(
            accuracy=CalculationResult(0.0, 0, 0),
            precision=CalculationResult(0.0, 0, 0),
            error_lines=np.array([]),
            correlation=CorrelatedAndCoordinateTransformedResult.empty(),
        )

    @property
    def is_valid(self) -> bool:
        if not self.correlation.is_valid:
            return False
        # TODO: Make validity check exhaustive
        return True


class ValidationInput:
    def __init__(self):
        self.clear()

    @property
    def gazer_class(self) -> T.Optional[T.Any]:
        return self.__gazer_class

    @property
    def gazer_params(self) -> T.Optional[T.Any]:
        return self.__gazer_params

    @property
    def gazer_class_name(self) -> T.Optional[str]:
        return self.__gazer_class.__name__ if self.__gazer_class is not None else None

    @property
    def pupil_list(self) -> T.Optional[T.Any]:
        return self.__pupil_list

    @property
    def ref_list(self) -> T.Optional[T.Any]:
        return self.__ref_list

    @property
    def is_complete(self) -> bool:
        return None not in (
            self.pupil_list,
            self.ref_list,
            self.gazer_class,
            self.gazer_params,
        )

    def clear(self):
        self.__pupil_list = None
        self.__ref_list = None
        self.__gazer_class = None
        self.__gazer_params = None

    def update(
        self, gazer_class_name: str, gazer_params=..., pupil_list=..., ref_list=...
    ):
        if (
            self.gazer_class_name is not None
            and self.gazer_class_name != gazer_class_name
        ):
            logger.debug(
                f'Overwriting gazer_class_name from "{self.gazer_class_name}" to '
                f'"{gazer_class_name}" and resetting the input.'
            )
            self.clear()

        self.__gazer_class = self.__gazer_class_from_name(gazer_class_name)

        if gazer_params is not ...:
            self.__gazer_params = gazer_params

        if pupil_list is not ...:
            self.__pupil_list = pupil_list

        if ref_list is not ...:
            self.__ref_list = ref_list

    @staticmethod
    def __gazer_class_from_name(gazer_class_name: str) -> T.Optional[T.Any]:
        gazers_by_name = gazer_classes_by_class_name(registered_gazer_classes())

        try:
            gazer_cls = gazers_by_name[gazer_class_name]
        except KeyError:
            logger.error(f'Unknown gazer "{gazer_class_name}"')
            return None

        return gazer_cls


class Accuracy_Visualizer(Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites not between
    """

    order = 0.8
    icon_chr = chr(0xEC11)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        outlier_threshold=5.0,
        vis_mapping_error=True,
        vis_calibration_area=True,
    ):
        super().__init__(g_pool)
        self.vis_mapping_error = vis_mapping_error
        self.vis_calibration_area = vis_calibration_area
        self.calibration_area = None
        self.accuracy = None
        self.precision = None
        self.error_lines = None

        self.recent_input = ValidationInput()

        # .5 degrees, used to remove outliers from precision calculation
        self.succession_threshold = np.cos(np.deg2rad(0.5))
        self._outlier_threshold = outlier_threshold  # in degrees

    def init_ui(self):
        from pyglui import ui

        self.add_menu()
        self.menu.label = "Accuracy Visualizer"

        mapping_error_help = """The mapping error (orange line) is the angular
                             distance between mapped pupil positions (red) and
                             their corresponding reference points (blue).
                             """.replace(
            "\n", " "
        ).replace(
            "  ", ""
        )

        calib_area_help = """The calibration area (green) is defined as the
                          convex hull of the reference points that were used
                          for calibration. 2D mapping looses accuracy outside
                          of this area. It is recommended to calibrate a big
                          portion of the subject's field of view.
                          """.replace(
            "\n", " "
        ).replace(
            "  ", ""
        )
        self.menu.append(ui.Info_Text(calib_area_help))
        self.menu.append(
            ui.Switch("vis_mapping_error", self, label="Visualize mapping error")
        )

        self.menu.append(ui.Info_Text(mapping_error_help))
        self.menu.append(
            ui.Switch("vis_calibration_area", self, label="Visualize calibration area")
        )

        general_help = """Measure gaze mapping accuracy and precision using samples
                          that were collected during calibration. The outlier threshold
                          discards samples with high angular errors.""".replace(
            "\n", " "
        ).replace(
            "  ", ""
        )
        self.menu.append(ui.Info_Text(general_help))

        # self.menu.append(ui.Info_Text(''))
        self.menu.append(
            ui.Text_Input(
                "outlier_threshold", self, label="Outlier Threshold [degrees]"
            )
        )

        accuracy_help = """Accuracy is calculated as the average angular
                        offset (distance) (in degrees of visual angle)
                        between fixation locations and the corresponding
                        locations of the fixation targets.""".replace(
            "\n", " "
        ).replace(
            "  ", ""
        )

        precision_help = """Precision is calculated as the Root Mean Square (RMS)
                            of the angular distance (in degrees of visual angle)
                            between successive samples during a fixation.""".replace(
            "\n", " "
        ).replace(
            "  ", ""
        )

        def ignore(_):
            pass

        self.menu.append(ui.Info_Text(accuracy_help))
        self.menu.append(
            ui.Text_Input(
                "accuracy",
                self,
                "Angular Accuracy",
                setter=ignore,
                getter=lambda: f"{self.accuracy.result:.3f} deg. Samples used: "
                f"{self.accuracy.num_used} / {self.accuracy.num_total}"
                if self.accuracy is not None
                else "Not available",
            )
        )
        self.menu.append(ui.Info_Text(precision_help))
        self.menu.append(
            ui.Text_Input(
                "precision",
                self,
                "Angular Precision",
                setter=ignore,
                getter=lambda: f"{self.precision.result:.3f} deg. Samples used: "
                f"{self.precision.num_used} / {self.precision.num_total}"
                if self.precision is not None
                else "Not available",
            )
        )

    def deinit_ui(self):
        self.remove_menu()

    @property
    def outlier_threshold(self):
        return self._outlier_threshold

    @outlier_threshold.setter
    def outlier_threshold(self, value):
        self._outlier_threshold = value
        self.notify_all(
            {"subject": "accuracy_visualizer.outlier_threshold_changed", "delay": 0.5}
        )

    def on_notify(self, notification):
        if self.__handle_calibration_setup_notification(notification):
            return

        if self.__handle_calibration_result_notification(notification):
            return

        if self.__handle_validation_data_notification(notification):
            return

        if notification["subject"] == "accuracy_visualizer.outlier_threshold_changed":
            if self.recent_input.is_complete:
                self.recalculate()

    def __handle_calibration_setup_notification(self, note_dict: dict) -> bool:
        try:
            note = CalibrationSetupNotification.from_dict(note_dict)
        except ValueError:
            return False

        self.recent_input.update(
            gazer_class_name=note.gazer_class_name,
            pupil_list=note.calib_data["pupil_list"],
            ref_list=note.calib_data["ref_list"],
        )
        return True

    def __handle_calibration_result_notification(self, note_dict: dict) -> bool:
        try:
            note = CalibrationResultNotification.from_dict(note_dict)
        except ValueError:
            return False

        self.recent_input.update(
            gazer_class_name=note.gazer_class_name,
            gazer_params=note.params,
        )

        self.recalculate()
        return True

    def __handle_validation_data_notification(self, note_dict: dict) -> bool:
        try:
            note = ChoreographyNotification.from_dict(note_dict)
            assert note.mode == ChoreographyMode.VALIDATION
            assert note.action == ChoreographyAction.DATA
        except (AssertionError, ValueError):
            return False

        self.recent_input.clear()
        self.recent_input.update(
            gazer_class_name=note_dict["gazer_class_name"],
            gazer_params=note_dict["gazer_params"],
            pupil_list=note_dict["pupil_list"],
            ref_list=note_dict["ref_list"],
        )

        self.recalculate()
        return True

    def recalculate(self):
        NOT_ENOUGH_DATA_COLLECTED_ERR_MSG = (
            "Did not collect enough data to estimate gaze mapping accuracy."
        )

        if not self.recent_input.is_complete:
            logger.warning(NOT_ENOUGH_DATA_COLLECTED_ERR_MSG)
            return

        results = self.calc_acc_prec_errlines(
            gazer_class=self.recent_input.gazer_class,
            g_pool=self.g_pool,
            gazer_params=self.recent_input.gazer_params,
            pupil_list=self.recent_input.pupil_list,
            ref_list=self.recent_input.ref_list,
            intrinsics=self.g_pool.capture.intrinsics,
            outlier_threshold=self.outlier_threshold,
            succession_threshold=self.succession_threshold,
        )

        if not results.is_valid:
            logger.warning(NOT_ENOUGH_DATA_COLLECTED_ERR_MSG)
            return

        if np.isnan(results.accuracy.result):
            self.accuracy = None
            logger.warning(
                "Not enough data available for angular accuracy calculation."
            )
        else:
            self.accuracy = results.accuracy
            logger.info(f"Angular accuracy: {results.accuracy.result:.3f} degrees")

        if np.isnan(results.precision.result):
            self.precision = None
            logger.warning(
                "Not enough data available for angular precision calculation."
            )
        else:
            self.precision = results.precision
            logger.info(f"Angular precision: {results.precision.result:.3f} degrees")

        self.error_lines = results.error_lines
        ref_locations = results.correlation.norm_space[1::2, :]
        if len(ref_locations) >= 3:
            try:
                # requires at least 3 points
                hull = scipy.spatial.ConvexHull(ref_locations)
                self.calibration_area = hull.points[hull.vertices, :]
            except scipy.spatial.qhull.QhullError:
                logger.warning("Calibration area could not be calculated")
                logger.debug(traceback.format_exc())

    @staticmethod
    def calc_acc_prec_errlines(
        g_pool,
        gazer_class,
        gazer_params,
        pupil_list,
        ref_list,
        intrinsics,
        outlier_threshold,
        succession_threshold=np.cos(np.deg2rad(0.5)),
    ) -> AccuracyPrecisionResult:
        gazer = gazer_class(g_pool, params=gazer_params)

        gaze_pos = gazer.map_pupil_to_gaze(pupil_list)
        ref_pos = ref_list

        try:
            correlation_result = Accuracy_Visualizer.correlate_and_coordinate_transform(
                gaze_pos, ref_pos, intrinsics
            )
            error_lines = correlation_result.norm_space.reshape(-1, 4)
            undistorted_3d = correlation_result.camera_space
        except CorrelationError:
            return AccuracyPrecisionResult.failed()

        # Accuracy is calculated as the average angular
        # offset (distance) (in degrees of visual angle)
        # between fixations locations and the corresponding
        # locations of the fixation targets.

        # Cosine distance of A and B: (A @ B) / (||A|| * ||B||)
        # No need to calculate norms, since A and B are normalized in our case.
        # np.einsum('ij,ij->i', A, B) equivalent to np.diagonal(A @ B.T) but faster.
        angular_err = np.einsum(
            "ij,ij->i", undistorted_3d[::2, :], undistorted_3d[1::2, :]
        )

        # Good values are close to 1. since cos(0) == 1.
        # Therefore we look for values greater than cos(outlier_threshold)
        selected_indices = angular_err > np.cos(np.deg2rad(outlier_threshold))
        selected_samples = angular_err[selected_indices]
        num_used, num_total = selected_samples.shape[0], angular_err.shape[0]

        error_lines = error_lines[selected_indices].reshape(
            -1, 2
        )  # shape: num_used x 2
        accuracy = np.rad2deg(np.arccos(selected_samples.clip(-1.0, 1.0).mean()))
        accuracy_result = CalculationResult(accuracy, num_used, num_total)

        # lets calculate precision:  (RMS of distance of succesive samples.)
        # This is a little rough as we do not compensate headmovements in this test.

        # Precision is calculated as the Root Mean Square (RMS)
        # of the angular distance (in degrees of visual angle)
        # between successive samples during a fixation
        undistorted_3d.shape = -1, 6  # shape: n x 6
        succesive_distances_gaze = np.einsum(
            "ij,ij->i", undistorted_3d[:-1, :3], undistorted_3d[1:, :3]
        )
        succesive_distances_ref = np.einsum(
            "ij,ij->i", undistorted_3d[:-1, 3:], undistorted_3d[1:, 3:]
        )

        # if the ref distance is to big we must have moved to a new fixation or there is
        # headmovement, if the gaze dis is to big we can assume human error
        # both times gaze data is not valid for this mesurement
        selected_indices = np.logical_and(
            succesive_distances_gaze > succession_threshold,
            succesive_distances_ref > succession_threshold,
        )
        succesive_distances = succesive_distances_gaze[selected_indices]
        num_used, num_total = (
            succesive_distances.shape[0],
            succesive_distances_gaze.shape[0],
        )
        precision = np.sqrt(
            np.mean(np.rad2deg(np.arccos(succesive_distances.clip(-1.0, 1.0))) ** 2)
        )
        precision_result = CalculationResult(precision, num_used, num_total)

        return AccuracyPrecisionResult(
            accuracy_result, precision_result, error_lines, correlation_result
        )

    @staticmethod
    def correlate_and_coordinate_transform(
        gaze_pos, ref_pos, intrinsics
    ) -> CorrelatedAndCoordinateTransformedResult:
        # reuse closest_matches_monocular to correlate one label to each prediction
        # correlated['ref']: prediction, correlated['pupil']: label location
        # NOTE the switch of the ref and pupil keys! This effects mostly hmd data.
        correlated = closest_matches_monocular(gaze_pos, ref_pos)
        # [[pred.x, pred.y, label.x, label.y], ...], shape: n x 4
        if not correlated:
            raise CorrelationError("No correlation possible")

        try:
            return Accuracy_Visualizer._coordinate_transform_ref_in_norm_space(
                correlated, intrinsics
            )
        except KeyError as err:
            if "norm_pos" in err.args:
                return Accuracy_Visualizer._coordinate_transform_ref_in_camera_space(
                    correlated, intrinsics
                )
            else:
                raise

    @staticmethod
    def _coordinate_transform_ref_in_norm_space(
        correlated, intrinsics
    ) -> CorrelatedAndCoordinateTransformedResult:
        width, height = intrinsics.resolution
        locations_norm = np.array(
            [(*e["ref"]["norm_pos"], *e["pupil"]["norm_pos"]) for e in correlated]
        )
        locations_image = locations_norm.copy()  # n x 4
        locations_image[:, ::2] *= width
        locations_image[:, 1::2] = (1.0 - locations_image[:, 1::2]) * height
        locations_image.shape = -1, 2
        locations_norm.shape = -1, 2
        locations_camera = intrinsics.unprojectPoints(locations_image, normalize=True)
        return CorrelatedAndCoordinateTransformedResult(
            locations_norm, locations_image, locations_camera
        )

    @staticmethod
    def _coordinate_transform_ref_in_camera_space(
        correlated, intrinsics
    ) -> CorrelatedAndCoordinateTransformedResult:
        width, height = intrinsics.resolution
        locations_mixed = np.array(
            # NOTE: This looks incorrect, but is actually correct. The switch comes from
            # using closest_matches_monocular() above with switched arguments.
            [(*e["ref"]["norm_pos"], *e["pupil"]["mm_pos"]) for e in correlated]
        )  # n x 5
        pupil_norm = locations_mixed[:, 0:2]  # n x 2
        pupil_image = pupil_norm.copy()
        pupil_image[:, 0] *= width
        pupil_image[:, 1] = (1.0 - pupil_image[:, 1]) * height
        pupil_camera = intrinsics.unprojectPoints(pupil_image, normalize=True)  # n x 3

        ref_camera = locations_mixed[:, 2:5]  # n x 3
        ref_camera /= np.linalg.norm(ref_camera, axis=1, keepdims=True)
        ref_image = intrinsics.projectPoints(ref_camera)  # n x 2
        ref_norm = ref_image.copy()
        ref_norm[:, 0] /= width
        ref_norm[:, 1] = 1.0 - (ref_norm[:, 1] / height)

        locations_norm = np.hstack([pupil_norm, ref_norm])  # n x 4
        locations_norm.shape = -1, 2

        locations_image = np.hstack([pupil_image, ref_image])  # n x 4
        locations_image.shape = -1, 2

        locations_camera = np.hstack([pupil_camera, ref_camera])  # n x 6
        locations_camera.shape = -1, 3

        return CorrelatedAndCoordinateTransformedResult(
            locations_norm, locations_image, locations_camera
        )

    def gl_display(self):
        import OpenGL.GL as gl
        from pyglui.cygl.utils import RGBA, draw_points_norm, draw_polyline_norm

        if self.vis_mapping_error and self.error_lines is not None:
            draw_polyline_norm(
                self.error_lines, color=RGBA(1.0, 0.5, 0.0, 0.5), line_type=gl.GL_LINES
            )
            draw_points_norm(
                self.error_lines[1::2], size=3, color=RGBA(0.0, 0.5, 0.5, 0.5)
            )
            draw_points_norm(
                self.error_lines[0::2], size=3, color=RGBA(0.5, 0.0, 0.0, 0.5)
            )
        if self.vis_calibration_area and self.calibration_area is not None:
            draw_polyline_norm(
                self.calibration_area,
                thickness=2.0,
                color=RGBA(0.663, 0.863, 0.463, 0.8),
                line_type=gl.GL_LINE_LOOP,
            )

    def get_init_dict(self):
        return {
            "outlier_threshold": self.outlier_threshold,
            "vis_mapping_error": self.vis_mapping_error,
            "vis_calibration_area": self.vis_calibration_area,
        }
