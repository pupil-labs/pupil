"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import pytest
from raw_data_exporter import Gaze_Positions_Exporter, Pupil_Positions_Exporter


def _test_exporter(exporter, positions, expected_dict_export, world_index=123):
    sample_labels = tuple(exporter.csv_export_labels())
    assert len(sample_labels) > 0, "There must be at least 1 label"
    assert len(sample_labels) == len(set(sample_labels)), "Labels must be unique"

    actual_dict_export = exporter.dict_export(raw_value=positions, world_index=123)
    assert set(sample_labels) == set(
        actual_dict_export.keys()
    ), "Labels must be the keys for the exported dict"
    assert (
        actual_dict_export == expected_dict_export
    ), "Actual pupil export must be the same as expeted export"


def test_pupil_positions_exporter_capture():
    _test_exporter(
        exporter=Pupil_Positions_Exporter(),
        positions=PUPIL_CAPTURE_PUPIL_POSITION_0,
        expected_dict_export=PUPIL_CAPTURE_PUPIL_EXPORT_DICT_0,
    )


def test_pupil_positions_exporter_pi():
    pass  # TODO


def test_gaze_positions_exporter_capture():
    _test_exporter(
        exporter=Gaze_Positions_Exporter(),
        positions=PUPIL_CAPTURE_GAZE_POSITION_0,
        expected_dict_export=PUPIL_CAPTURE_GAZE_EXPORT_DICT_0,
    )


def test_gaze_positions_exporter_pi():
    _test_exporter(
        exporter=Gaze_Positions_Exporter(),
        positions=PUPIL_INVISIBLE_GAZE_POSITION_0,
        expected_dict_export=PUPIL_INVISIBLE_GAZE_EXPORT_DICT_0,
    )


PUPIL_CAPTURE_PUPIL_TIMESTAMP_0 = 18147.38145
PUPIL_CAPTURE_PUPIL_POSITION_0 = {
    "topic": "pupil.0",
    "circle_3d": {
        "center": (-2.589605454357244, 5.16258305418917, 106.22798206853192),
        "normal": (0.021375378493324514, 0.3073983655417788, -0.951340810675391),
        "radius": 2.178582911117089,
    },
    "confidence": 0.9661007130628978,
    "timestamp": 18147.38145,
    "diameter_3d": 4.357165822234178,
    "ellipse": {
        "center": (80.87985171196992, 126.05637442643379),
        "axes": (23.82040062925261, 25.43866813955349),
        "angle": 89.7491053060753,
    },
    "norm_pos": (0.42124922766651, 0.3434563831956573),
    "diameter": 25.43866813955349,
    "sphere": {
        "center": (-2.8461099962771383, 1.4738026676878244, 117.64407179663661),
        "radius": 12.0,
    },
    "projected_sphere": {
        "center": (81.00061948941932, 103.76713726422189),
        "axes": (126.48321137440784, 126.48321137440784),
        "angle": 90.0,
    },
    "model_confidence": 0.8313087355695914,
    "model_id": 14,
    "model_birth_timestamp": 18144.589568,
    "theta": 1.8832541345741016,
    "phi": -1.5483314201277814,
    "method": "3d c++",
    "id": 0,
}
PUPIL_CAPTURE_PUPIL_EXPORT_DICT_0 = {
    "pupil_timestamp": "18147.38145",
    "world_index": 123,
    "eye_id": 0,
    "confidence": 0.9661007130628978,
    "norm_pos_x": 0.42124922766651,
    "norm_pos_y": 0.3434563831956573,
    "diameter": 25.43866813955349,
    "method": "3d c++",
    "ellipse_center_x": 80.87985171196992,
    "ellipse_center_y": 126.05637442643379,
    "ellipse_axis_a": 23.82040062925261,
    "ellipse_axis_b": 25.43866813955349,
    "ellipse_angle": 89.7491053060753,
    "diameter_3d": 4.357165822234178,
    "model_confidence": 0.8313087355695914,
    "model_id": 14,
    "sphere_center_x": -2.8461099962771383,
    "sphere_center_y": 1.4738026676878244,
    "sphere_center_z": 117.64407179663661,
    "sphere_radius": 12.0,
    "circle_3d_center_x": -2.589605454357244,
    "circle_3d_center_y": 5.16258305418917,
    "circle_3d_center_z": 106.22798206853192,
    "circle_3d_normal_x": 0.021375378493324514,
    "circle_3d_normal_y": 0.3073983655417788,
    "circle_3d_normal_z": -0.951340810675391,
    "circle_3d_radius": 2.178582911117089,
    "theta": 1.8832541345741016,
    "phi": -1.5483314201277814,
    "projected_sphere_center_x": 81.00061948941932,
    "projected_sphere_center_y": 103.76713726422189,
    "projected_sphere_axis_a": 126.48321137440784,
    "projected_sphere_axis_b": 126.48321137440784,
    "projected_sphere_angle": 90.0,
}

PUPIL_CAPTURE_GAZE_TIMESTAMP_0 = 18147.383181999998
PUPIL_CAPTURE_GAZE_POSITION_0 = {
    "topic": "gaze.3d.01.",
    "eye_centers_3d": {
        "0": (19.646972429677437, 16.653615316725187, -16.741350196951785),
        "1": (-41.020316707467195, 10.609042518566177, -34.50487956145177),
    },
    "gaze_normals_3d": {
        "0": (-0.02317613920692292, 0.0503770422758375, 0.998461326333173),
        "1": (0.008199922593243814, 0.017997633703220717, 0.9998044040963957),
    },
    "gaze_point_3d": (-26.67606235654398, 87.20744316999192, 2124.3326293817645),
    "confidence": 0.9661007130628978,
    "timestamp": 18147.383181999998,
    "base_data": [
        {
            "topic": "pupil.0",
            "circle_3d": {
                "center": (-2.589605454357244, 5.16258305418917, 106.22798206853192),
                "normal": (
                    0.021375378493324514,
                    0.3073983655417788,
                    -0.951340810675391,
                ),
                "radius": 2.178582911117089,
            },
            "confidence": 0.9661007130628978,
            "timestamp": 18147.38145,
            "diameter_3d": 4.357165822234178,
            "ellipse": {
                "center": (80.87985171196992, 126.05637442643379),
                "axes": (23.82040062925261, 25.43866813955349),
                "angle": 89.7491053060753,
            },
            "norm_pos": (0.42124922766651, 0.3434563831956573),
            "diameter": 25.43866813955349,
            "sphere": {
                "center": (-2.8461099962771383, 1.4738026676878244, 117.64407179663661),
                "radius": 12.0,
            },
            "projected_sphere": {
                "center": (81.00061948941932, 103.76713726422189),
                "axes": (126.48321137440784, 126.48321137440784),
                "angle": 90.0,
            },
            "model_confidence": 0.8313087355695914,
            "model_id": 14,
            "model_birth_timestamp": 18144.589568,
            "theta": 1.8832541345741016,
            "phi": -1.5483314201277814,
            "method": "3d c++",
            "id": 0,
        },
        {
            "topic": "pupil.1",
            "circle_3d": {
                "center": (
                    -1.8241334685531108,
                    -4.1069075704320746,
                    101.41643695709374,
                ),
                "normal": (
                    -0.008068882229597849,
                    -0.30213665999147843,
                    -0.9532304715171239,
                ),
                "radius": 2.0663996742618562,
            },
            "confidence": 0.9668871617583898,
            "timestamp": 18147.384914,
            "diameter_3d": 4.1327993485237124,
            "ellipse": {
                "center": (84.84988578600691, 70.96598316624396),
                "axes": (23.769241864453477, 25.268831954531986),
                "angle": 86.07716997149748,
            },
            "norm_pos": (0.44192648846878596, 0.6303855043424793),
            "diameter": 25.268831954531986,
            "sphere": {
                "center": (-1.7273068817979365, -0.481267650534333, 112.85520261529922),
                "radius": 12.0,
            },
            "projected_sphere": {
                "center": (86.51058132990725, 93.35602846464754),
                "axes": (131.85036803950402, 131.85036803950402),
                "angle": 90.0,
            },
            "model_confidence": 0.600632750497887,
            "model_id": 12,
            "model_birth_timestamp": 18141.421905,
            "theta": 1.2638630532171315,
            "phi": -1.5792609004322697,
            "method": "3d c++",
            "id": 1,
        },
    ],
    "norm_pos": (0.5077714574008085, 0.39313176450355647),
}
PUPIL_CAPTURE_GAZE_EXPORT_DICT_0 = {
    "gaze_timestamp": "18147.383181999998",
    "world_index": 123,
    "confidence": 0.9661007130628978,
    "norm_pos_x": 0.5077714574008085,
    "norm_pos_y": 0.39313176450355647,
    "base_data": "18147.38145-0 18147.384914-1",
    "gaze_point_3d_x": -26.67606235654398,
    "gaze_point_3d_y": 87.20744316999192,
    "gaze_point_3d_z": 2124.3326293817645,
    "eye_center0_3d_x": 19.646972429677437,
    "eye_center0_3d_y": 16.653615316725187,
    "eye_center0_3d_z": -16.741350196951785,
    "gaze_normal0_x": -0.02317613920692292,
    "gaze_normal0_y": 0.0503770422758375,
    "gaze_normal0_z": 0.998461326333173,
    "eye_center1_3d_x": -41.020316707467195,
    "eye_center1_3d_y": 10.609042518566177,
    "eye_center1_3d_z": -34.50487956145177,
    "gaze_normal1_x": 0.008199922593243814,
    "gaze_normal1_y": 0.017997633703220717,
    "gaze_normal1_z": 0.9998044040963957,
}

PUPIL_INVISIBLE_GAZE_TIMESTAMP_0 = 14088010559.20657
PUPIL_INVISIBLE_GAZE_POSITION_0 = {
    "topic": "gaze.pi",
    "norm_pos": (0.49499332203584556, 0.3979458844220197),
    "timestamp": 14088010559.20657,
    "confidence": 1.0,
}
PUPIL_INVISIBLE_GAZE_EXPORT_DICT_0 = {
    "gaze_timestamp": "14088010559.20657",
    "world_index": 123,
    "confidence": 1.0,
    "norm_pos_x": 0.49499332203584556,
    "norm_pos_y": 0.3979458844220197,
    "base_data": None,
    "gaze_point_3d_x": None,
    "gaze_point_3d_y": None,
    "gaze_point_3d_z": None,
    "eye_center0_3d_x": None,
    "eye_center0_3d_y": None,
    "eye_center0_3d_z": None,
    "gaze_normal0_x": None,
    "gaze_normal0_y": None,
    "gaze_normal0_z": None,
    "eye_center1_3d_x": None,
    "eye_center1_3d_y": None,
    "eye_center1_3d_z": None,
    "gaze_normal1_x": None,
    "gaze_normal1_y": None,
    "gaze_normal1_z": None,
}


if __name__ == "__main__":
    # Test pupil/gaze exporter with recording from Pupil Capture
    test_pupil_positions_exporter_capture()
    test_gaze_positions_exporter_capture()
    # Test pupil/gaze exporter with recording from Pupil Invisible
    test_pupil_positions_exporter_pi()
    test_gaze_positions_exporter_pi()
