"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import itertools

import pytest
from surface_tracker.surface_marker import (
    Surface_Marker,
    Surface_Marker_TagID,
    Surface_Marker_Type,
    _Apriltag_V3_Marker_Detection,
    _Square_Marker_Detection,
    create_surface_marker_uid,
    parse_surface_marker_tag_family,
    parse_surface_marker_tag_id,
    parse_surface_marker_type,
)


def test_surface_marker_from_raw_detection():
    # TODO: Test `from_*_detection` methods below
    # Surface_Marker.from_square_tag_detection({})
    # Surface_Marker.from_apriltag_v3_detection(pupil_apriltags.Detection())
    assert True


def test_surface_marker_deserialize():
    # Square tag deserialization test

    SQUARE_MARKER_TAG_ID = 55
    SQUARE_MARKER_CONF = 0.0039215686274509665
    SQUARE_MARKER_VERTS = [
        [[1084.0, 186.0]],
        [[1089.0, 198.0]],
        [[1099.0, 195.0]],
        [[1095.0, 184.0]],
    ]
    SQUARE_MARKER_PERIM = 46.32534599304199

    # This is the format in which old (before Apriltag support was added) square tags where represented when serialized to msgpack
    old_serialized_square = [
        SQUARE_MARKER_TAG_ID,
        SQUARE_MARKER_CONF,
        SQUARE_MARKER_VERTS,
        SQUARE_MARKER_PERIM,
    ]
    # This is the format in which new square tags are represented when serialized to msgpack
    new_serialized_square = [
        [
            SQUARE_MARKER_TAG_ID,
            SQUARE_MARKER_CONF,
            SQUARE_MARKER_VERTS,
            SQUARE_MARKER_PERIM,
            _Square_Marker_Detection.marker_type.value,
        ]
    ]

    # Both formats should be supported by `Surface_Marker.deserialize` for backwards compatibility
    old_marker_square = Surface_Marker.deserialize(old_serialized_square)
    new_marker_square = Surface_Marker.deserialize(new_serialized_square)

    assert old_marker_square.marker_type == Surface_Marker_Type.SQUARE
    assert old_marker_square.tag_id == SQUARE_MARKER_TAG_ID
    assert old_marker_square.id_confidence == SQUARE_MARKER_CONF
    assert old_marker_square.verts_px == SQUARE_MARKER_VERTS
    assert old_marker_square.perimeter == SQUARE_MARKER_PERIM

    assert new_marker_square.marker_type == old_marker_square.marker_type
    assert new_marker_square.tag_id == old_marker_square.tag_id
    assert new_marker_square.id_confidence == old_marker_square.id_confidence
    assert new_marker_square.verts_px == old_marker_square.verts_px
    assert new_marker_square.perimeter == old_marker_square.perimeter

    # Apriltag V3 deserialization test

    APRILTAG_V3_FAMILY = "tag36h11"
    APRILTAG_V3_TAG_ID = 10
    APRILTAG_V3_HAMMING = 2
    APRILTAG_V3_DECISION_MARGIN = 44.26249694824219
    APRILTAG_V3_HOMOGRAPHY = [
        [0.7398546228643903, 0.24224258644348548, -22.823628761428765],
        [-0.14956381555697143, -0.595697080889624, -53.73760032443805],
        [0.00036910994224440203, -0.001201257450114289, -0.07585102600797115],
    ]
    APRILTAG_V3_CENTER = [300.90072557529066, 708.4624052256166]
    APRILTAG_V3_CORNERS = [
        [317.3298034667968, 706.38671875],
        [300.56298828125, 717.4339599609377],
        [284.8282165527345, 710.4930419921874],
        [301.2247619628906, 699.854797363281],
    ]
    APRILTAG_V3_POSE_R = None
    APRILTAG_V3_POSE_T = None
    APRILTAG_V3_POSE_ERR = None

    new_serialized_apriltag_v3 = [
        [
            APRILTAG_V3_FAMILY,
            APRILTAG_V3_TAG_ID,
            APRILTAG_V3_HAMMING,
            APRILTAG_V3_DECISION_MARGIN,
            APRILTAG_V3_HOMOGRAPHY,
            APRILTAG_V3_CENTER,
            APRILTAG_V3_CORNERS,
            APRILTAG_V3_POSE_R,
            APRILTAG_V3_POSE_T,
            APRILTAG_V3_POSE_ERR,
            _Apriltag_V3_Marker_Detection.marker_type.value,
        ]
    ]

    new_marker_apriltag_v3 = Surface_Marker.deserialize(new_serialized_apriltag_v3)

    APRILTAG_V3_CONF = APRILTAG_V3_DECISION_MARGIN / 100
    APRILTAG_V3_VERTS = [[c] for c in APRILTAG_V3_CORNERS]
    APRILTAG_V3_PERIM = 74.20

    assert new_marker_apriltag_v3.marker_type == Surface_Marker_Type.APRILTAG_V3
    assert new_marker_apriltag_v3.tag_id == APRILTAG_V3_TAG_ID
    assert new_marker_apriltag_v3.id_confidence == APRILTAG_V3_CONF
    assert new_marker_apriltag_v3.verts_px == APRILTAG_V3_VERTS
    assert round(new_marker_apriltag_v3.perimeter, 2) == APRILTAG_V3_PERIM


def test_surface_marker_uid_helpers():
    all_marker_types = set(Surface_Marker_Type)
    all_tag_ids = [Surface_Marker_TagID(123)]
    all_tag_families = ["best_tags", None]
    all_combinations = itertools.product(
        all_marker_types, all_tag_families, all_tag_ids
    )

    for marker_type, tag_family, tag_id in all_combinations:
        uid = create_surface_marker_uid(
            marker_type=marker_type, tag_family=tag_family, tag_id=tag_id
        )
        assert len(uid) > 0, "Surface_Marker_UID is not valid"

        assert parse_surface_marker_type(uid=uid) == marker_type
        assert parse_surface_marker_tag_id(uid=uid) == tag_id
        assert parse_surface_marker_tag_family(uid=uid) == tag_family
