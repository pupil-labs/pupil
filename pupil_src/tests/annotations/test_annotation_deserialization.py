"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import io
import json

import pytest
from annotations import Annotation_Player


@pytest.fixture
def expected_definitions():
    return (("label 0", "hotkey 0"), ("label 1", "hotkey 1"))


@pytest.fixture
def expected_definitions_json_object(expected_definitions):
    return {
        "version": Annotation_Player._FILE_DEFINITIONS_VERSION,
        "definitions": dict(expected_definitions),
    }


def test_expected_version():
    assert (
        Annotation_Player._FILE_DEFINITIONS_VERSION == 1
    ), "Version changed. Adjust tests accordingly!"


def test_deserialize_definitions_from_file(
    expected_definitions_json_object, expected_definitions
):
    json_buffer = io.StringIO()
    json.dump(expected_definitions_json_object, json_buffer)
    json_buffer.seek(0)

    definitions = Annotation_Player._deserialize_definitions_from_file(
        readable_json_file=json_buffer,
        expected_version=Annotation_Player._FILE_DEFINITIONS_VERSION,
    )
    assert definitions == expected_definitions


def test_deserialize_definitions_from_file_missing_version(
    expected_definitions_json_object,
):
    json_buffer = io.StringIO()
    del expected_definitions_json_object["version"]
    json.dump(expected_definitions_json_object, json_buffer)
    json_buffer.seek(0)

    with pytest.raises(Annotation_Player.VersionMismatchError):
        Annotation_Player._deserialize_definitions_from_file(
            readable_json_file=json_buffer,
            expected_version=Annotation_Player._FILE_DEFINITIONS_VERSION,
        )


def test_deserialize_definitions_from_file_missing_definitions(
    expected_definitions_json_object,
):
    json_buffer = io.StringIO()
    del expected_definitions_json_object["definitions"]
    json.dump(expected_definitions_json_object, json_buffer)
    json_buffer.seek(0)

    with pytest.raises(KeyError):
        Annotation_Player._deserialize_definitions_from_file(
            readable_json_file=json_buffer,
            expected_version=Annotation_Player._FILE_DEFINITIONS_VERSION,
        )


def test_deserialize_definitions_from_file_definitions_not_a_map(
    expected_definitions_json_object,
):
    json_buffer = io.StringIO()
    # should be map, but setting to sequence to trigger AttributeError down the line
    expected_definitions_json_object["definitions"] = []
    json.dump(expected_definitions_json_object, json_buffer)
    json_buffer.seek(0)

    with pytest.raises(AttributeError):
        Annotation_Player._deserialize_definitions_from_file(
            readable_json_file=json_buffer,
            expected_version=Annotation_Player._FILE_DEFINITIONS_VERSION,
        )


def test_deserialize_definitions_from_file_invalid_json(
    expected_definitions_json_object,
):
    json_buffer = io.StringIO()
    json.dump(expected_definitions_json_object, json_buffer)
    json_buffer.seek(1)  # 1 instead of 0, looks like invalid json to parser

    with pytest.raises(json.JSONDecodeError):
        Annotation_Player._deserialize_definitions_from_file(
            readable_json_file=json_buffer,
            expected_version=Annotation_Player._FILE_DEFINITIONS_VERSION,
        )
