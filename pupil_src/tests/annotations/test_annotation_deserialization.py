import io
import pytest

from annotations import Annotation_Player


@pytest.fixture
def expected_definitions():
    return (("label 0", "hotkey 0"), ("label 1", "hotkey 1"))


@pytest.fixture
def serialized_definitions_valid(expected_definitions):
    json_buffer = io.StringIO()
    Annotation_Player._serialize_definitions_to_file(
        writable_file=json_buffer,
        definitions=expected_definitions,
        version=Annotation_Player._FILE_DEFINITIONS_VERSION,
    )
    json_buffer.seek(0)
    return json_buffer


def test_expected_version():
    assert (
        Annotation_Player._FILE_DEFINITIONS_VERSION == 1
    ), "Version changed. Adjust tests accordingly!"


def test_deserialize_definitions_from_file(
    serialized_definitions_valid, expected_definitions
):
    definitions = Annotation_Player._deserialize_definitions_from_file(
        readable_json_file=serialized_definitions_valid,
        expected_version=Annotation_Player._FILE_DEFINITIONS_VERSION,
    )
    assert definitions == expected_definitions
