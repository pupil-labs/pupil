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
from gaze_mapping.notifications import (
    CalibrationFailureNotification,
    CalibrationResultNotification,
    CalibrationSetupNotification,
    CalibrationSuccessNotification,
)


def test_success_notification_serialization():
    cls = CalibrationSuccessNotification

    serialized_sample_1 = {
        "subject": "calibration.successful",
        "gazer_class_name": "FooGazer",
        "timestamp": 123.456,
        "record": True,
    }
    serialized_sample_2 = {
        "subject": "calibration.successful",
        "gazer_class_name": "BazGazer",
        "timestamp": 123.456,
    }

    _test_notification_serialization(
        cls=cls,
        valid_samples=[
            serialized_sample_1,
            serialized_sample_2,
        ],
        invalid_samples=[
            {
                "subject": "calibration.successful.xyz",
                "gazer_class_name": "BazGazer",
                "timestamp": 123.456,
            },
            # TODO: Add samples with type missmatch
        ],
    )

    deserialized_sample_1 = cls.from_dict(serialized_sample_1)

    assert deserialized_sample_1.subject == "calibration.successful"
    assert deserialized_sample_1.gazer_class_name == "FooGazer"
    assert deserialized_sample_1.timestamp == 123.456
    assert deserialized_sample_1.record == True

    deserialized_sample_2 = cls.from_dict(serialized_sample_2)

    assert deserialized_sample_2.subject == "calibration.successful"
    assert deserialized_sample_2.gazer_class_name == "BazGazer"
    assert deserialized_sample_2.timestamp == 123.456
    assert deserialized_sample_2.record == False  # default


def test_failure_notification_serialization():
    pass  # TODO: Test CalibrationFailureNotification


def test_setup_notification_serialization():
    pass  # TODO: Test CalibrationSetupNotification


def test_result_notification_serialization():
    pass  # TODO: Test CalibrationResultNotification


### Private helpers


def _test_notification_serialization(cls, valid_samples, invalid_samples):
    deserialized_notifications = []

    # Test deserialization successful for valid samples
    for serialized in valid_samples:
        deserialized = cls.from_dict(serialized)
        deserialized_notifications.append(deserialized)

    # Test deserialization unsuccessful for invalid samples
    for serialized in invalid_samples:
        with pytest.raises(ValueError):
            _ = cls.from_dict(serialized)

    serialized_notifications = []

    # Test serialization successful for valid deserialized samples
    for deserialized in deserialized_notifications:
        serialized = deserialized.as_dict()
        serialized_notifications.append(serialized)

    # Sanity check
    assert (
        len(valid_samples)
        == len(serialized_notifications)
        == len(deserialized_notifications)
    )
    assert valid_samples is not serialized_notifications

    # Assert deserialization-serialization rountrip produces the same result as input
    for sample, serialized in zip(valid_samples, serialized_notifications):
        assert sample == _dict_subset(serialized, sample)


def _dict_subset(origin: dict, subset: dict) -> dict:
    return {k: origin[k] for k in subset}
