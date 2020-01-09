"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import pytest

from roi import RoiModel


@pytest.fixture
def invalid_model():
    return RoiModel((0, 0))


def test_fixture_invalid_model(invalid_model):
    assert invalid_model.is_invalid(), "fixture should be invalid"


@pytest.fixture
def valid_model():
    return RoiModel((300, 200))


def test_fixture_valid_model(valid_model):
    assert not valid_model.is_invalid(), "fixture should be valid"


def test_model_init_validity():
    model = RoiModel((300, 200))
    assert not model.is_invalid()

    model = RoiModel((0, 0))
    assert model.is_invalid(), "any 0 dimension should be invalid"
    model = RoiModel((1, 0))
    assert model.is_invalid(), "any 0 dimension should be invalid"
    model = RoiModel((0, 1))
    assert model.is_invalid(), "any 0 dimension should be invalid"
    model = RoiModel((1, -100))
    assert model.is_invalid(), "any 0 dimension should be invalid"
    model = RoiModel((-100, 1))
    assert model.is_invalid(), "any 0 dimension should be invalid"


def test_model_invalidation_by_set(valid_model: RoiModel):
    valid_model.set_invalid()
    assert valid_model.is_invalid(), "set_invalid should make the model invalid"


def test_model_invalidation_by_frame_size(valid_model: RoiModel):
    valid_model.frame_size = (-1, 100)
    assert (
        valid_model.is_invalid()
    ), "setting invalid frame_size should make the model invalid"


def test_model_revalidation(invalid_model: RoiModel):
    invalid_model.frame_size = (200, 400)
    assert (
        not invalid_model.is_invalid()
    ), "settingvalid frame_size should revalidate model"


def test_model_init_bounds():
    model = RoiModel((100, 200))
    assert model.bounds == (0, 0, 99, 199), "initial bounds should be full frame"


def test_model_revalidation_bounds(invalid_model):
    invalid_model.frame_size = (100, 200)
    assert invalid_model.bounds == (0, 0, 99, 199), "revalidation should set bounds to full frame"

def test_bounds_cutoff():
    model = RoiModel((100, 200))
    model.bounds = (-1, -100, 200, 300)
    assert model.bounds == (0, 0, 99, 199), "model bounds should stay within full frame"
    model.bounds = (500, 500, 400, 400)
    assert model.bounds == (99, 199, 99, 199), "model bounds should stay within full frame"
    model.bounds = (-100, -200, -400, -300)
    assert model.bounds == (0, 0, 0, 0), "model bounds should stay within full frame"
    

def test_frame_size_bounds_scaling():
    model = RoiModel((400, 800))
    model.bounds = (100, 200, 300, 400)
    model.frame_size = (800, 400)
    assert model.bounds == (200, 100, 600, 200), "bounds should be scaled by frame_size changes"
    model.frame_size = (400, 800)
    assert model.bounds == (100, 200, 300, 400), "bounds should be scaled by frame_size changes"
