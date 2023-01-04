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
from roi import RoiModel


@pytest.fixture
def invalid_model():
    return RoiModel((0, 0))


def test_fixture_invalid_model(invalid_model):
    # ensure fixture is invalid
    assert invalid_model.is_invalid()


@pytest.fixture
def valid_model():
    return RoiModel((300, 200))


def test_fixture_valid_model(valid_model):
    # ensure fixture is valid
    assert not valid_model.is_invalid()


def test_model_init_validity():
    model = RoiModel((300, 200))
    assert not model.is_invalid()

    # any dimension <= 0 should be invalid
    model = RoiModel((0, 0))
    assert model.is_invalid()
    model = RoiModel((1, 0))
    assert model.is_invalid()
    model = RoiModel((0, 1))
    assert model.is_invalid()
    model = RoiModel((1, -100))
    assert model.is_invalid()
    model = RoiModel((-100, 1))
    assert model.is_invalid()


def test_model_invalidation_by_set(valid_model: RoiModel):
    # set_invalid should make the model invalid
    valid_model.set_invalid()
    assert valid_model.is_invalid()


def test_model_invalidation_by_frame_size(valid_model: RoiModel):
    # setting invalid frame_size should make the model invalid
    valid_model.frame_size = (-1, 100)
    assert valid_model.is_invalid()


def test_model_revalidation(invalid_model: RoiModel):
    # setting valid frame_size should revalidate model
    invalid_model.frame_size = (200, 400)
    assert not invalid_model.is_invalid()


def test_model_init_bounds():
    # initial bounds should be full frame
    model = RoiModel((100, 200))
    assert model.bounds == (0, 0, 99, 199)


def test_model_revalidation_bounds(invalid_model):
    # revalidation should set bounds to full frame
    invalid_model.frame_size = (100, 200)
    assert invalid_model.bounds == (0, 0, 99, 199)


def test_bounds_cutoff():
    model = RoiModel((100, 200))

    # model bounds should stay within full frame
    model.bounds = (-1, -100, 200, 300)
    assert model.bounds == (0, 0, 99, 199)

    # model bounds should always have area > 0
    model.bounds = (500, 500, 400, 400)
    minx, miny, maxx, maxy = model.bounds
    assert 0 <= minx < maxx < 100 and 0 <= miny < maxy < 200

    # model bounds should always have area > 0
    model.bounds = (-100, -200, -400, -300)
    minx, miny, maxx, maxy = model.bounds
    assert 0 <= minx < maxx < 100 and 0 <= miny < maxy < 200


def test_frame_size_bounds_scaling():
    model = RoiModel((400, 800))
    model.bounds = (100, 200, 300, 400)

    # bounds should be scaled by frame_size changes
    model.frame_size = (800, 400)
    assert model.bounds == (200, 100, 600, 200)
    model.frame_size = (400, 800)
    assert model.bounds == (100, 200, 300, 400)
