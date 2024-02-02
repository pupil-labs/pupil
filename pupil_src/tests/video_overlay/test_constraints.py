import pytest

from video_overlay.utils.constraints import (
    NoConstraint,
    InclusiveConstraint,
    BooleanConstraint,
    ConstraintedValue,
    ConstraintedPosition,
)


def test_no_constraint():
    constraint = NoConstraint()
    assert constraint.apply_to(123.4) == 123.4, "NoConstraint shouldn't constrain the input"


def test_inclusive_constraint():
    constraint = InclusiveConstraint()
    assert constraint.apply_to(-123.4) == -123.4, "InclusiveConstraint with no arguments shouldn't constrain the input"
    assert constraint.apply_to(+123.4) == +123.4, "InclusiveConstraint with no arguments shouldn't constrain the input"

    constraint = InclusiveConstraint(low=-100.0)
    assert constraint.apply_to(-123.4) == -100.0, "InclusiveConstraint should constrain the input on the lower bound"
    assert constraint.apply_to(+123.4) == +123.4, "InclusiveConstraint shouldn't constrain the input on the upper bound"

    constraint = InclusiveConstraint(high=+100.0)
    assert constraint.apply_to(-123.4) == -123.4, "InclusiveConstraint shouldn't constrain the input on the lower bound"
    assert constraint.apply_to(+123.4) == +100.0, "InclusiveConstraint should constrain the input on the upper bound"

    constraint = InclusiveConstraint(low=-100.0, high=+100.0)
    assert constraint.apply_to(-123.4) == -100.0, "InclusiveConstraint should constrain the input on the lower bound"
    assert constraint.apply_to(+123.4) == +100.0, "InclusiveConstraint should constrain the input on the upper bound"


def test_boolean_constraint():
    constraint = BooleanConstraint()
    assert constraint.apply_to(0.0) == False, ""
    assert constraint.apply_to(1.0) == True, ""


def test_constrainted_value():
    val = ConstraintedValue(5)
    assert val.value == 5

    val.value = -123.4
    assert val.value == -123.4

    val.value = +123.4
    assert val.value == +123.4

    val.constraint = InclusiveConstraint(low=-100.0, high=+100.0)
    val.value = 5
    assert val.value == 5

    val.value = -123.4
    assert val.value == -100.0

    val.value = +123.4
    assert val.value == +100.0

    del val.constraint
    assert val.value == +123.4
