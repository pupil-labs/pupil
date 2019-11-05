"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import pytest
from unittest import mock

from observable import Observable, ObserverError


class FakeObservable(Observable):
    def bound_method(self):
        pass

    def bound_method_with_arguments(self, arg1, arg2):
        pass

    @staticmethod
    def static_method():
        pass

    @classmethod
    def class_method(cls):
        pass


@pytest.fixture()
def observable():
    return FakeObservable()


class TestObservabilityOfMethods:
    @pytest.fixture()
    def observer(self):
        return lambda: None

    def test_bound_method_is_observable(self, observable, observer):
        observable.add_observer("bound_method", observer)

    def test_static_method_is_not_observable(self, observable, observer):
        with pytest.raises(TypeError):
            observable.add_observer("static_method", observer)

    def test_class_method_is_not_observable(self, observable, observer):
        with pytest.raises(TypeError):
            observable.add_observer("class_method", observer)


class TestDifferentKindsOfObservers:
    def test_bound_method_without_arguments_can_be_observer(self, observable):
        # We need to be very careful with these tests!
        # Normally, we would use mock.patch.object() to patch methods and later
        # assert that they got called.
        # However, in this case we would replace the method with a mock,
        # so we wouldn't test the behavior for real methods, but for mocks.
        # Therefore, we create a FakeController that calls a mock and assert against
        # this inner mock. So we put the mock inside the method instead of wrapping
        # it around.
        # This also applies to the other tests here!

        mock_function = mock.Mock()

        class FakeController:
            def on_bound_method(self):
                mock_function()

        controller = FakeController()
        observable.add_observer("bound_method", controller.on_bound_method)
        observable.bound_method()
        mock_function.assert_called_once_with()

    def test_bound_method_with_arguments_can_be_observer(self, observable):
        mock_function = mock.Mock()

        class FakeController:
            def on_bound_method_with_arguments(self, arg1, arg2):
                mock_function(arg1, arg2)

        controller = FakeController()
        observable.add_observer(
            "bound_method_with_arguments", controller.on_bound_method_with_arguments
        )
        observable.bound_method_with_arguments(1, 2)
        mock_function.assert_called_once_with(1, 2)

    def test_callable_can_be_observer(self, observable):
        mock_function = mock.Mock()

        class FakeCallable:
            def __call__(self, *args, **kwargs):
                mock_function(*args, **kwargs)

        callable_ = FakeCallable()
        observable.add_observer("bound_method", callable_)
        observable.bound_method()
        mock_function.assert_called_once_with()

    def test_lambda_can_be_observer(self, observable):
        mock_function = mock.Mock()
        observable.add_observer("bound_method", lambda: mock_function())
        observable.bound_method()
        mock_function.assert_called_once()

    def test_function_can_be_observer(self, observable):
        mock_function = mock.Mock()

        def my_function():
            mock_function()

        observable.add_observer("bound_method", my_function)
        observable.bound_method()
        mock_function.assert_called_once()


class TestObserverCalls:
    def test_observers_are_called_with_the_same_arguments(self, observable):
        observer = mock.Mock()
        observable.add_observer("bound_method_with_arguments", observer)
        observable.bound_method_with_arguments(1, "test")
        observer.assert_called_once_with(1, "test")

    def test_observers_are_called_after_the_actual_method(self):
        mock_function = mock.Mock()

        class FakeObservable(Observable):
            def method(self):
                mock_function("method")

        observable = FakeObservable()
        observable.add_observer("method", lambda: mock_function("observer"))
        observable.method()

        mock_function.assert_has_calls(
            [mock.call("method"), mock.call("observer")], any_order=False
        )

    def test_multiple_observers_are_called(self, observable):
        observer1 = mock.Mock()
        observer2 = mock.Mock()
        observer3 = mock.Mock()

        observable.add_observer("bound_method", observer1)
        observable.add_observer("bound_method", observer2)
        observable.add_observer("bound_method", observer3)

        observable.bound_method()

        observer1_called = observer1.call_count > 0
        observer2_called = observer2.call_count > 0
        observer3_called = observer3.call_count > 0

        # no guarantees for the actual order!
        assert observer1_called and observer2_called and observer3_called


class TestRemovingObservers:
    def test_observers_that_are_functions_can_be_removed(self, observable):
        observer = mock.Mock()
        observable.add_observer("bound_method", observer)
        observable.remove_observer("bound_method", observer)
        observable.bound_method()
        observer.assert_not_called()

    def test_observers_that_are_methods_can_be_removed(self, observable):
        mock_function = mock.Mock()

        class FakeController:
            def on_bound_method(self):
                mock_function()

        controller = FakeController()
        observable.add_observer("bound_method", controller.on_bound_method)
        observable.remove_observer("bound_method", controller.on_bound_method)
        observable.bound_method()
        mock_function.assert_not_called()

    def test_invalid_method_name_raises_AttributeError(self, observable):
        with pytest.raises(AttributeError):
            observable.remove_observer("does_not_exist", lambda: None)

    def test_invalid_observer_raises_ValueError(self, observable):
        observable.add_observer("bound_method", lambda: 1)
        with pytest.raises(ValueError):
            observable.remove_observer("bound_method", lambda: 2)

    def test_never_wrapped_raises_TypeError(self, observable):
        # because we never added an observer, bound_method will just be a normal method
        with pytest.raises(TypeError):
            observable.remove_observer("bound_method", lambda: None)

    def test_all_observers_can_be_removed_at_once(self, observable):
        observer1 = mock.Mock()
        observer2 = mock.Mock()

        observable.add_observer("bound_method", observer1)
        observable.add_observer("bound_method", observer2)
        observable.remove_all_observers("bound_method")

        observable.bound_method()

        observer1_called = observer1.call_count > 0
        observer2_called = observer2.call_count > 0
        assert not observer1_called and not observer2_called

    def test_remove_all_from_empty_observer_list_is_fine(self, observable):
        observable.add_observer("bound_method", lambda: None)

        observable.remove_all_observers("bound_method")
        # at this point the list of observers is empty
        observable.remove_all_observers("bound_method")


class TestExceptionThrowingMethods:
    class FakeException(Exception):
        pass

    @pytest.fixture()
    def exception_throwing_observable(self):
        class ExceptionObservable(Observable):
            def method(self):
                raise TestExceptionThrowingMethods.FakeException

        return ExceptionObservable()

    def test_should_still_call_observers(self, exception_throwing_observable):
        mock_function = mock.Mock()
        exception_throwing_observable.add_observer("method", mock_function)
        try:
            exception_throwing_observable.method()
        except TestExceptionThrowingMethods.FakeException:
            pass
        mock_function.assert_called()

    def test_should_raise_exception(self, exception_throwing_observable):
        exception_throwing_observable.add_observer("method", lambda: None)
        with pytest.raises(TestExceptionThrowingMethods.FakeException):
            exception_throwing_observable.method()


class TestExceptionThrowingObservers:
    def test_should_raise_ObserverError(self, observable):
        def fake_observer():
            raise ValueError

        observable.add_observer("bound_method", fake_observer)

        with pytest.raises(ObserverError):
            observable.bound_method()


class TestDeletingObserverObjects:
    def test_for_observers_that_are_methods_its_object_can_get_deleted(
        self, observable
    ):
        mock_function = mock.Mock()

        class FakeController:
            def on_bound_method(self):
                mock_function()

        controller = FakeController()
        observable.add_observer("bound_method", controller.on_bound_method)
        del controller
        observable.bound_method()
        # the fake observer does get garbage collected because its referenced
        # only weakly in the observable
        mock_function.assert_not_called()

    def test_observers_that_are_callable_classes_cannot_get_deleted(self, observable):
        mock_function = mock.Mock()

        class FakeObserver:
            def __call__(self, *args, **kwargs):
                mock_function()

        observer = FakeObserver()
        observable.add_observer("bound_method", observer)
        del observer
        observable.bound_method()
        # the fake observer does not get garbage collected because its referenced
        # strongly in the observable
        mock_function.assert_any_call()


def test_observers_can_be_observed(observable):
    # This test is more delicate than it might seem at first sight.
    # When adding the first observer, the method is replaced by a wrapper of the
    # original method, which handles the calls of observers.
    # If we store a reference to the method before adding the observer, this can lead
    # to problems. When we call this reference instead of the method, we will not
    # call the wrapper but only the method. Thus, no callbacks get called.
    # Consequently, one might expect this test to fail because `on_bound_method` is
    # wrapped _after_ itself is made an observer. If `bound_method` is called,
    # one might expect that only `on_bound_method` is executed but not its associated
    # observers.
    # However, we store _weak_ references to methods and evaluate the actual method
    # anew each time, so it actually works!

    mock_function = mock.Mock()

    class FakeController(Observable):
        def on_bound_method(self):
            mock_function("First")

    controller = FakeController()
    observable.add_observer("bound_method", controller.on_bound_method)
    controller.add_observer("on_bound_method", lambda: mock_function("Second"))

    observable.bound_method()

    mock_function.assert_has_calls(
        [mock.call("First"), mock.call("Second")], any_order=False
    )
