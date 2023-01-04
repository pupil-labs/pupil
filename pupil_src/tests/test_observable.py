"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import gc
import types
from unittest import mock

import pytest
from observable import Observable, ObserverError, ReplaceWrapperError


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

    def __del__(self):
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

    def test_bound_method_from_parent_class_is_observable(self, observer):
        class FakeObservableChild(FakeObservable):
            pass

        observable_child = FakeObservableChild()
        observable_child.add_observer("bound_method", observer)

    def test_static_method_is_not_observable(self, observable, observer):
        with pytest.raises(TypeError):
            observable.add_observer("static_method", observer)

    def test_class_method_is_not_observable(self, observable, observer):
        with pytest.raises(TypeError):
            observable.add_observer("class_method", observer)

    def test_monkey_patched_methods_are_observable(self, observable, observer):
        class FakeClass:
            def fake_method(self):
                pass

        fake_class_instance = FakeClass()
        observable.bound_method = fake_class_instance.fake_method

        observable.add_observer("bound_method", observer)

    def test_new_methods_are_observable(self, observable, observer):
        class FakeClass:
            def fake_method(self):
                pass

        fake_class_instance = FakeClass()
        # new_method is not part of FakeObservable
        observable.new_method = fake_class_instance.fake_method

        observable.add_observer("new_method", observer)


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

    def test_private_method_cannot_be_observer_from_inside(self, observable):
        class FakeController:
            def __private_method(self):
                pass

            def add_private_observer(self, observable):
                observable.add_observer("bound_method", self.__private_method)

        controller = FakeController()
        with pytest.raises(TypeError):
            controller.add_private_observer(observable)

    def test_private_method_cannot_be_observer_from_outside(self, observable):
        class FakeController:
            def __private_method(self):
                pass

        controller = FakeController()
        with pytest.raises(TypeError):
            # in your own class you would just write self.__private_method,
            # but here we need to give the mangled name, otherwise python tries to
            # access controller._TestDifferentKindsOfObservers__private_method
            observable.add_observer(
                "bound_method", controller._FakeController__private_method
            )


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

    def test_observers_of_monkey_patched_methods_are_called(self, observable):
        observer = mock.Mock()

        class FakeClass:
            def fake_method(self, arg1, arg2):
                pass

        fake_object = FakeClass()
        observable.bound_method_with_arguments = fake_object.fake_method
        observable.add_observer("bound_method_with_arguments", observer)
        observable.bound_method_with_arguments(1, "test")
        observer.assert_called_once_with(1, "test")


class TestWrappedMethodCalls:
    def test_wrapped_functions_are_called_with_right_arguments(self):
        mock_function = mock.Mock()

        class FakeObservable(Observable):
            def method(self, arg1, arg2):
                mock_function(arg1, arg2)

        observable = FakeObservable()
        observable.add_observer("method", lambda arg1, arg2: None)

        observable.method(1, 2)

        mock_function.assert_called_once_with(1, 2)

    def test_wrapped_functions_return_values(self):
        class FakeObservable(Observable):
            def method(self):
                return 1

        observable = FakeObservable()
        observable.add_observer("method", lambda: None)

        ret_val = observable.method()

        assert ret_val == 1

    def test_wrapped_monkey_patched_functions_are_called_with_right_arguments(
        self, observable
    ):
        mock_function = mock.Mock()

        class FakeClass:
            # we choose a name that is also in FakeObservable to check that the
            # two methods are not confused with each other
            def bound_method_with_arguments(self, arg1, arg2):
                mock_function(arg1, arg2)

        fake_object = FakeClass()
        observable.bound_method_with_arguments = fake_object.bound_method_with_arguments
        observable.add_observer("bound_method_with_arguments", lambda arg1, arg2: None)

        observable.bound_method_with_arguments(1, 2)

        mock_function.assert_called_once_with(1, 2)

    def test_wrapped_monkey_patched_methods_not_referenced_elsewhere_are_called_part1(
        self, observable
    ):
        mock_function = mock.Mock()

        def patch(obj):
            def fake_method(self):
                mock_function()

            obj.bound_method = types.MethodType(fake_method, obj)

        patch(observable)

        # This tests that observer wrappers reference the wrapped method strongly.
        # If it is only referenced weakly, it will get garbage collected as soon as
        # the wrapper replaces obj.bound_method. At this point, obj.bound_method is
        # the only reference to this instance of fake_method, because it was only
        # defined inside patch(), which just ended.
        observable.add_observer("bound_method", lambda: None)
        observable.bound_method()

        mock_function.assert_called_once()

    def test_wrapped_monkey_patched_methods_not_referenced_elsewhere_are_called_part2(
        self, observable
    ):
        mock_function = mock.Mock()

        def patch(obj):
            class FakeClass:
                def fake_method(self):
                    mock_function()

            fake_object = FakeClass()
            obj.bound_method = fake_object.fake_method

        patch(observable)

        # Similarly to part 1 of this test, the monkey patched method is now only
        # referenced in observable. What's different is that in this part also the
        # class the method belongs to is at risk of being garbage collected.
        observable.add_observer("bound_method", lambda: None)
        observable.bound_method()

        mock_function.assert_called_once()


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

    def test_wrapper_can_be_removed(self, observable):
        original_method = observable.bound_method
        observable.add_observer("bound_method", lambda: None)
        observable.bound_method.remove_wrapper()
        method_after_removal = observable.bound_method

        assert method_after_removal == original_method


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


class TestDeletingObservableObjects:
    @pytest.fixture()
    def disable_generational_garbage_collection(self):
        gc_enabled = gc.isenabled()
        gc.disable()
        yield
        if gc_enabled:
            gc.enable()

    def test_observable_object_can_be_freed_by_reference_counting(
        self, disable_generational_garbage_collection
    ):
        # do not use our fixture `observable` here, for some reason pytest keeps
        # additional references to the object which makes the following test impossible
        observable = FakeObservable()
        with mock.patch.object(FakeObservable, "__del__") as mock_function:
            observable.add_observer("bound_method", mock.Mock())
            del observable
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


class TestWrapperProtectionDescriptor:
    def test_wrappers_of_methods_cannot_be_set(self, observable):
        observable.add_observer("bound_method", lambda: None)
        with pytest.raises(ReplaceWrapperError):
            observable.bound_method = 42

    def test_wrappers_of_monkey_patched_methods_cannot_be_set(self, observable):
        class FakeClass:
            def fake_method(self):
                pass

        fake_object = FakeClass()
        observable.bound_method = fake_object.fake_method
        observable.add_observer("bound_method", lambda: None)
        with pytest.raises(ReplaceWrapperError):
            observable.bound_method = 42

    def test_wrappers_not_in_class_cannot_be_set(self, observable):
        def fake_method(self):
            pass

        observable.bound_method = types.MethodType(fake_method, observable)
        observable.add_observer("bound_method", lambda: None)

        # fake_method is not part of FakeObservable, so the installation of the
        # protection descriptor needs to account for that
        with pytest.raises(ReplaceWrapperError):
            observable.bound_method = 42

    def test_other_instances_without_wrapper_can_be_set(self, observable):
        observable.add_observer("bound_method", lambda: None)
        unwrapped_observable = FakeObservable()
        unwrapped_observable.bound_method = 42
