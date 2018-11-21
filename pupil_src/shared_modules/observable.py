"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import functools
import inspect
import weakref


class ObserverError(Exception):
    pass


class Observable:
    """
    An object that allows to add observers to any of its methods. See
    the docstring for add_observer for more information.
    """

    def add_observer(self, method_name, observer):
        """
        Adds an observer to a method. An observer is an arbitrary callable. Every time
        the method is invoked, the observer will be called.

        The observer is called with the same arguments as the method, so make sure
        that the observer can handle these arguments!

        You cannot add observers to static and class methods.

        It's possible to add multiple observers to the same method.

        Observers are called AFTER the method completed. Even if the method raises an
        exception, observers are still called.
        Observers should NOT raise exceptions. If they do, exceptions are wrapped
        in an ObserverError and raised to the caller of the method.

        You only add the observer to this instance of the class. The observer is not
        triggered for calls to the same method of a different instance of the class.

        If the observer is a bound (i.e. non-static) method, it is safe to delete the
        corresponding object the observer is bound to. In this case, the observer
        will be ignored automatically. No need to remove it before deletion.

        Never use a bound method as an observer unless the corresponding object is
        still referenced somewhere else. Observables will only keep weak references
        to bound methods s.t. the objects can get deleted safely (see above).
        However, this means that the object will get garbage collected if its only
        referenced from an Observable. Use a callable class or lambda in this case.

        Args:
            method_name (String): The name of a bound method to which the observer
                will be added. Unbound methods (i.e. static and class methods) are NOT
                supported.
            observer (Callable): Will be called every time the method is invoked.

        Raises:
            AttributeError: The attribute specified by method_name is not found in
                the class
            TypeError: The attribute specified by method_name is
                1) no method, or
                2) a class or static method.

        """
        add_observer(self, method_name, observer)

    def remove_observer(self, method_name, observer):
        """
        Removes a specific observer from a method. You have to specify the same
        callable as you added before.

        Caveat: Removing lambdas does not work with this function. For example,
        the following will fail:
        > add_observer("foo", lambda: 1)
        > remove_observer("foo", lambda: 1)
        The two lambdas do the same but are two distinct objects.
        Your only option is to call remove_all_observers(), which also deletes lambdas.

        Args:
            method_name (String): Name of a bound method to which the observer was
                added before.
            observer (Callable): The observer you would like to remove.

        Raises:
            AttributeError: The attribute specified by method_name was not found in
                the class.
            ValueError: The attribute specified by method_name is not observable
                (i.e. no observer was ever added to it)
            TypeError: The observer was not removed because it was not found in the
                   list of observers.

        """
        remove_observer(self, method_name, observer)

    def remove_all_observers(self, method_name):
        """
        Removes all observers from a method. It's ok to call this if the list of
        observers is empty. However, if called on a method for which no observer was
        ever added, it raises an ValueError.

        Args:
            method_name: Name of a bound method to which some observers were
                added before.

        Raises:
            AttributeError: The attribute specified by method_name is not found in
                the class.
            TypeError: The attribute is not observable (i.e. no observer was ever
                added to it).

        """
        remove_all_observers(self, method_name)


def add_observer(obj, method_name, observer):
    """
    Adds an observer to a bound method of an arbitrary instance object.

    Usually you should not use this directly. Instead, make your class inherit from
    Observable and use the corresponding method there.

    """
    observable = _get_wrapper_and_create_if_not_exists(obj, method_name)
    observable.add_observer(observer)


def _get_wrapper_and_create_if_not_exists(obj, method_name):
    observed_method = getattr(obj, method_name)
    method_already_wrapped = isinstance(observed_method, _ObservableMethodWrapper)
    # this case has to be tested first, because if it is already wrapped,
    # its not a method anymore and the next case would raise an exception!
    if method_already_wrapped:
        return observed_method
    elif not inspect.ismethod(observed_method):
        raise TypeError(
            "Attribute {} of object {} is not a method but {} and, thus, "
            "cannot be observed!".format(method_name, obj, observed_method)
        )
    elif _is_classmethod(obj, method_name):
        raise TypeError(
            "Attribute {} of object {} is a class method and, thus, "
            "cannot be observed!".format(method_name, obj)
        )
    else:
        return _ObservableMethodWrapper(obj, method_name)


def _is_classmethod(obj, method_name):
    try:
        method_def = obj.__class__.__dict__[method_name]
        return isinstance(method_def, classmethod)
    except (AttributeError, KeyError):
        return False


def remove_observer(obj, method_name, observer):
    """
    Removes an observer from a bound method of an arbitrary object.

    Usually you should not use this directly. Instead, make your class inherit from
    Observable and use the corresponding method there.

    """
    observable = _get_wrapper_or_raise_if_not_exists(obj, method_name)
    observable.remove_observer(observer)


def remove_all_observers(obj, method_name):
    """
    Removes all observers from a bound method of an arbitrary object.

    Usually you should not use this directly. Instead, make your class inherit from
    Observable and use the corresponding method there.

    """
    observable = _get_wrapper_or_raise_if_not_exists(obj, method_name)
    observable.remove_all_observers()


def _get_wrapper_or_raise_if_not_exists(obj, method_name):
    observable_wrapper = getattr(obj, method_name)
    if not isinstance(observable_wrapper, _ObservableMethodWrapper):
        raise TypeError(
            "Attribute {} of object {} is not observable. You never added an "
            "observer to this!".format(method_name, obj)
        )
    return observable_wrapper


class _ObservableMethodWrapper:
    def __init__(self, obj, method_name):
        self._obj = obj
        self._original_method = getattr(obj, method_name)
        self._method_name = method_name
        self._observers = []
        self._was_removed = False
        self._patch_method_to_call_wrapper_instead()

    def _patch_method_to_call_wrapper_instead(self):
        functools.update_wrapper(self, self._original_method)
        setattr(self._obj, self._method_name, self)

    def remove_wrapper(self):
        setattr(self._obj, self._method_name, self._original_method)
        self._was_removed = True

    def add_observer(self, observer):
        # Observers that are bound methods are referenced weakly. That means,
        # they might get deleted together with their object if the object is not
        # referenced anymore elsewhere. The weak references are critical, because a
        # normal (=strong) reference would prevent the object from getting deleted.
        # All other observers (unbound methods, functions, lambdas, callables) are
        # referenced strongly. Otherwise, lambdas or callables would get garbage
        # collected instantly.
        if inspect.ismethod(observer):
            observer_ref = _WeakReferenceToMethod(observer)
        else:
            observer_ref = _StrongReferenceToCallable(observer)
        self._observers.append(observer_ref)

    def remove_observer(self, observer):
        try:
            self._observers.remove(observer)
        except ValueError:
            raise ValueError(
                "No observer {} found that could be removed!".format(observer)
            ) from None

    def remove_all_observers(self):
        self._observers = []

    def __call__(self, *args, **kwargs):
        if self._was_removed:
            raise RuntimeError(
                "You cannot call the wrapper after removing it from "
                "its method. Your wrapper was probably referenced "
                "elsewhere and called via this reference after you "
                "removed the wrapper!"
            )
        try:
            return self._original_method(*args, **kwargs)
        except Exception:
            raise
        finally:
            self.call_all_observers(args, kwargs)

    def call_all_observers(self, args, kwargs):
        for observer in self._observers:
            try:
                observer(*args, **kwargs)
            except _ReferenceNoLongerValidError:
                self._observers.remove(observer)
            except Exception as e:
                raise ObserverError("An observer raised an exception.") from e


class _ReferenceNoLongerValidError(Exception):
    pass


class _StrongReferenceToCallable:
    def __init__(self, observer):
        self._observer = observer

    def __call__(self, *args, **kwargs):
        return self._observer(*args, **kwargs)

    def __eq__(self, other_observer):
        return self._observer == other_observer


class _WeakReferenceToMethod:
    # One cannot create weakrefs to bound class methods directly, because Python
    # returns a new reference each time some method is accessed and that reference
    # gets deleted instantly with weakref (see https://stackoverflow.com/a/19443624
    # for more details).
    # That's why we store two weak refs, one to the method's object and one to the
    # unbound class method. Later, we can get the "actual" method via getattr (see e.g.
    # https://stackoverflow.com/a/6975682)
    def __init__(self, observer):
        self._obj_ref = weakref.ref(observer.__self__)
        self._func_ref = weakref.ref(observer.__func__)

    def __call__(self, *args, **kwargs):
        obj_deref = self._obj_ref()
        func_deref = self._func_ref()
        try:
            method = getattr(obj_deref, func_deref.__name__)
        except AttributeError:
            raise _ReferenceNoLongerValidError
        return method(*args, **kwargs)

    def __eq__(self, other_observer):
        if inspect.ismethod(other_observer):
            obj_deref = self._obj_ref()
            func_deref = self._func_ref()
            if self._method_still_exists(func_deref, obj_deref):
                equal_obj = obj_deref == other_observer.__self__
                equal_func = func_deref == other_observer.__func__
                return equal_obj and equal_func
            else:
                return False
        else:
            return False

    @staticmethod
    def _method_still_exists(func_deref, obj_deref):
        return obj_deref is not None and func_deref is not None
