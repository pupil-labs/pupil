"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import functools
import inspect
import types
import weakref


class ObserverError(Exception):
    pass


class ReplaceWrapperError(Exception):
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
                supported. Additionally, private methods are not supported.
            observer (Callable): Will be called every time the method is invoked.

        Raises:
            AttributeError: The attribute specified by method_name is not found in
                the class
            TypeError: The attribute specified by method_name is
                1) no method, or
                2) a class or static method, or
                3) a private method.

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
    wrapper = _get_wrapper_and_create_if_not_exists(obj, method_name)
    wrapper.add_observer(observer)
    _install_protection_descriptor_if_not_exists(wrapper)


def _get_wrapper_and_create_if_not_exists(obj, method_name):
    observed_method = getattr(obj, method_name)
    method_already_wrapped = isinstance(observed_method, _ObservableMethodWrapper)
    # this case has to be tested first, because if it is already wrapped,
    # its not a method anymore and the next case would raise an exception!
    if method_already_wrapped:
        return observed_method
    elif not inspect.ismethod(observed_method):
        raise TypeError(
            f"Attribute '{method_name}' of object {obj} is not a method but "
            f"{type(observed_method)} and, thus, cannot be observed!"
        )
    elif _is_classmethod(obj, method_name):
        raise TypeError(
            f"Attribute '{method_name}' of object {obj} is a class method and, thus, "
            "cannot be observed!"
        )
    else:
        return _ObservableMethodWrapper(obj, method_name)


def _is_classmethod(obj, method_name):
    try:
        method_def = obj.__class__.__dict__[method_name]
        return isinstance(method_def, classmethod)
    except (AttributeError, KeyError):
        return False


def _install_protection_descriptor_if_not_exists(wrapper):
    # we need to get class and method name from the wrapped method, not the observed
    # method. Let's say we have objects a = A() and b = B(). We monkey patch a.a = b.b
    # and observe a.a. Then we need to install the descriptor for B.b (the wrapped
    # method) and not for A.a (the observed method)
    wrapped_method = wrapper.get_wrapped_bound_method()
    cls_ = type(wrapped_method.__self__)
    name = wrapped_method.__func__.__name__
    if hasattr(cls_, name):
        already_installed = isinstance(
            getattr(cls_, name), _WrapperProtectionDescriptor
        )
    else:
        # this can happen when the method was artificially created like, for example, in
        # test_wrapped_monkey_patched_methods_not_referenced_elsewhere_are_called
        already_installed = False
    if not already_installed:
        setattr(cls_, name, _WrapperProtectionDescriptor(cls_, name))


class _WrapperProtectionDescriptor:
    """
    Protects wrappers from being replaced which would silently disable observers.
    Besides this, default python behavior is emulated.
    """

    def __init__(self, type, name):
        self.original = getattr(type, name, None)
        assert not inspect.isdatadescriptor(self.original)
        self.name = name

    def __get__(self, obj, type=None):
        # This is a data discriptor, so it's called first.
        # Emulate python lookup chain:
        # 1.) original is data discriptor: Cannot happen, we only allow methods
        #     (= non-data descriptors) to be wrapped.
        # 2.) original is in object's __dict__
        # 3.) original is in object's class or some of its parents: We don't need to
        #     worry about if it's in the class or some parent, this is covered by
        #     getattr() in __init__
        # 4.) if there is no original we raise the correct AttributeError
        if obj is not None and self.name in obj.__dict__:
            return obj.__dict__[self.name]
        elif self.original is not None:
            return self.original.__get__(obj, type)
        elif type is None:
            raise AttributeError(
                f"type object '{obj.__name__}' has no attribute '{self.name}'"
            )
        else:
            raise AttributeError(
                f"'{type.__name__}' object has no attribute '{self.name}'"
            )

    def __set__(self, obj, value):
        instance_method_wrapped = isinstance(
            getattr(obj, self.name), _ObservableMethodWrapper
        )

        obj.__dict__[self.name] = value

        # Raise error after the attribute is set. Makes it possible to change the
        # attribute and ignore the error by catching it
        if instance_method_wrapped:
            raise ReplaceWrapperError(
                f"Cannot set attribute '{self.name}' of object {obj} because it is an "
                "ObservableMethodWrapper. Replacing it would silently disable observer "
                "functionality. If you really want to, you can catch and then ignore "
                "this error."
            )


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
            f"Attribute '{method_name}' of object {obj} is not observable. You never "
            "added an observer to this!"
        )
    return observable_wrapper


class _ObservableMethodWrapper:
    def __init__(self, obj, method_name):
        # the wrapper should not block garbage collection by reference counting for the
        # observable object. Hence, we need to avoid strong references to the object,
        # which would lead to cyclic references:
        #   - The object is only referenced weakly. This is ok, because the wrapper
        #     only needs to function as long as the object exists, so there is no need
        #     to keep it alive with a strong reference.
        #   - The original wrapped method is only stored as an unbound method
        #   - In case the originally wrapped method was monkey patched, we also need to
        #     store the object it belongs to. This object we have to reference strongly
        #     as the wrapper might be (or become) the only object referencing it. This
        #     likely causes cyclic references hindering fast garbage collection in many
        #     cases, but it cannot be avoided.
        self._obj_ref = weakref.ref(obj)

        wrapped_method = getattr(obj, method_name)
        wrapped_method_self = wrapped_method.__self__
        if wrapped_method_self is obj:
            self._wrapped_method_self = None
        else:
            self._wrapped_method_self = wrapped_method_self
        self._wrapped_method_func = wrapped_method.__func__

        self._method_name = method_name
        self._observers = []
        self._was_removed = False

        self._patch_wrapper_to_look_like_original_method(wrapped_method)
        self._patch_object_to_call_wrapper_instead()

    def _patch_wrapper_to_look_like_original_method(self, wrapped_method):
        functools.update_wrapper(self, wrapped_method)
        # functools adds a reference to the wrapped method that we need to delete
        # to avoid cyclic references
        self.__wrapped__ = None

    def _patch_object_to_call_wrapper_instead(self):
        setattr(self._obj_ref(), self._method_name, self)

    def remove_wrapper(self):
        try:
            setattr(
                self._obj_ref(),
                self._method_name,
                self.get_wrapped_bound_method(),
            )
        except ReplaceWrapperError:
            pass
        self._was_removed = True

    def get_wrapped_bound_method(self):
        if self._wrapped_method_self is None:
            wrapped_method_self = self._obj_ref()
        else:
            wrapped_method_self = self._wrapped_method_self
        return types.MethodType(self._wrapped_method_func, wrapped_method_self)

    def add_observer(self, observer):
        # Observers that are bound methods are referenced weakly. That means,
        # they might get deleted together with their object if the object is not
        # referenced anymore elsewhere. The weak references are critical, because a
        # normal (=strong) reference would prevent the object from getting deleted.
        # All other observers (unbound methods, functions, lambdas, callables) are
        # referenced strongly. Otherwise, lambdas or callables would get garbage
        # collected instantly.
        if inspect.ismethod(observer):
            observer_ref = _WeakReferenceToMethodByName(observer)
        else:
            observer_ref = _StrongReferenceToCallable(observer)
        self._observers.append(observer_ref)

    def remove_observer(self, observer):
        try:
            self._observers.remove(observer)
        except ValueError:
            raise ValueError(
                f"No observer {observer} found that could be removed!"
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
        wrapped_bound_method = self.get_wrapped_bound_method()
        try:
            return wrapped_bound_method(*args, **kwargs)
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
                # exceptions by observers are wrapped s.t. they cannot be confused
                # with exceptions from the original method.
                # If there is a chain of observers calling other observers, we wrap
                # the exception only once, making the traceback a bit clearer
                if isinstance(e, ObserverError):
                    raise e
                else:
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


class _WeakReferenceToMethodByName:
    # One cannot create weakrefs to bound class methods directly, because Python
    # returns a new reference each time some method is accessed and that reference
    # gets deleted instantly with weakref (see https://stackoverflow.com/a/19443624
    # for more details).
    # That's why we store two weak refs, one to the method's object and one to the
    # unbound class method. Later, we can get the "actual" method via getattr.
    # The method is dereferenced by its name, i.e. if the method was replaced in the
    # meantime by a different one, you would get the new one when dereferencing.
    # This is to allow observers to be observed (see test case
    # test_observers_can_be_observed for a longer explanation)
    def __init__(self, method):
        self._obj_ref = weakref.ref(method.__self__)
        self._func_ref = weakref.ref(method.__func__)
        # Sanity check. Try to dereference method, to make sure that the arguments
        # are valid. Otherwise, in some cases (e.g. methods starting with '__')
        # trying to call the method will fail with _ReferenceNoLongerValidError,
        # and the observer will get silently removed. This check ensures that
        # the method is valid on creation.
        try:
            _ = self._deref_method()
        except _ReferenceNoLongerValidError:
            raise TypeError(
                f"The method {method} cannot be an observer, because it cannot be "
                "referenced by its name. Most likely, you tried to add a private method"
                " (starting with __) for which name mangling prevents referencing."
            )

    def __call__(self, *args, **kwargs):
        method = self._deref_method()
        return method(*args, **kwargs)

    def __eq__(self, other):
        if inspect.ismethod(other):
            obj_deref = self._obj_ref()
            func_deref = self._func_ref()
            if self._method_still_exists(func_deref, obj_deref):
                equal_obj = obj_deref == other.__self__
                equal_func = func_deref == other.__func__
                return equal_obj and equal_func
            else:
                return False
        else:
            return False

    def _deref_method(self):
        obj_deref = self._obj_ref()
        func_deref = self._func_ref()
        if obj_deref is None or func_deref is None:
            raise _ReferenceNoLongerValidError
        # see https://stackoverflow.com/a/6975682)
        try:
            deref_method = getattr(obj_deref, func_deref.__name__)
        except AttributeError:
            raise _ReferenceNoLongerValidError from None
        return deref_method

    @staticmethod
    def _method_still_exists(func_deref, obj_deref):
        return obj_deref is not None and func_deref is not None
