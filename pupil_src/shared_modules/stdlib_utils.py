import typing
import itertools
import collections


class sliceable_deque(collections.deque):
    """
    deque subclass with support for slicing.
    """
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(itertools.islice(self, index.start, index.stop, index.step), maxlen=self.maxlen)
        return collections.deque.__getitem__(self, index)


Unique_Element = collections.abc.Hashable
Unique_Key = collections.abc.Hashable
Unique_Key_Getter = typing.Callable[[Unique_Element], Unique_Key]
Unique_Select = typing.Callable[[Unique_Key, Unique_Key], Unique_Key]


class unique(collections.abc.Iterable):

    def __init__(self, it: typing.Iterable, key: Unique_Key_Getter=..., select: Unique_Select=...):
        self._it = list(it)
        self._key = key if key is not ... else lambda elem: elem
        self._select = select if select is not ... else lambda x, y: x

    def __iter__(self):
        by_key = collections.OrderedDict()

        for new_elem in self._it:
            key = self._key(new_elem)

            try:
                old_elem = by_key[key]
            except KeyError:
                old_elem = None

            if old_elem is None:
                by_key[key] = new_elem
            else:
                by_key[key] = self._select(old_elem, new_elem)

        return iter(by_key.values())


def test_unique():
    arr = [1, 2, 3, 1, 4, 2, 2, 4, 5, 6, 0]
    assert list(unique(arr)) == [1, 2, 3, 4, 5, 6, 0]
    assert list(unique(arr)) == list(unique(unique(arr)))

    assert list(unique(arr, key=lambda _: 0)) == [1]
    assert list(unique(arr, key=lambda _: 0, select=lambda old, new: new)) == [0]
    assert list(unique(arr, select=lambda old, new: old*10 + new)) == [11, 222, 3, 44, 5, 6, 0]

if __name__ == '__main__':
    test_unique()
