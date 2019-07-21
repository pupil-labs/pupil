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

