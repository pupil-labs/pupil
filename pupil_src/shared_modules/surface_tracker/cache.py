"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

logger = logging.getLogger(__name__)
import itertools


class Cache(list):
    """Cache list is a list of False
    [False,False,False]
    with update() 'False' can be overwritten with a result (anything not 'False')
    self.visited_ranges show ranges where the cache content is False
    self.positive_ranges show ranges where the cache does not evaluate as 'False' using eval_fn
    this allows to use ranges a a way of showing where no caching has happed (default) or whatever you do with eval_fn
    self.complete indicated that the cache list has no unknowns aka False
    """

    def __init__(self, init_list):
        super().__init__(init_list)

        self.length = len(self)

        self._positive_ranges = self.recompute_ranges(self.positive_eval_fn)
        self._visited_ranges = self.recompute_ranges(self.visited_eval_fn)

    @property
    def visited_ranges(self):
        return self._visited_ranges

    @property
    def positive_ranges(self):
        return self._positive_ranges

    def update(self, key, item, force=False):
        if self[key] is not None:
            if not force:
                raise IndexError(
                    "Can not overwrite an already cached position without force!"
                )
            self[key] = item
            self._visited_ranges = self.recompute_ranges(self.visited_eval_fn)
            self._positive_ranges = self.recompute_ranges(self.positive_eval_fn)

        elif item is not None:
            # unvisited
            self[key] = item

            self.update_ranges(self._visited_ranges, key)
            if self.positive_eval_fn(item):
                self.update_ranges(self._positive_ranges, key)
        else:
            raise ValueError("`None` is not a valid value to be assigned in the cache!")

    @staticmethod
    def visited_eval_fn(x):
        return x is not None

    @staticmethod
    def positive_eval_fn(x):
        return bool(x)

    def recompute_ranges(self, eval_fn):
        group_end_index = -1
        ranges = []
        for key, group in itertools.groupby(self, eval_fn):
            group_start_index = group_end_index + 1
            group_end_index += sum(1 for _ in group)
            if key:
                ranges.append([group_start_index, group_end_index])
        return ranges

    @staticmethod
    def update_ranges(ranges, index):
        for _range in ranges:
            # most common case: extend a range
            if index == _range[0] - 1:
                _range[0] = index
                Cache.merge_ranges(ranges)
                return
            elif index == _range[1] + 1:
                _range[1] = index
                Cache.merge_ranges(ranges)
                return
        # somewhere outside of range proximity
        ranges.append([index, index])
        ranges.sort(key=lambda x: x[0])

    @staticmethod
    def merge_ranges(ranges):
        for i in range(len(ranges) - 1):
            if ranges[i][1] == ranges[i + 1][0] - 1:
                # merge touching fields
                ranges[i] = [ranges[i][0], ranges[i + 1][1]]
                # del second field
                del ranges[i + 1]
                return
