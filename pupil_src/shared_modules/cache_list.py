'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import logging
logger = logging.getLogger(__name__)
import itertools

class Cache_List(list):
    """Cache list is a list of False
        [False,False,False]
        with update() 'False' can be overwritten with a result (anything not 'False')
        self.visited_ranges show ranges where the cache contect is False
        self.positive_ranges show ranges where the cache does not evaluate as 'False' using eval_fn
        this allows to use ranges a a way of showing where no caching has happed (default) or whatever you do with eval_fn
        self.complete indicated that the cache list has no unknowns aka False
    """

    def __init__(self, init_list,positive_eval_fn=None):
        super().__init__(init_list)

        self.visited_eval_fn = lambda x: x!=False
        self._visited_ranges = init_ranges(l = self,eval_fn = self.visited_eval_fn )
        self._togo = self.count(False)
        self.length = len(self)

        if positive_eval_fn == None:
            self.positive_eval_fn = lambda x: False
            self._positive_ranges = []
        else:
            self.positive_eval_fn = positive_eval_fn
            self._positive_ranges = init_ranges(l = self,eval_fn = self.positive_eval_fn )

    @property
    def visited_ranges(self):
        return self._visited_ranges

    @visited_ranges.setter
    def visited_ranges(self, value):
        raise Exception("Read only")

    @property
    def positive_ranges(self):
        return self._positive_ranges

    @positive_ranges.setter
    def positive_ranges(self, value):
        raise Exception("Read only")


    @property
    def complete(self):
        return self._togo == 0

    @complete.setter
    def complete(self, value):
        raise Exception("Read only")


    def update(self,key,item):
        if self[key] != False:
            logger.warning("You are overwriting a precached result.")
            self[key] = item
            self._visited_ranges = init_ranges(l = self,eval_fn = self.visited_eval_fn )
            self._positive_ranges = init_ranges(l = self,eval_fn = self.positive_eval_fn )

        elif item != False:
            #unvisited
            self._togo -= 1
            self[key] = item

            update_ranges(self._visited_ranges,key)
            if self.positive_eval_fn(item):
                update_ranges(self._positive_ranges,key)
        else:
            #writing False to list entry already false, do nothing
            pass


    def to_list(self):
        return list(self)



def init_ranges(l,eval_fn):
    i = -1
    ranges = []
    for t,g in itertools.groupby(l,eval_fn):
        l = i + 1
        i += len(list(g))
        if t:
            ranges.append([l,i])
    return ranges

def update_ranges(l,i):
    for _range in l:
        #most common case: extend a range
        if i == _range[0]-1:
            _range[0] = i
            merge_ranges(l)
            return
        elif i == _range[1]+1:
            _range[1] = i
            merge_ranges(l)
            return
    #somewhere outside of range proximity
    l.append([i,i])
    l.sort(key=lambda x: x[0])

def merge_ranges(l):
    for i in range(len(l)-1):
        if l[i][1] == l[i+1][0]-1:
            #merge touching fields
            l[i] = ([l[i][0],l[i+1][1]])
            #del second field
            del l[i+1]
            return
    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cl = Cache_List(range(100),positive_eval_fn = lambda x: bool(x%2))
    cl.update(0,1)
    cl.update(4,1)
    print(cl.positive_ranges)
    print(cl)