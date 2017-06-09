'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import multiprocessing as mp
from ctypes import c_bool

import logging
logger = logging.getLogger(__name__)


class EarlyCancellationError(Exception):
    pass


class Task_Proxy(object):
    '''Future like object that runs a given generator in the background and returns is able to return the results incrementally'''
    def __init__(self, name, generator,force_spawn=False, args=(), kwargs={}):
        super(Task_Proxy, self).__init__()

        if force_spawn:
            import multiprocessing as mp
            mp = mp.get_context('spawn')
        else:
            import multiprocessing as mp

        self.should_terminate_flag = mp.Value(c_bool, 0)
        self._completed = False

        pipe_recv, pipe_send = mp.Pipe(False)
        wrapper_args = [pipe_send, self.should_terminate_flag, generator]
        wrapper_args.extend(args)
        self.process = mp.Process(target=self._wrapper, name=name, args=wrapper_args, kwargs=kwargs)
        self.process.start()
        self.pipe = pipe_recv

    def _wrapper(self, pipe, should_terminate_flag, generator, *args, **kwargs):
        '''Executed in background, pipes generator results to foreground'''
        logger.debug('Entering _wrapper')
        try:
            for datum in generator(*args, **kwargs):
                if should_terminate_flag.value:
                    raise EarlyCancellationError('Task was cancelled')
                pipe.send(datum)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            pipe.send(e)
        else:
            pipe.send(0xDeadBeef)
        finally:
            pipe.close()
            logger.debug('Exiting _wrapper')

    def fetch(self):
        '''Fetches progress and available results from background'''
        logger.debug('Fetching')
        while self.pipe.poll(0):
            datum = self.pipe.recv()
            if datum == 0xDeadBeef:
                self._completed = True
                return
            elif isinstance(datum, Exception):
                raise datum
            else:
                yield datum

    def cancel(self, wait=False):
        self.should_terminate_flag.value = True
        if wait:
            self.process.join()

    @property
    def completed(self):
        return self._completed

    def __del__(self):
        self.cancel()
        self.process = None
        self.pipe.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(processName)s - [%(levelname)s] %(name)s: %(message)s')

    def example_generator(mu=0., sigma=1., steps=100):
        '''samples `N(\mu, \sigma^2)`'''
        import numpy as np
        from time import sleep
        for i in range(steps):
            # yield progress, datum
            yield (i + 1) / steps, sigma * np.random.randn() + mu
            sleep(np.random.rand() * .1)

    # initialize task proxy
    task = Task_Proxy('Background', example_generator, args=(5., 3.), kwargs={'steps': 50})

    from time import time, sleep
    start = time()
    maximal_duration = 2.
    while time() - start < maximal_duration:
        # fetch all available results
        for progress, random_number in task.fetch():
            logger.debug('[{:3.0f}%] {:0.2f}'.format(progress * 100, random_number))

        # test if task is completed
        if task.completed:
            break
        sleep(1.)

    logger.debug('Canceling task')
    task.cancel(wait=True)
    logger.debug('Task done')
