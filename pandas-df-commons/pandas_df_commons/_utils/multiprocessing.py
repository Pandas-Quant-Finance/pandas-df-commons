from time import sleep
from typing import Callable, Generator, List, T, Any


def blocking_parallel(func, args):
    from pathos.multiprocessing import Pool
    return Pool().map(func, args)


def streaming_parallel(func: Callable[[Any], T], args_generator: Callable[[], Generator]) -> List[T]:
    from pathos.multiprocessing import ProcessingPool as Pool

    def wait_for_future(futures, results):
        for i, r in futures.items():
            if r.ready():
                results[i] = r.get()
                futures.pop(i)
                break

            sleep(.001)

    results = dict()
    futures = dict()
    pool = Pool()

    for i, arg in enumerate(args_generator()):
        futures[i] = pool.apipe(func, arg)
        if len(futures) > pool.ncpus:
            # we don't want to allocate all the memory if we have no free a worker, so we wait a bit
            wait_for_future(futures, results)

    while len(futures) > 0:
        wait_for_future(futures, results)

    return [v for k, v in sorted(results.items())]
