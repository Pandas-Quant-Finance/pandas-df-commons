import os
from time import sleep
from typing import Callable, Generator, List, T, Any


def _get_pool(streaming=False):
    if streaming:
        from pathos.multiprocessing import ProcessingPool as Pool
    else:
        from pathos.multiprocessing import Pool

    pool_size = os.environ.get("PANDAS_DF_COMMONS_POOL_SIZE", "")
    return Pool(nodes=int(pool_size)) if pool_size else Pool()


def blocking_parallel(func, args):
    return _get_pool().map(func, args)


def streaming_parallel(func: Callable[[Any], T], args_generator: Callable[[], Generator]) -> List[T]:
    def wait_for_future(futures, results):
        for i, r in futures.items():
            if r.ready():
                results[i] = r.get()
                futures.pop(i)
                break

            sleep(.001)

    results = dict()
    futures = dict()
    pool = _get_pool(True)
    pool.restart(force=True)

    try:
        for i, arg in enumerate(args_generator()):
            futures[i] = pool.apipe(func, arg)
            if len(futures) > pool.ncpus:
                # we don't want to allocate all the memory if we have no free a worker, so we wait a bit
                wait_for_future(futures, results)

        while len(futures) > 0:
            wait_for_future(futures, results)

        return [v for k, v in sorted(results.items())]
    finally:
        try:
            pool.close()
        except Exception as ignore:
            pass
