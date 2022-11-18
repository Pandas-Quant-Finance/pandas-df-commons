

def blocking_parallel(func, args):
    from pathos.multiprocessing import Pool
    return Pool().map(func, args)


def async_parallel(func, args):
    from pathos.multiprocessing import ProcessingPool as Pool
    return Pool().amap(func, args)