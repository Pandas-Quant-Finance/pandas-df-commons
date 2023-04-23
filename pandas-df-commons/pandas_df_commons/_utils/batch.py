from typing import Iterator


class Batch(object):

    def __init__(self, iterable, batch_size=1):
        super().__init__()
        self.iterable = iterable
        self.batch_size = batch_size
        self.batch = []

    def __iter__(self):
        return self

    def __next__(self):
        if not isinstance(self.iterable, Iterator):
            self.iterable = iter(self.iterable)

        if self.batch is None:
            raise StopIteration()

        try:
            while len(self.batch) < self.batch_size:
                self.batch.append(next(self.iterable))

            batch = self.batch
            self.batch = []
            return batch

        except StopIteration as si:
            batch = self.batch
            self.batch = None
            return batch


