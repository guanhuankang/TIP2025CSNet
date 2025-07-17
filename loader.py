import logging
from joblib import Parallel, delayed
import numpy as np

class Loader:
    def __init__(self, dataset, collate_fn, batch_size, num_workers, shuffle=True, **kwargs):
        super().__init__()
        self.n = len(dataset)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.bs = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.init()

    def __len__(self):
        return int(np.ceil(self.n / self.bs))

    def init(self):
        indexs = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indexs)
        remain = self.n % self.bs
        if remain>0:
            repeat_samples = np.random.randint(0, self.n, self.bs-remain)
            indexs = np.concatenate([indexs, repeat_samples])
        indexs = indexs.reshape(-1, self.bs)
        self.indexs = indexs
        self.ind = 0

    def __iter__(self):
        logging.info("re-initial dataset")
        self.init()
        return self

    def __next__(self):
        if self.ind>=len(self):
            raise StopIteration
        batch = [ self.dataset[i] for i in self.indexs[self.ind] ]
        self.ind += 1
        return self.collate_fn(batch)