from threading import Thread

from scipy.sparse import isspmatrix_csr
from multiprocessing import Queue
from abc import abstractmethod, ABC
from tfdiv.utility import sparse_repr, cartesian_product
import pandas as pd
import numpy as np


class Dataset:

    def __init__(self):
        self._shuffle = False
        self.batch_size = -1

    def shuffle(self, shuffle):
        self._shuffle = shuffle
        return self

    def batch(self, batch_size):
        self.batch_size = batch_size
        return self

    @abstractmethod
    def get_next(self):
        pass

    @abstractmethod
    def batch_to_feed_dict(self, *args):
        pass


class PairDataset(Dataset):
    """
    Attributes
    ----------
    x : scipy.sparse.csr_matrix, shape (n_samples, n_features)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y : np.array or None, shape (n_samples,)
        Target vector relative to X.

    batch_size : int
        Size of batches.
        Use -1 for full-size batches

    shuffle: bool
        Whether to shuffle the dataset or not

    """

    def __init__(self, pos, neg=None, ntype=np.float32, bootstrap_sampling='no_sample',
                 frac=0.5, shuffle_size=1000, n_threads=2):
        super(PairDataset, self).__init__()
        if not isspmatrix_csr(pos) or (neg is not None
                                       and not isspmatrix_csr(neg)):
            raise TypeError("Unsupported Type: {}.\n"
                            "Use scipy.sparse.csr_matrix instead"
                            .format(type(pos)))
        self.pos = pos
        self.neg = neg
        self.ntype = ntype
        if bootstrap_sampling == 'uniform_user':
            self.sampler = UniformUserSampler(pos, neg,
                                              frac=frac,
                                              shuffle_size=shuffle_size,
                                              n_threads=n_threads)
        elif bootstrap_sampling == 'no_sample':
            self.sampler = NoSample(pos, neg)
        else:
            raise "Unsupported Bootstrap Sampling Type: {}. " \
                  "Please use Random or Uniform User." \
                .format(bootstrap_sampling)

    def batch_to_feed_dict(self, pos, neg=None, core=None):
        fd = {core.x: sparse_repr(pos, self.ntype)}

        if neg is not None:
            fd[core.y] = sparse_repr(neg, self.ntype)
        return fd

    def get_next(self):
        for idx in self.sampler.sample(self.batch_size):
            if self.neg is None:
                pos = self.pos[idx]
                pos.sort_indices()
                yield pos
            else:
                pos = self.pos[idx[:, 0]]
                pos.sort_indices()
                neg = self.neg[idx[:, 1]]
                neg.sort_indices()
                yield pos, neg

    def _get_batch_size(self, n_samples):
        if self.batch_size == -1:
            batch_size = n_samples
        elif self.batch_size < 1:
            raise ValueError('Parameter batch_size={} '
                             'is unsupported'.format(self.batch_size))
        else:
            batch_size = self.batch_size
        return batch_size


class Sampler(ABC):

    def __init__(self, pos, neg):
        self.pdf, self.ndf = self.get_index(pos, neg)

    @staticmethod
    def get_index(pos, neg):
        poo = np.transpose(pos.nonzero())
        pdf = pd.DataFrame(poo, columns=['prow', 'user']) \
            .groupby('prow', as_index=False).min()

        ndf = None
        if neg is not None:
            noo = np.transpose(neg.nonzero())
            ndf = pd.DataFrame(noo, columns=['nrow', 'user']) \
                .groupby('nrow', as_index=False).min()
        return pdf, ndf

    @abstractmethod
    def sample(self, batch_size):
        pass


class NoSample(Sampler):

    def __init__(self, pos, neg=None):
        super(NoSample, self).__init__(pos, neg)

    def sample(self, batch_size):
        for u in self.pdf.user.unique():
            pos = self.pdf.loc[self.pdf['user'] == u, 'prow'].values
            if self.ndf is not None:
                neg = self.ndf.loc[self.ndf['user'] == u, 'nrow'].values
                idx = cartesian_product(pos, neg)
            else:
                idx = np.array(pos)
            for i in range(0, idx.shape[0], batch_size):
                upper_bound = min(i + batch_size, idx.shape[0])
                yield idx[i:upper_bound]


class UniformUserSampler(Sampler):

    def __init__(self, pos, neg=None,
                 frac=0.5, shuffle_size=1000, n_threads=2):
        super(UniformUserSampler, self).__init__(pos, neg)
        self.shuffle_size = shuffle_size
        self.n_threads = n_threads
        self.frac = frac

    def sample(self, batch_size):
        def cartesian(inbound_queue, outbound_queue, pdf, ndf, frac):
            while True:
                u = inbound_queue.get()
                if type(u) is dict:
                    outbound_queue.put(u)
                    break
                pos = pdf.loc[pdf['user'] == u, 'prow'].values
                if ndf is not None:
                    neg = ndf.loc[ndf['user'] == u, 'nrow'].values
                    idx = cartesian_product(pos, neg)
                else:
                    idx = np.array(pos)
                rows_idx = np.arange(idx.shape[0])
                np.random.shuffle(rows_idx)
                idx = idx[rows_idx[0:int(frac * idx.shape[0])]]
                outbound_queue.put(idx)

        out_queue = Queue(maxsize=0 if batch_size == -1 else self.shuffle_size)
        in_queue = Queue()

        pos_idx, neg_idx = self.pdf, self.ndf
        for i in range(self.n_threads):
            worker = Thread(target=cartesian,
                            args=(in_queue, out_queue,
                                  pos_idx, neg_idx, self.frac))
            worker.setDaemon(True)
            worker.start()

        def main(pidx, q, n_threads):
            for u in pidx.user.unique():
                q.put(u)
            for j in range(n_threads):
                q.put({"end": j})

        Thread(target=main,
               args=(self.pdf, in_queue, self.n_threads),
               daemon=True).start()

        n_thread_ended = 0
        buffer = []
        while True:
            item = out_queue.get()
            if type(item) is dict:
                n_thread_ended += 1
            else:
                buffer.append(item)
            if n_thread_ended == self.n_threads:
                break
            if batch_size != -1 and self.shuffle_size != 0 and \
                    len(buffer) == self.shuffle_size:
                buffer = np.concatenate(buffer)
                np.random.shuffle(buffer)
                for i in range(0, buffer.shape[0], batch_size):
                    upper_bound = min(i + batch_size, buffer.shape[0])
                    yield buffer[i:upper_bound]
                buffer = []

        if batch_size == -1:
            np.random.shuffle(buffer)
            yield np.concatenate(buffer)
        else:
            buffer = np.concatenate(buffer)
            np.random.shuffle(buffer)
            for i in range(0, buffer.shape[0], batch_size):
                upper_bound = min(i + batch_size, buffer.shape[0])
                yield buffer[i:upper_bound]


#
#
# class RandomSampler(Sampler):
#
#     def __init__(self, pos, neg=None, shuffle_size=1000, n_threads=2):
#         super(RandomSampler, self).__init__(pos, neg)
#         self.shuffle_size = shuffle_size
#         self.n_threads = n_threads
#
#     def sample(self, batch_size):
#
#         def cartesian_shuffling(id, batch, q, shuffle_size, pos_idx, neg_idx):
#             head = None
#             for u in batch:
#                 pos = pos_idx[u]
#                 if neg_idx is not None:
#                     neg = neg_idx[u]
#                     idx = cartesian_product(pos, neg)
#                 else:
#                     idx = np.array(pos)
#                 if head is not None:
#                     idx = np.concatenate((head, idx))
#                     head = None
#                 if idx.shape[0] >= shuffle_size:
#                     np.random.shuffle(idx)
#                     for i in range(0, idx.shape[0], shuffle_size):
#                         upper_bound = min(i + shuffle_size, idx.shape[0])
#                         if upper_bound - i == shuffle_size:
#                             q.put(idx[i:upper_bound])
#                         else:
#                             head = idx[i:upper_bound]
#                 else:
#                     head = idx
#             if head is not None:
#                 q.put(head)
#             q.put({'id': id})
#
#         q = Queue(maxsize=self.shuffle_size)
#
#         users = np.array(list(self.pos_idx.keys()))
#         np.random.shuffle(users)
#
#         batch_users = int(users.shape[0] / self.n_threads)
#         threads = []
#         for i in range(0, users.shape[0], batch_users):
#             upper_bound = min(i + batch_users, users.shape[0])
#             t = Thread(target=cartesian_shuffling,
#                        args=(i, users[i:upper_bound], q, self.shuffle_size,
#                              self.pos_idx, self.neg_idx))
#             t.setDaemon(True)
#             threads.append(t)
#
#         for t in threads:
#             t.start()
#
#         final = None
#         ended_threads = set()
#         while True:
#             if len(ended_threads) == len(threads):
#                 break
#             shuffled_batch = q.get()
#             if type(shuffled_batch) is dict:
#                 ended_threads.add(shuffled_batch['id'])
#             elif batch_size != -1:
#                 for i in range(0, shuffled_batch.shape[0], batch_size):
#                     upper_bound = min(i + batch_size, shuffled_batch.shape[0])
#                     yield shuffled_batch[i:upper_bound]
#             elif final is None:
#                 final = shuffled_batch
#             else:
#                 final = np.concatenate((final, shuffled_batch))


class SimpleDataset(Dataset):
    """
    Attributes
    ----------
    x : scipy.sparse.csr_matrix, shape (n_samples, n_features)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y : np.array or None, shape (n_samples,)
        Target vector relative to X.

    batch_size : int
        Size of batches.
        Use -1 for full-size batches

    shuffle: bool
        Whether to shuffle the dataset or not

    """

    def __init__(self, x, y=None,
                 ntype=np.float32):
        super(SimpleDataset, self).__init__()
        if not isspmatrix_csr(x):
            raise TypeError("Unsupported Type: {}.\n"
                            "Use scipy.sparse.csr_matrix instead"
                            .format(type(x)))
        self.ntype = ntype
        self.x = x
        self.y = y

    def get_next(self):
        """
        Split data to mini-batches.

        Yields
        -------
        batch_x : scipy.sparse.csr_matrix, shape (batch_size, n_features)
            Same type as input samples

        batch_y : np.array or None, shape (batch_size,)

        """
        n_samples = self.x.shape[0]

        if self.batch_size == -1:
            batch_size = n_samples
        elif self.batch_size < 1:
            raise ValueError('Parameter batch_size={} '
                             'is unsupported'.format(self.batch_size))
        else:
            batch_size = self.batch_size

        if self._shuffle:
            idx = np.random.permutation(n_samples)
        else:
            idx = np.arange(n_samples)

        x = self.x[idx, :]
        x.sort_indices()
        if self.y is not None:
            y = self.y[idx]

        for i in range(0, n_samples, batch_size):
            upper_bound = min(i + batch_size, n_samples)
            batch_x = x[i:upper_bound]
            if self.y is not None:
                batch_y = y[i:i + batch_size]
                yield (batch_x, batch_y)
            else:
                yield batch_x

    def batch_to_feed_dict(self, x, y=None, core=None):
        """
        Parameters:
        -------
        batch_x : scipy.sparse.csr_matrix, shape (batch_size, n_features)
            Batch of training vector

        batch_y : np.array or None, shape (batch_size,), default=None
            Batch of target vector relative to batch_x

        core : tfdiv.graph.ComputationalGraph
            ComputationalGraph associated to the Classifier

        Returns
        -------
        fd : dict
            Dict with formatted placeholders
        """
        fd = {}

        # sparse case
        indices, values, shape = sparse_repr(x, self.ntype)

        fd[core.x] = (indices, values, shape)

        if y is not None:
            fd[core.y] = y.astype(self.ntype)

        return fd
