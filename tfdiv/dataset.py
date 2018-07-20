from scipy.sparse import isspmatrix_csr
from abc import abstractmethod, ABC
from tfdiv.utility import sparse_repr, cartesian_product
from collections import defaultdict
import random
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

    def __init__(self, pos, neg=None, ntype=np.float32, bootstrap_sampling='no_sample', max_samples=None):
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
            self.sampler = UniformUserSampler(pos, neg, max_samples=max_samples)
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

    def __init__(self, pos, neg=None, max_samples=None):
        super(UniformUserSampler, self).__init__(pos, neg)
        self.max_samples = max_samples or self.pdf.shape[0]

        self.pos_idx = defaultdict(list)
        for xi in self.pdf.values:
            self.pos_idx[xi[0]].append(xi[1])
        if neg is not None:
            self.neg_idx = defaultdict(list)
            for xi in self.ndf.values:
                self.neg_idx[xi[0]].append(xi[1])
        self.users = self.pdf.user.unique()

    def uniform_user(self):
        return random.choice(self.users)

    def sample(self, batch_size):
        idx = []
        for _ in np.arange(self.max_samples):
            u = self.uniform_user()
            p = random.choice(self.pos_idx[u])
            if self.ndf is not None:
                n = random.choice(self.neg_idx[u])
                idx.append([p, n])
            else:
                idx.append(p)
            if len(idx) == batch_size:
                yield np.array(idx)
                idx = []
        if len(idx) > 0:
            yield np.array(idx)


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

    def __init__(self, x, y=None, w=None,
                 ntype=np.float32):
        super(SimpleDataset, self).__init__()
        if not isspmatrix_csr(x):
            raise TypeError("Unsupported Type: {}.\n"
                            "Use scipy.sparse.csr_matrix instead"
                            .format(type(x)))
        self.ntype = ntype
        self.x = x
        self.y = y
        self.w = w

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
            w = self.w[idx]

        for i in range(0, n_samples, batch_size):
            upper_bound = min(i + batch_size, n_samples)
            batch_x = x[i:upper_bound]
            if self.y is not None:
                batch_y = y[i:i + batch_size]
                batch_w = w[i:i + batch_size]
                yield (batch_x, batch_y, batch_w)
            else:
                yield batch_x

    def batch_to_feed_dict(self, x, y=None, w=None, core=None):
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
            fd[core.w] = w.astype(self.ntype)

        return fd
