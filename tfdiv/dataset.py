from collections import defaultdict

from scipy.sparse import isspmatrix_csr
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

    def __init__(self, pos, neg=None, ntype=np.float32,
                 bootstrap_sampling='no_sample', frac=0.5, ):
        super(PairDataset, self).__init__()
        if not isspmatrix_csr(pos) or (neg is not None
                                       and not isspmatrix_csr(neg)):
            raise TypeError("Unsupported Type: {}.\n"
                            "Use scipy.sparse.csr_matrix instead"
                            .format(type(pos)))
        self.pos = pos
        self.neg = neg
        self.ntype = ntype
        if bootstrap_sampling == 'random':
            self.sampler = RandomSampler(pos, neg,
                                         frac=frac,
                                         ntype=ntype)
        elif bootstrap_sampling == 'uniform_user':
            self.sampler = UniformUserSampler(pos, neg,
                                              frac=frac,
                                              ntype=ntype)
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
        self.size = None
        self.pos_idx, self.neg_idx = self.get_index(pos, neg)

    @staticmethod
    def get_index(pos, neg):
        poo = np.transpose(pos.nonzero())
        pdf = pd.DataFrame(poo, columns=['prow', 'user']) \
            .groupby('prow', as_index=False).min()
        pos_idx = defaultdict(list)
        for k, g in pdf.groupby('user', as_index=False):
            pos_idx[k].extend(g['prow'])

        neg_idx = None
        if neg is not None:
            noo = np.transpose(neg.nonzero())
            ndf = pd.DataFrame(noo, columns=['nrow', 'user']) \
                .groupby('nrow', as_index=False).min()
            neg_idx = defaultdict(list)
            for k, g in ndf.groupby('user', as_index=False):
                neg_idx[k].extend(g['nrow'])
        return pos_idx, neg_idx

    @abstractmethod
    def sample(self, batch_size):
        pass


class NoSample(Sampler):

    def __init__(self, pos, neg=None):
        super(NoSample, self).__init__(pos, neg)

    def sample(self, batch_size):
        head = None
        for k in self.pos_idx.keys():
            pos = self.pos_idx[k]
            if self.neg_idx is not None:
                neg = self.neg_idx[k]
                idx = cartesian_product(pos, neg)
            else:
                idx = np.array(pos)
            if head is not None:
                idx = np.concatenate((head, idx))
            if batch_size != -1:
                for i in range(0, idx.shape[0], batch_size):
                    upper_bound = min(i+batch_size, idx.shape[0])
                    if idx[i:upper_bound].shape[0] < batch_size:
                        head = idx[i:upper_bound]
                        break
                    else:
                        yield idx[i:upper_bound]
                        head = None
            else:
                head = head if head is not None else idx
        if head is not None:
            yield head


class RandomSampler(Sampler):

    def __init__(self, pos, neg=None,
                 frac=0.5, ntype=np.float32):
        super(RandomSampler, self).__init__()
        self.pos = pos
        self.neg = neg
        self.ntype = ntype
        n_samples, _ = pos.shape
        self.size = int(n_samples * frac)
        self.indexes = self.get_index()

    def _get_pos_sample_index(self):
        self.sample_idx = self.indexes.sample(self.size)
        return self.sample_idx['prow']

    def _get_neg_sample_index(self):
        return self.sample_idx['nrow']


class UniformUserSampler(Sampler):

    def __init__(self, pos, neg=None,
                 frac=0.5, ntype=np.float32):
        super(UniformUserSampler, self).__init__()

        self.pos = pos
        self.neg = neg
        self.ntype = ntype
        self.frac = frac
        self.indexes = self.get_index()

    def _get_pos_sample_index(self):
        self.sample_idx = self.indexes.groupby(by='user', as_index=False)\
            .apply(lambda x: x.sample(frac=self.frac)).sample(frac=1.0)
        return self.sample_idx['prow'].values

    def _get_neg_sample_index(self):
        return self.sample_idx['nrow'].values


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
