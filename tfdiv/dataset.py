from collections import defaultdict

from scipy.sparse import isspmatrix_csr
from abc import abstractmethod, ABC
from tfdiv.utility import sparse_repr
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
        pos, neg = self.sampler.sample()
        n_samples = self.sampler.size
        batch_size = self._get_batch_size(n_samples)

        for i in range(0, n_samples, batch_size):
            upper_bound = min(i + batch_size, n_samples)
            if neg is None:
                yield pos[i:upper_bound]
            else:
                yield pos[i:upper_bound], neg[i:upper_bound]

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

    def __init__(self):
        self.size = None
        self.pos = None
        self.neg = None

    def get_index(self):
        poo = np.transpose(self.pos.nonzero())

        pdf = pd.DataFrame(poo, columns=['row', 'col']) \
            .groupby('row', as_index=False).min()

        def get_dict(df):
            dic = defaultdict(list)
            for k, v in df.groupby('col'):
                dic[k].extend(list(v['row'].values))
            return dic

        pdict = get_dict(pdf)
        pdf = pd.DataFrame(pdict, columns=['user', 'prow'])
        ind = pdf

        if self.neg is None:
            noo = np.transpose(self.neg.nonzero())
            ndf = pd.DataFrame(noo, columns=['row', 'col']) \
                .groupby('row', as_index=False).min()
            ndict = get_dict(ndf)
            ndf = pd.DataFrame(ndict, columns=['user', 'nrow'])
            ind = pd.merge(pdf, ndf, on="user")
        return ind

    @abstractmethod
    def _get_pos_sample_index(self):
        pass

    @abstractmethod
    def _get_neg_sample_index(self):
        pass

    def sample(self):
        sample_pos_idx = self._get_pos_sample_index()
        self.size = sample_pos_idx.shape[0]

        pos_samples = self.pos[sample_pos_idx]
        pos_samples.sort_indices()

        if self.neg is not None:
            sample_neg_idx = self._get_neg_sample_index()
            neg_samples = self.neg[sample_neg_idx]
            neg_samples.sort_indices()
        else:
            neg_samples = None
        return pos_samples, neg_samples


class NoSample(Sampler):

    def __init__(self, pos, neg=None):
        super(NoSample, self).__init__()
        self.pos = pos
        self.neg = neg
        self.size, _ = self.pos.shape
        self.indexes = self.get_index()

    def _get_pos_sample_index(self):
        return self.indexes['prow'].values

    def _get_neg_sample_index(self):
        return self.indexes['nrow'].values


class RandomSampler(Sampler):

    def __init__(self, pos, neg=None,
                 frac=0.5, ntype=np.float32):
        super(RandomSampler, self).__init__()
        if neg is not None:
            assert pos.shape == neg.shape, \
                "positive and negative sample-sets" \
                "must have the same dimensions"

        self.pos = pos
        self.neg = neg
        self.ntype = ntype
        n_samples, _ = pos.shape
        self.size = int(n_samples * frac)
        self.indexes = self.get_index()

    def _get_sample_index(self):
        self.sample_idx = self.indexes.sample(self.size)
        return self.sample_idx['prow']

    def _get_pos_sample_index(self):
        return self.sample_idx['nrow']


class UniformUserSampler(Sampler):

    def __init__(self, pos, neg=None,
                 frac=0.5, ntype=np.float32):
        super(UniformUserSampler, self).__init__()
        if neg is not None:
            assert pos.shape == neg.shape, \
                "positive and negative sample-sets" \
                "must have the same dimensions"

        self.pos = pos
        self.neg = neg
        self.ntype = ntype
        self.frac = frac
        self.indexes = self.get_index()

    def _get_pos_sample_index(self):
        self.sample_idx = self.indexes.groupby(by='user', as_index=False)\
            .apply(lambda x: x.sample(frac=self.frac)).sample(frac=1)
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
