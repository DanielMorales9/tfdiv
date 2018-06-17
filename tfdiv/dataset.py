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


class Sampler(ABC):

    def __init__(self):
        self.size = None
        self.pos = None
        self.neg = None

    @abstractmethod
    def _get_sample_index(self):
        pass

    def sample(self):
        sample_idx = self._get_sample_index()
        self.size = sample_idx.shape[0]

        pos_samples = self.pos[sample_idx]
        pos_samples.sort_indices()

        if self.neg is not None:
            neg_samples = self.neg[sample_idx]
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

    def _get_sample_index(self):
        sample_idx = np.arange(self.size)
        return sample_idx


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
        self.idx = np.arange(n_samples)
        self.size = int(n_samples * frac)

    def _get_sample_index(self):
        sample_idx = np.random.choice(self.idx, size=self.size)
        return sample_idx


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
        coo = pos.tocoo()
        idx = pd.DataFrame()
        idx['index'] = coo.row
        idx['user'] = coo.col

        idx = idx.groupby('index', as_index=False).min()
        self.idx = idx
        self.frac = frac

    def _get_sample_index(self):
        sample_idx = self.idx.groupby(by='user',
                                      as_index=False) \
            .apply(lambda x: x.sample(frac=self.frac))['index'].values
        np.random.shuffle(sample_idx)
        return sample_idx

