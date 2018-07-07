from tfdiv.graph import PointwiseGraph, BPRLFPGraph, \
    BayesianPersonalizedRankingGraph as BPRGraph, PointwiseRankingGraph
from tfdiv.utility import sparse_repr, loss_logistic, \
    matrix_swap_at_k
from sklearn.base import BaseEstimator, ClassifierMixin
from tfdiv.graph import PointwiseLFPGraph
from tfdiv.dataset import PairDataset, SimpleDataset
from abc import abstractmethod
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import warnings


class BaseClassifier(BaseEstimator, ClassifierMixin):
    """
    Abstract Base Classifier implements sklearn.base.BaseEstimator

    Parameters
    ------
    epochs: int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size: int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    dtype: `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    seed: int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress: bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir: string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    """
    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 dtype=tf.float32,
                 seed=1,
                 show_progress=True,
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_factors=10,
                 n_iter_no_change=10):
        self.train = None
        self.seed = seed
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        self.session_config = session_config
        self.session = tf.Session(config=self.session_config,
                                  graph=self.graph)

        self.dtype = dtype
        self.n_factors = n_factors
        self.ntype = np.float32 if dtype is tf.float32 else np.float64
        self.epochs = epochs
        self.show_progress = show_progress
        self.batch_size = batch_size

        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self._stopping = self.tol is not None
        self._best_loss = np.inf
        self._no_improvement = 0

        self.log_dir = log_dir
        self.logging_enabled = log_dir is not None
        self.log_writer = None
        self.core = None
        self.n_features = None

    @property
    def intercept(self):
        return self.session.run(self.core.bias)

    @property
    def weights(self):
        return self.session.run(self.core.weights)

    @property
    def params(self):
        return self.session.run(self.core.params)

    @abstractmethod
    def init_computational_graph(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_dataset(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_input(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):
        pass

    @abstractmethod
    def init_core(self, core):
        pass

    def score(self, X, y=None, sample_weight=None):
        pass

    def _update_no_improvement_count(self, acc_loss):
        if self._stopping:
            if acc_loss > self._best_loss - self.tol:
                self._no_improvement += 1
            else:
                self._no_improvement = 0
            if acc_loss < self._best_loss:
                self._best_loss = acc_loss

    def save_state(self, path):
        self.core.saver.save(self.session, path)

    def load_state(self, path):
        if self.graph is None:
            self.init_session()
        with self.graph.as_default():
            self.init_computational_graph()
        self.core.saver.restore(self.session, path)

    def destroy(self):
        """Terminates session and destroys graph."""
        self.session.close()
        self.session = None
        self.graph = None

    def init_session(self):
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        self.session = tf.Session(config=self.session_config,
                                  graph=self.graph)
        self.init_core(None)

    def log_summary(self, summary, step):
        if self.log_writer is None and self.logging_enabled:
            self.log_writer = tf.summary.FileWriter(self.log_dir, self.graph)
        if self.logging_enabled:
            self.log_writer.add_summary(summary, step)
            self.log_writer.flush()

    def log_graph(self):
        if self.logging_enabled:
            self.log_writer.add_graph(self.graph)


class Pointwise(BaseClassifier):
    """
    Abstract Pointwise Classifier BaseClassifier

    Parameters
    ------
    epochs : int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size : int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    n_factors : int, optional (Default 10)
        the number of factors used to factorize
        pairwise interactions between variables.
    dtype : `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    init_std : float, optional (Default 0.01)
        The standard deviation with which initialize model parameters.
    loss_function : tf.losses, tensorflow function, optional (Default tf.losses.mean_squared_error).
        The loss function to minimize while training.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    seed : int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress : bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir : string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    core : ``tfdiv.graph``, optional (Default None)
        Computational Graph
    """
    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 loss_function=tf.losses.mean_squared_error,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 show_progress=True,
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10,
                 seed=1,
                 core=None):
        super(Pointwise, self).__init__(n_factors=n_factors,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        show_progress=show_progress,
                                        seed=seed,
                                        log_dir=log_dir,
                                        session_config=session_config,
                                        n_iter_no_change=n_iter_no_change,
                                        tol=tol)
        self.init_std = init_std
        self.dtype = dtype
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.opt_kwargs = opt_kwargs
        self.train = None

        self.init_core(core)

    def init_core(self, core):
        # Computational graph initialization
        self.core = core if core else PointwiseGraph(n_factors=self.n_factors,
                                                     init_std=self.init_std,
                                                     dtype=self.dtype,
                                                     opt_kwargs=self.opt_kwargs,
                                                     optimizer=self.optimizer,
                                                     learning_rate=self.learning_rate,
                                                     loss_function=self.loss_function,
                                                     l2_v=self.l2_v,
                                                     l2_w=self.l2_w)

    def init_input(self, x, y=None):
        if not x.has_sorted_indices:
            x.sort_indices()
        n_samples, self.n_features = x.shape
        if y is not None:
            return x, y, n_samples
        return x, n_samples

    def init_dataset(self, x, y=None):
        dataset = SimpleDataset(x, y=y, ntype=self.ntype) \
            .batch(self.batch_size)
        if y is not None:
            dataset.shuffle(True)
        return dataset

    def init_computational_graph(self):
        self.core.define_graph()
        self.core.init_saver()
        self.train = True

    def fit(self, X, y=None):
        with self.graph.as_default():
            x, y, n_samples = self.init_input(X, y)
            dataset = self.init_dataset(x, y)
            if self.train is None:
                self.init_computational_graph()

        if not self.session.run(tf.is_variable_initialized(self.core.global_step)):
            self.session.run(self.core.init_all_vars, feed_dict={self.core.n_features: self.n_features})

        ops = self.core.ops
        for epoch in tqdm(range(self.epochs),
                          unit='epochs',
                          disable=not self.show_progress):
            loss = 0.0
            for x, y in dataset.get_next():
                fd = dataset.batch_to_feed_dict(x, y, self.core)
                _, summary, step, batch_loss = self.session.run(ops, feed_dict=fd)
                self.log_summary(summary, step)
                loss += batch_loss

            loss /= n_samples

            self._update_no_improvement_count(loss)

            if self._stopping and self._no_improvement > self.n_iter_no_change:
                warnings.warn("Stopping at epoch: %s with loss %s" % (epoch, loss))
                break

    def predict(self, X):
        self.train = False
        with self.graph.as_default():
            x, n_samples = self.init_input(X)
            dataset = self.init_dataset(x)

        results = []
        for x in dataset.get_next():
            fd = dataset.batch_to_feed_dict(x, core=self.core)
            res = self.session.run(self.core.y_hat, feed_dict=fd)
            results.append(res)

        results = np.concatenate(results).reshape(-1)
        return self.decision_function(results)

    @abstractmethod
    def decision_function(self, x):
        pass


class Regression(Pointwise):

    """
    Regression Module

    Parameters
    ------
    epochs : int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size : int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    n_factors : int, optional (Default 10)
        the number of factors used to factorize
        pairwise interactions between variables.
    dtype : `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    init_std : float, optional (Default 0.01)
        The standard deviation with which initialize model parameters.
    loss_function : ``tf.losses``, tensorflow function, optional (Default tf.losses.mean_squared_error).
        The loss function to minimize while training.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    seed : int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress : bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir : string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    core : ``tfdiv.graph``, optional (Default None)
        Computational Graph
    """

    def __init__(self,
                 loss_function=tf.losses.mean_squared_error,
                 epochs=100,
                 batch_size=-1,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 show_progress=True,
                 log_dir=None,
                 opt_kwargs=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10,
                 seed=1,
                 core=None):
        super(Regression, self).__init__(epochs=epochs,
                                         loss_function=loss_function,
                                         n_factors=n_factors,
                                         dtype=dtype,
                                         init_std=init_std,
                                         l2_w=l2_w,
                                         l2_v=l2_v,
                                         opt_kwargs=opt_kwargs,
                                         learning_rate=learning_rate,
                                         batch_size=batch_size,
                                         show_progress=show_progress,
                                         optimizer=optimizer,
                                         seed=seed,
                                         log_dir=log_dir,
                                         session_config=session_config,
                                         n_iter_no_change=n_iter_no_change,
                                         tol=tol,
                                         core=core)

    def decision_function(self, x):
        return x

    def score(self, X, y=None, sample_weight=None):
        pass


class Classification(Pointwise):
    """
    Binary Classification Module

    Parameters
    ------
    epochs : int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size : int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    label_transform: function, optional
        Function that transforms y labels to a value interval
        that better suits the passed loss_function.
    n_factors : int, optional (Default 10)
        the number of factors used to factorize
        pairwise interactions between variables.
    dtype : `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    init_std : float, optional (Default 0.01)
        The standard deviation with which initialize model parameters.
    loss_function : ``tf.losses``, tensorflow function, optional (Default ``tfdiv.utility.loss_logistic``).
        The loss function to minimize while training.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    seed : int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress : bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir : string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    core : ``tfdiv.graph``, optional (Default None)
        Computational Graph
        """

    def __init__(self,
                 loss_function=loss_logistic,
                 label_transform=lambda y: y * 2 - 1,
                 epochs=100,
                 batch_size=-1,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 show_progress=True,
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10,
                 seed=1,
                 core=None):
        super(Classification, self).__init__(epochs=epochs,
                                             loss_function=loss_function,
                                             n_factors=n_factors,
                                             dtype=dtype,
                                             init_std=init_std,
                                             l2_w=l2_w,
                                             l2_v=l2_v,
                                             opt_kwargs=opt_kwargs,
                                             learning_rate=learning_rate,
                                             batch_size=batch_size,
                                             show_progress=show_progress,
                                             optimizer=optimizer,
                                             seed=seed,
                                             log_dir=log_dir,
                                             session_config=session_config,
                                             n_iter_no_change=n_iter_no_change,
                                             tol=tol,
                                             core=core)
        self.label_transform = label_transform

    def decision_function(self, x):
        return (x > 0).astype(int)

    def score(self, X, y=None, sample_weight=None):
        pass


class Ranking(BaseClassifier):

    """
    Abstract Ranking Module.
    """
    def init_computational_graph(self):
        self.core.define_graph()
        self.core.ranking_computation()
        self.core.init_saver()
        self.train = True

    def predict(self, X, n_users, n_items, k=10):
        raise NotImplementedError


class RegressionRanking(Regression, Ranking):
    """
    Pointwise Ranking Module implemented with a regression classifier

    Parameters
    ------
    epochs : int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size : int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    n_factors : int, optional (Default 10)
        the number of factors used to factorize
        pairwise interactions between variables.
    dtype : `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    init_std : float, optional (Default 0.01)
        The standard deviation with which initialize model parameters.
    loss_function : ``tf.losses``, tensorflow function, optional (Default tf.losses.mean_squared_error).
        The loss function to minimize while training.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    seed : int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress : bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir : string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    core : ``tfdiv.graph``, optional (Default None)
        Computational Graph
    """

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 loss_function=tf.losses.mean_squared_error,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 seed=1,
                 show_progress=True,
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10,
                 core=None):
        # Computational graph initialization
        self.n_factors = n_factors
        self.init_std = init_std
        self.dtype = dtype
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2_v = l2_v
        self.l2_w = l2_w
        self.opt_kwargs=opt_kwargs
        self.init_core(core)

        super(RegressionRanking, self).__init__(epochs=epochs,
                                                batch_size=batch_size,
                                                n_factors=n_factors,
                                                dtype=dtype,
                                                init_std=init_std,
                                                loss_function=loss_function,
                                                l2_w=l2_w,
                                                l2_v=l2_v,
                                                learning_rate=learning_rate,
                                                optimizer=optimizer,
                                                seed=seed,
                                                show_progress=show_progress,
                                                log_dir=log_dir,
                                                session_config=session_config,
                                                n_iter_no_change=n_iter_no_change,
                                                tol=tol, core=self.core)

    def init_core(self, core):
        self.core = core if core \
            else PointwiseRankingGraph(n_factors=self.n_factors,
                                       init_std=self.init_std,
                                       dtype=self.dtype,
                                       opt_kwargs=self.opt_kwargs,
                                       optimizer=self.optimizer,
                                       learning_rate=self.learning_rate,
                                       l2_v=self.l2_v,
                                       l2_w=self.l2_w)

    def init_computational_graph(self):
        Ranking.init_computational_graph(self)

    def predict(self, X, n_users, n_items, k=10):
        pred = Regression.predict(self, X)
        rank_res = self.session.run(self.core.ranking_results,
                                    feed_dict={self.core.pred: pred,
                                               self.core.n_users: n_users,
                                               self.core.n_items: n_items,
                                               self.core.k: k})
        return rank_res


class ClassificationRanking(Classification, Ranking):
    """
    Pointwise Ranking Module implemented as a Classification classifier

    Parameters
    ------
    epochs : int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size : int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    n_factors : int, optional (Default 10)
        the number of factors used to factorize
        pairwise interactions between variables.
    dtype : `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    init_std : float, optional (Default 0.01)
        The standard deviation with which initialize model parameters.
    loss_function : ``tf.losses``, tensorflow function, optional (Default tf.losses.mean_squared_error).
        The loss function to minimize while training.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    seed : int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress : bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir : string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    core : ``tfdiv.graph``, optional (Default None)
        Computational Graph
    """

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 loss_function=loss_logistic,
                 label_transform=lambda y: y * 2 - 1,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 seed=1,
                 show_progress=True,
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10,
                 core=None):
        self.n_factors = n_factors
        self.init_std = init_std
        self.dtype = dtype
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2_v = l2_v
        self.l2_w = l2_w
        self.opt_kwargs = opt_kwargs
        self.init_core(core)

        super(ClassificationRanking, self).__init__(epochs=epochs,
                                                    label_transform=label_transform,
                                                    loss_function=loss_function,
                                                    l2_w=l2_w,
                                                    l2_v=l2_v,
                                                    learning_rate=learning_rate,
                                                    batch_size=batch_size,
                                                    n_factors=n_factors,
                                                    dtype=dtype,
                                                    optimizer=optimizer,
                                                    seed=seed,
                                                    show_progress=show_progress,
                                                    log_dir=log_dir,
                                                    session_config=session_config,
                                                    n_iter_no_change=n_iter_no_change,
                                                    tol=tol,
                                                    core=self.core)

    def init_core(self, core):
        self.core = core if core \
            else PointwiseRankingGraph(n_factors=self.n_factors,
                                       init_std=self.init_std,
                                       opt_kwargs=self.opt_kwargs,
                                       dtype=self.dtype,
                                       optimizer=self.optimizer,
                                       learning_rate=self.learning_rate,
                                       l2_v=self.l2_v,
                                       l2_w=self.l2_w)

    def decision_function(self, x):
        return x

    def init_computational_graph(self):
        Ranking.init_computational_graph(self)

    def predict(self, X, n_users, n_items, k=10):
        pred = Classification.predict(self, X)
        rank_res = self.session.run(self.core.ranking_results,
                                    feed_dict={self.core.pred: pred,
                                               self.core.n_users: n_users,
                                               self.core.n_items: n_items,
                                               self.core.k: k})
        return rank_res


class BayesianPersonalizedRanking(Ranking):
    """
    Bayesian Personalized Ranking a Pairwise Learning-to-Rank solution.

    Parameters
    ------
    epochs : int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size : int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    n_factors : int, optional (Default 10)
        the number of factors used to factorize
        pairwise interactions between variables.
    dtype : `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    init_std : float, optional (Default 0.01)
        The standard deviation with which initialize model parameters.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    seed : int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress : bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir : string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    core : ``tfdiv.graph``, optional (Default None)
        Computational Graph
    """

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 seed=1,
                 frac=0.5,
                 show_progress=True,
                 bootstrap_sampling='uniform_user',
                 log_dir=None,
                 session_config=None,
                 n_threads=2,
                 shuffle_size=100,
                 tol=None,
                 n_iter_no_change=10,
                 core=None):
        super(BayesianPersonalizedRanking, self).__init__(epochs=epochs,
                                                          batch_size=batch_size,
                                                          dtype=dtype,
                                                          n_factors=n_factors,
                                                          seed=seed,
                                                          show_progress=show_progress,
                                                          log_dir=log_dir,
                                                          session_config=session_config,
                                                          n_iter_no_change=n_iter_no_change,
                                                          tol=tol)
        self.frac = frac
        self.bootstrap_sampling = bootstrap_sampling
        self.init_std = init_std
        self.dtype = dtype
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.shuffle_size = shuffle_size
        self.n_threads = n_threads
        self.opt_kwargs = opt_kwargs
        # Computational graph initialization
        self.init_core(core)

    def init_core(self, core):
        self.core = core if core \
            else BPRGraph(n_factors=self.n_factors,
                          init_std=self.init_std,
                          opt_kwargs=self.opt_kwargs,
                          dtype=self.dtype,
                          optimizer=self.optimizer,
                          learning_rate=self.learning_rate,
                          l2_v=self.l2_v,
                          l2_w=self.l2_w)

    def init_input(self, pos, neg=None):
        if not pos.has_sorted_indices:
            pos.sort_indices()
        if neg is not None:
            if not neg.has_sorted_indices:
                neg.sort_indices()
            _, self.n_features = pos.shape
            return pos, neg
        return pos

    def init_dataset(self, pos, neg=None, bootstrap_sampling='no_sample'):
        dataset = PairDataset(pos, neg=neg,
                              frac=self.frac,
                              ntype=self.ntype,
                              shuffle_size=self.shuffle_size,
                              n_threads=self.n_threads,
                              bootstrap_sampling=bootstrap_sampling) \
            .batch(self.batch_size)
        return dataset

    def init_computational_graph(self):
        Ranking.init_computational_graph(self)

    def fit(self, pos, neg, *args):
        with self.graph.as_default():
            pos, neg = self.init_input(pos, neg)
            n_samples = pos.shape[0]
            dataset = self.init_dataset(pos, neg, self.bootstrap_sampling)
            if self.train is None:
                self.init_computational_graph()

        if not self.session.run(tf.is_variable_initialized(
                self.core.global_step)):
            self.session.run(self.core.init_all_vars,
                             feed_dict={self.core.n_features: self.n_features})

        ops = self.core.ops
        for epoch in tqdm(range(self.epochs), unit='epochs',
                          disable=not self.show_progress):
            loss = 0.0
            for pos, neg in dataset.get_next():
                fd = dataset.batch_to_feed_dict(pos, neg, self.core)
                _, summary, step, batch_loss = self.session.run(ops, feed_dict=fd)
                self.log_summary(summary, step)
                loss += batch_loss
            loss /= n_samples

            self._update_no_improvement_count(loss)

            if self._stopping and self._no_improvement > self.n_iter_no_change:
                warnings.warn("Stopping at epoch: %s with loss %s" % (epoch, loss))
                break

    def predict(self, x, n_users, n_items, k=10):
        with self.graph.as_default():
            x = self.init_input(x)
            dataset = self.init_dataset(x)

        ops = self.core.y_hat
        results = []
        for pos in dataset.get_next():
            feed_dict = dataset.batch_to_feed_dict(pos, core=self.core)
            hat = self.session.run(ops, feed_dict=feed_dict)
            results.append(hat)

        pred = np.concatenate(results).reshape(-1)
        rank_res = self.session.run(self.core.ranking_results,
                                    feed_dict={self.core.pred: pred,
                                               self.core.n_users: n_users,
                                               self.core.n_items: n_items,
                                               self.core.k: k})
        return rank_res


class LatentFactorPortfolio(Ranking):
    """
    Abstract Latent Factor Portfolio Class.

    Parameters
    ------
    epochs : int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size : int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    dtype : `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    seed : int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress : bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir : string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    """

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 n_factors=10,
                 dtype=tf.float32,
                 seed=1,
                 show_progress=True,
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10):
        super(LatentFactorPortfolio, self).__init__(epochs=epochs,
                                                    batch_size=batch_size,
                                                    n_factors=n_factors,
                                                    dtype=dtype,
                                                    seed=seed,
                                                    show_progress=show_progress,
                                                    log_dir=log_dir,
                                                    session_config=session_config,
                                                    n_iter_no_change=n_iter_no_change,
                                                    tol=tol)

    @property
    def variance(self):
        return self.session.run(self.core.variance)

    def fit(self, X, y, n_users, n_items):
        raise NotImplementedError

    def predict(self, X, n_users, n_items, k=10, b=0.0):
        raise NotImplementedError

    def init_computational_graph(self):
        self.core.define_graph()
        self.core.ranking_computation()
        self.core.variance_estimate()
        self.core.delta_f_computation()
        self.core.init_saver()
        self.train = True

    def compute_variance(self, indices, values, shape, n_users):
        self.session.run(self.core.init_variance_vars, feed_dict={
            self.core.n_users: n_users
        })
        self.session.run(self.core.variance_op,
                         feed_dict={self.core.x: (indices,
                                                  values,
                                                  shape),
                                    self.core.n_users: n_users})

    def delta_predict(self, k, b, n_users, pred, rank, X):
        sparse_x = sparse_repr(X, self.ntype)

        def parametric_feed_dict(this, pred, rank, i):
            return {
                this.core.x: sparse_x,
                this.core.predictions: pred,
                this.core.rankings: rank,
                this.core.k: i,
                this.core.b: b,
                this.core.n_users: n_users,
            }

        for i in tqdm(range(1, k), unit='k',
                      disable=not self.show_progress):
            delta_f = self.session.run(self.core.delta_f,
                                       feed_dict=parametric_feed_dict(self, pred, rank, i))
            delta_arg_max = np.argmax(delta_f, axis=1)
            matrix_swap_at_k(delta_arg_max, k, pred)
            matrix_swap_at_k(delta_arg_max, k, rank)
        return rank[:, :k]


class RegressionLFP(RegressionRanking, LatentFactorPortfolio):
    """
    Regression version of the Latent Factor Portfolio (LFP)

    Parameters
    ------
    epochs : int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size : int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    n_factors : int, optional (Default 10)
        the number of factors used to factorize
        pairwise interactions between variables.
    dtype : `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    init_std : float, optional (Default 0.01)
        The standard deviation with which initialize model parameters.
    loss_function : ``tf.losses``, tensorflow function, optional (Default tf.losses.mean_squared_error).
        The loss function to minimize while training.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    seed : int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress : bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir : string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    core : ``tfdiv.graph``, optional (Default None)
        Computational Graph
    """

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 loss_function=tf.losses.mean_squared_error,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 seed=1,
                 show_progress=True,
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10,
                 core=None):

        self.n_factors = n_factors
        self.init_std = init_std
        self.dtype = dtype
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2_v = l2_v
        self.l2_w = l2_w
        self.opt_kwargs = opt_kwargs
        self.init_core(core)

        super(RegressionLFP, self).__init__(epochs=epochs,
                                            batch_size=batch_size,
                                            n_factors=n_factors,
                                            dtype=dtype,
                                            init_std=init_std,
                                            loss_function=loss_function,
                                            l2_w=l2_w,
                                            l2_v=l2_v,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            seed=seed,
                                            show_progress=show_progress,
                                            log_dir=log_dir,
                                            session_config=session_config,
                                            n_iter_no_change=n_iter_no_change,
                                            tol=tol,
                                            core=self.core)

    def init_core(self, core):
        self.core = core if core \
            else PointwiseLFPGraph(n_factors=self.n_factors,
                                   init_std=self.init_std,
                                   dtype=self.dtype,
                                   opt_kwargs=self.opt_kwargs,
                                   optimizer=self.optimizer,
                                   learning_rate=self.learning_rate,
                                   l2_v=self.l2_v,
                                   l2_w=self.l2_w)

    def delta_predict(self, k, b, n_users, pred, rank, X):
        x, n_samples = self.init_input(X)
        dataset = SimpleDataset(x, ntype=self.ntype)

        core = self.core
        for i in tqdm(range(1, k),
                      unit='k',
                      disable=not self.show_progress):
            for x in dataset.get_next():
                fd = dataset.batch_to_feed_dict(x, core=core)
                fd[core.predictions] = pred
                fd[core.rankings] = rank
                fd[core.k] = i
                fd[core.b] = b
                fd[core.n_users] = n_users
                delta_f = self.session.run(self.core.delta_f,
                                           feed_dict=fd)
                delta_arg_max = np.argmax(delta_f, axis=1)
                matrix_swap_at_k(delta_arg_max, k, pred)
                matrix_swap_at_k(delta_arg_max, k, rank)
                matrix_swap_at_k(delta_arg_max, k, rank)
        return rank[:, :k]

    def init_computational_graph(self):
        LatentFactorPortfolio.init_computational_graph(self)

    def fit(self, X, y, n_users, n_items):
        RegressionRanking.fit(self, X, y)
        indices, values, shape = sparse_repr(X, self.ntype)
        self.compute_variance(indices, values, shape, n_users)

    def predict(self, X, n_users, n_items, k=10, b=0.0):
        pred, rank = RegressionRanking.predict(self, X, n_users, n_items, n_items)
        predictions = self.delta_predict(k, b, n_users, pred, rank, X)
        return predictions


class ClassificationLFP(ClassificationRanking, LatentFactorPortfolio):
    """
    Classification version of the Latent Factor Portfolio (LFP)

    Parameters
    ------
    epochs : int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size : int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    n_factors : int, optional (Default 10)
        the number of factors used to factorize
        pairwise interactions between variables.
    dtype : `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    init_std : float, optional (Default 0.01)
        The standard deviation with which initialize model parameters.
    loss_function : ``tf.losses``, tensorflow function, optional (Default tf.losses.mean_squared_error).
        The loss function to minimize while training.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    seed : int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress : bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir : string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    core : ``tfdiv.graph``, optional (Default None)
        Computational Graph
    """
    def __init__(self,
                 loss_function=loss_logistic,
                 label_transform=lambda y: y * 2 - 1,
                 epochs=100,
                 batch_size=-1,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 show_progress=True,
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10,
                 seed=1,
                 core=None):

        self.n_factors = n_factors
        self.init_std = init_std
        self.dtype = dtype
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2_v = l2_v
        self.l2_w = l2_w
        self.opt_kwargs = opt_kwargs
        self.init_core(core)
        super(ClassificationLFP, self).__init__(epochs=epochs,
                                                loss_function=loss_function,
                                                label_transform=label_transform,
                                                init_std=init_std,
                                                batch_size=batch_size,
                                                n_factors=n_factors,
                                                dtype=dtype,
                                                l2_w=l2_w,
                                                l2_v=l2_v,
                                                learning_rate=learning_rate,
                                                optimizer=optimizer,
                                                seed=seed,
                                                show_progress=show_progress,
                                                log_dir=log_dir,
                                                session_config=session_config,
                                                n_iter_no_change=n_iter_no_change,
                                                tol=tol,
                                                core=self.core)

    def init_core(self, core):
        self.core = core if core \
            else PointwiseLFPGraph(n_factors=self.n_factors,
                                   init_std=self.init_std,
                                   opt_kwargs=self.opt_kwargs,
                                   dtype=self.dtype,
                                   optimizer=self.optimizer,
                                   learning_rate=self.learning_rate,
                                   l2_v=self.l2_v,
                                   l2_w=self.l2_w)

    def delta_predict(self, k, b, n_users, pred, rank, X):
        x, n_samples = self.init_input(X)
        dataset = SimpleDataset(x, ntype=self.ntype)

        core = self.core
        for i in tqdm(range(1, k),
                      unit='k',
                      disable=not self.show_progress):
            for x in dataset.get_next():
                fd = dataset.batch_to_feed_dict(x, core=core)
                fd[core.predictions] = pred
                fd[core.rankings] = rank
                fd[core.k] = i
                fd[core.b] = b
                fd[core.n_users] = n_users
                delta_f = self.session.run(self.core.delta_f,
                                           feed_dict=fd)
                delta_arg_max = np.argmax(delta_f, axis=1)
                matrix_swap_at_k(delta_arg_max, k, pred)
                matrix_swap_at_k(delta_arg_max, k, rank)
                matrix_swap_at_k(delta_arg_max, k, rank)
        return rank[:, :k]

    def init_computational_graph(self):
        LatentFactorPortfolio.init_computational_graph(self)

    def fit(self, X, y, n_users, n_items):
        ClassificationRanking.fit(self, X, y)
        indices, values, shape = sparse_repr(X, self.ntype)
        self.compute_variance(indices, values, shape, n_users)

    def predict(self, X, n_users, n_items, k=10, b=0.0):
        pred, rank = ClassificationRanking \
            .predict(self, X, n_users, n_items, n_items)
        predictions = self.delta_predict(k, b, n_users, pred, rank, X)
        return predictions


class BayesianPersonalizedRankingLFP(BayesianPersonalizedRanking, LatentFactorPortfolio):
    """
    Bayesian Personalized Ranking version of the Latent Factor Portfolio (LFP)

    Parameters
    ------
    epochs : int, optional
        number of training cycles to perform in order to fit classifier
        Default 100 epochs
    batch_size : int, optional (Default -1)
        Batch size to use while training classifier.
        -1 means no batch_size to use.
    n_factors : int, optional (Default 10)
        the number of factors used to factorize
        pairwise interactions between variables.
    dtype : `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    init_std : float, optional (Default 0.01)
        The standard deviation with which initialize model parameters.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    seed : int, optional (Default 1).
        Seed integer for `tf.Graph`.
    show_progress : bool, optional (Default True).
        Enables progress bar during fitting.
    log_dir : string, optional (Default None)
        Tensorflow logging directory.
    tol : float, optional (Default None)
        Tolerance for stopping criteria.
    n_iter_no_change : int, optional (Default 10)
        Maximum number of epochs to not meet ``tol`` improvement.
    core : ``tfdiv.graph``, optional (Default None)
        Computational Graph
    """

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 seed=1,
                 frac=0.5,
                 show_progress=True,
                 bootstrap_sampling='uniform_user',
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 shuffle_size=1000,
                 n_threads=10,
                 n_iter_no_change=10,
                 core=None):

        self.n_factors = n_factors
        self.init_std = init_std
        self.dtype = dtype
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2_v = l2_v
        self.l2_w = l2_w
        self.shuffle_size = shuffle_size
        self.n_threads = n_threads
        self.opt_kwargs = opt_kwargs
        self.init_core(core)
        super(BayesianPersonalizedRankingLFP, self).__init__(epochs=epochs,
                                                             batch_size=batch_size,
                                                             frac=frac,
                                                             bootstrap_sampling=bootstrap_sampling,
                                                             n_factors=n_factors,
                                                             dtype=dtype,
                                                             init_std=init_std,
                                                             l2_w=l2_w,
                                                             l2_v=l2_v,
                                                             learning_rate=learning_rate,
                                                             optimizer=optimizer,
                                                             seed=seed,
                                                             show_progress=show_progress,
                                                             log_dir=log_dir,
                                                             session_config=session_config,
                                                             n_iter_no_change=n_iter_no_change,
                                                             tol=tol,
                                                             core=self.core)

    def init_core(self, core):
        self.core = core if core \
            else BPRLFPGraph(n_factors=self.n_factors,
                             init_std=self.init_std,
                             dtype=self.dtype,
                             opt_kwargs=self.opt_kwargs,
                             optimizer=self.optimizer,
                             learning_rate=self.learning_rate,
                             l2_v=self.l2_v,
                             l2_w=self.l2_w)

    def init_computational_graph(self):
        LatentFactorPortfolio.init_computational_graph(self)

    def fit(self, X, y, n_users, n_items):
        BayesianPersonalizedRanking.fit(self, X, y)
        indices, values, shape = sparse_repr(X, self.ntype)
        self.compute_variance(indices, values, shape, n_users)

    def predict(self, X, n_users, n_items, k=10, b=0.0):
        pred, rank = BayesianPersonalizedRanking.predict(self, X, n_users, n_items, n_items)
        predictions = self.delta_predict(k, b, n_users, pred, rank, X)
        return predictions
