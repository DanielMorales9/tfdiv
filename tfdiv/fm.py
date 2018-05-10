from tfdiv.graph import PointwiseGraph, BPRLFPGraph, \
    BayesianPersonalizedRankingGraph as BPRGraph
from tfdiv.utility import sparse_repr, loss_logistic, \
    num_of_users_from_indices, unique_sparse_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from tfdiv.graph import PointwiseLFPGraph
from tfdiv.dataset import PairDataset
from abc import abstractmethod
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import warnings


class BaseClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 shuffle_size=1000,
                 dtype=tf.float32,
                 seed=1,
                 show_progress=True,
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10):
        self.seed = seed
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        self.session_config = session_config
        self.session = tf.Session(config=self.session_config,
                                  graph=self.graph)

        self.dtype = dtype
        self.ntype = np.float32 if dtype is tf.float32 else np.float64
        self.epochs = epochs
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size

        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self._stopping = self.tol is not None
        self._best_loss = np.inf
        self._no_improvement = 0

        self.log_dir = log_dir
        self.logging_enabled = log_dir is not None

        if self.logging_enabled:
            self.summary_writer = tf.summary.FileWriter(self.log_dir,
                                                        self.graph)

        self.handle = None
        self.core = None
        self.n_features = None

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
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
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
        raise NotImplementedError

    def load_state(self, path):
        raise NotImplementedError

    def log_summary(self, summary, step):
        if self.logging_enabled:
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()


class FMPointwise(BaseClassifier):

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 shuffle_size=1000,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 loss_function=tf.losses.mean_squared_error,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 seed=1,
                 show_progress=True,
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10,
                 cmp_graph=PointwiseGraph):
        super(FMPointwise, self).__init__(epochs=epochs,
                                          batch_size=batch_size,
                                          shuffle_size=shuffle_size,
                                          show_progress=show_progress,
                                          seed=seed,
                                          log_dir=log_dir,
                                          session_config=session_config,
                                          n_iter_no_change=n_iter_no_change,
                                          tol=tol)
        self.n_factors = n_factors
        self.init_std = init_std
        self.dtype = dtype
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.l2_w = l2_w
        self.l2_v = l2_v

        assert issubclass(cmp_graph, PointwiseGraph), \
            'Computational Graph must be a subclass of PointwiseGraph'
        # Computational graph initialization
        self.core = cmp_graph(n_factors=self.n_factors,
                              init_std=self.init_std,
                              dtype=self.dtype,
                              optimizer=self.optimizer,
                              learning_rate=self.learning_rate,
                              loss_function=self.loss_function,
                              l2_v=self.l2_v,
                              l2_w=self.l2_w)

    def init_computational_graph(self, it):
        if it:
            next_x, next_y = it.get_next()
            self.core.set_params(**{'x': next_x, 'y': next_y})
            self.core.define_graph()

    def init_iterator(self, dataset):
        iterator = None
        if self.handle is None:
            self.handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                self.handle,
                output_types=dataset.output_types,
                output_shapes=dataset.output_shapes,
                output_classes=dataset.output_classes)
        return iterator

    def init_dataset(self, x_d, y_d, n_samples, train):
        batch_size = n_samples if self.batch_size == -1 \
            else self.batch_size
        dataset = tf.data.Dataset.from_tensor_slices((x_d, y_d)) \
            .batch(batch_size)
        if train:
            dataset = dataset.shuffle(self.shuffle_size)

        iterator = dataset.make_initializable_iterator()
        return dataset, iterator

    def fit(self, X, y=None):

        with self.graph.as_default():
            input_vars = self.init_input(X, y)
            dataset, train_iterator = self.init_dataset(*input_vars)
            it = self.init_iterator(dataset)
            self.init_computational_graph(it)
            n_samples = input_vars[2]

        if not self.session.run(tf.is_variable_initialized(
                self.core.global_step)):
            self.session.run(self.core.init_all_vars,
                             feed_dict={self.core.n_features: self.n_features})

        ops = self.core.ops
        train_handle = self.session.run(train_iterator.string_handle())
        fd = {self.handle: train_handle}
        for epoch in tqdm(range(self.epochs),
                          unit='epochs',
                          disable=not self.show_progress):
            self.session.run(train_iterator.initializer)
            loss = 0.0
            while True:
                try:
                    _, summary, step, batch_loss = self.session.run(ops,
                                                                    feed_dict=fd)
                    self.log_summary(summary, step)
                    loss += batch_loss
                except tf.errors.OutOfRangeError:
                    break

            loss /= n_samples

            self._update_no_improvement_count(loss)

            if self._stopping and self._no_improvement > self.n_iter_no_change:
                warnings.warn("Stopping at epoch: %s with loss %s" % (epoch, loss))
                break

    def predict(self, X):
        with self.graph.as_default():
            input_vars = self.init_input(X)
            _, it = self.init_dataset(*input_vars)

        pred_handle = self.session.run(it.string_handle())
        results = []
        self.session.run(it.initializer)
        while True:
            try:
                res = self.session.run(self.core.y_hat,
                                       feed_dict={self.handle: pred_handle})
                results.append(res)
            except tf.errors.OutOfRangeError:
                break

        results = np.concatenate(results).reshape(-1)
        return self.decision_function(results)

    @abstractmethod
    def decision_function(self, x):
        pass


class FMRegression(FMPointwise):

    def __init__(self, loss_function=tf.losses.mean_squared_error, **kwargs):
        kwargs['loss_function'] = loss_function
        super(FMRegression, self).__init__(**kwargs)

    def init_input(self, x, y=None):
        if not x.has_sorted_indices:
            x.sort_indices()
        x_d = tf.SparseTensor(*sparse_repr(x, self.ntype))

        train = y is not None
        if train:
            n_samples, self.n_features = x.shape
            y_d = tf.convert_to_tensor(y, self.dtype)
        else:
            n_samples, _ = x.shape
            y_d = tf.zeros(x.shape[0], dtype=self.dtype)
        return x_d, y_d, n_samples, train

    def decision_function(self, x):
        return x

    def score(self, X, y=None, sample_weight=None):
        pass


class FMClassification(FMPointwise):

    def __init__(self, loss_function=loss_logistic,
                 label_transform=lambda y: y * 2 - 1,
                 **kwargs):
        kwargs['loss_function'] = loss_function
        self.label_transform = label_transform
        super(FMClassification, self).__init__(**kwargs)

    def init_input(self, x, y=None):
        if not x.has_sorted_indices:
            x.sort_indices()
        x_d = tf.SparseTensor(*sparse_repr(x, self.ntype))

        train = y is not None
        if train:
            if not (set(y) == {0, 1}):
                raise ValueError("Input labels must be in set {0,1}.")
            y = self.label_transform(y)
            n_samples, self.n_features = x.shape
            y_d = tf.convert_to_tensor(y, self.dtype)
        else:
            n_samples, _ = x.shape
            y_d = tf.zeros(x.shape[0], dtype=self.dtype)
        return x_d, y_d, n_samples, train

    def decision_function(self, x):
        return (x > 0).astype(int)

    def score(self, X, y=None, sample_weight=None):
        pass


class FMPairwiseRanking(BaseClassifier):

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 shuffle_size=1000,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer,
                 seed=1,
                 frac=0.5,
                 show_progress=True,
                 bootstrap_sampling='uniform_user',
                 log_dir=None,
                 session_config=None,
                 tol=None,
                 n_iter_no_change=10,
                 cmp_graph=BPRGraph):
        super(FMPairwiseRanking, self).__init__(epochs=epochs,
                                                batch_size=batch_size,
                                                shuffle_size=shuffle_size,
                                                show_progress=show_progress,
                                                seed=seed,
                                                log_dir=log_dir,
                                                n_iter_no_change=n_iter_no_change,
                                                tol=tol,
                                                session_config=session_config)
        self.frac = frac
        self.bootstrap_sampling = bootstrap_sampling
        self.n_factors = n_factors
        self.init_std = init_std
        self.dtype = dtype
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2_w = l2_w
        self.l2_v = l2_v
        # Computational graph initialization
        self.core = cmp_graph(n_factors=n_factors,
                              init_std=init_std,
                              dtype=dtype,
                              optimizer=optimizer,
                              learning_rate=learning_rate,
                              l2_v=l2_v,
                              l2_w=l2_w)

    def init_input(self, pos, neg=None):
        if not pos.has_sorted_indices:
            pos.sort_indices()
        if neg is not None:
            if not neg.has_sorted_indices:
                neg.sort_indices()
            _, self.n_features = pos.shape
            return pos, neg
        return pos

    def init_dataset(self, pos, neg=None,
                     bootstrap_sampling='no_sample'):
        dataset = PairDataset(pos, neg=neg,
                              frac=self.frac,
                              ntype=self.ntype,
                              bootstrap_sampling=bootstrap_sampling) \
            .batch(self.batch_size)
        return dataset

    def init_computational_graph(self):
        self.core.define_graph()

    def fit(self, pos, neg=None):

        with self.graph.as_default():
            pos, neg = self.init_input(pos, neg)
            n_samples = pos.shape[0]
            dataset = self.init_dataset(pos, neg, self.bootstrap_sampling)
            self.init_computational_graph()

        if not self.session.run(tf.is_variable_initialized(
                self.core.global_step)):
            self.session.run(self.core.init_all_vars)

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

    def predict(self, x):
        with self.graph.as_default():
            x = self.init_input(x)
            dataset = self.init_dataset(x)

        ops = self.core.y_hat
        results = []
        for pos in dataset.get_next():
            feed_dict = dataset.batch_to_feed_dict(pos, core=self.core)
            hat = self.session.run(ops, feed_dict=feed_dict)
            results.append(hat)
        return np.concatenate(results).reshape(-1)

    def score(self, X, y=None, sample_weight=None):
        pass


class LatentFactorPortfolio:

    def __init__(self):
        self.graph = None
        self.session = None
        self.core = None

    def compute_variance(self, X):
        indices, values, shape = sparse_repr(X, np.float32)
        n_users = num_of_users_from_indices(indices)
        with self.graph.as_default():
            with tf.name_scope(name='variance_estimate'):
                self.core.variance_estimate()

        fd = {self.core.indices: indices,
              self.core.values: values,
              self.core.shape: shape,
              self.core.n_users: n_users}

        self.session.run(self.core.init_variance_vars)
        self.session.run(self.core.variance, feed_dict=fd)


class FMRegressionLFP(FMRegression, LatentFactorPortfolio):

    def __init__(self,
                 loss_function=tf.losses.mean_squared_error,
                 **kwargs):
        kwargs['loss_function'] = loss_function
        kwargs['cmp_graph'] = PointwiseLFPGraph
        super(FMRegressionLFP, self).__init__(**kwargs)

    def fit(self, X, y=None):
        super(FMRegressionLFP, self).fit(X, y)
        self.compute_variance(X)


class FMClassificationLFP(FMClassification, LatentFactorPortfolio):

    def __init__(self,
                 loss_function=tf.losses.mean_squared_error,
                 **kwargs):
        kwargs['loss_function'] = loss_function
        kwargs['cmp_graph'] = PointwiseLFPGraph
        super(FMClassificationLFP, self).__init__(**kwargs)

    def fit(self, X, y=None):
        super(FMClassificationLFP, self).fit(X, y)
        self.compute_variance(X)


class FMPairwiseRankingLFP(FMPairwiseRanking, LatentFactorPortfolio):

    def __init__(self, use_scipy=True, **kwargs):
        kwargs['cmp_graph'] = BPRLFPGraph
        super(FMPairwiseRankingLFP, self).__init__(**kwargs)
        self.use_scipy = use_scipy

    def fit(self, pos, neg=None):
        super(FMPairwiseRankingLFP, self).fit(pos, neg)
        # free memory
        del neg
        self.compute_variance(x)

