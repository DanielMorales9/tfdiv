from sklearn.base import BaseEstimator, ClassifierMixin
from graph import RegressionGraph, BayesianPersonalizedRankingGraph as BPR
from utility import sparse_repr
from abc import abstractmethod
from dataset import PairDataset
import tensorflow as tf
from tqdm import tqdm
import numpy as np


class BaseClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 shuffle_size=1000,
                 dtype=tf.float32,
                 seed=1,
                 show_progress=True,
                 log_dir=None,
                 core=None):
        self.seed = seed
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        self.session = tf.Session(graph=self.graph)
        self.core = core

        self.dtype = dtype
        self.ntype = np.float32 if dtype is tf.float32 else np.float64
        self.epochs = epochs
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size

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

    def log_summary(self, *args):
        _, summary, step = args
        if self.logging_enabled:
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y=None, sample_weight=None):
        pass


class FMRegression(BaseClassifier):

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
                 core=None):
        super(FMRegression, self).__init__(epochs=epochs,
                                           batch_size=batch_size,
                                           shuffle_size=shuffle_size,
                                           show_progress=show_progress,
                                           seed=seed,
                                           log_dir=log_dir,
                                           core=core)

        # Computational graph initialization
        self.core = RegressionGraph(n_factors=n_factors,
                                    init_std=init_std,
                                    dtype=dtype,
                                    optimizer=optimizer,
                                    learning_rate=learning_rate,
                                    loss_function=loss_function,
                                    l2_v=l2_v,
                                    l2_w=l2_w)

    def fit(self, X, y=None):
        with self.graph.as_default():
            input_vars = self.init_input(X, y)
            dataset, train_iterator = self.init_dataset(*input_vars)
            it = self.init_iterator(dataset)
            self.init_computational_graph(it)

        if it:
            self.session.run(self.core.init_all_vars)

        ops = self.core.ops
        train_handle = self.session.run(train_iterator.string_handle())
        feed_dict = {self.handle: train_handle}
        for _ in tqdm(range(self.epochs),
                      unit='epochs',
                      disable=not self.show_progress):
            self.session.run(train_iterator.initializer)
            while True:
                try:
                    logs = self.session.run(ops, feed_dict=feed_dict)
                    self.log_summary(*logs)
                except tf.errors.OutOfRangeError:
                    break

    def predict(self, X):

        with self.graph.as_default():
            input_vars = self.init_input(X, None)
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
        return np.concatenate(results).reshape(-1)

    def init_computational_graph(self, it):
        if it:
            self.core.n_features = self.n_features
            self.core.set_params(*it.get_next())
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

    def init_input(self, x, y):
        x_d = tf.SparseTensor(*sparse_repr(x, self.ntype))

        if y is not None:
            n_samples, self.n_features = x.shape
            y_d = tf.convert_to_tensor(y, self.dtype)
            return x_d, y_d, n_samples, True
        else:
            n_samples, _ = x.shape
            y_d = tf.zeros(x.shape[0], dtype=self.dtype)
            return x_d, y_d, n_samples, False

    def init_dataset(self, x_d, y_d, n_samples, train):
        batch_size = n_samples if self.batch_size == -1 \
            else self.batch_size
        dataset = tf.data.Dataset.from_tensor_slices((x_d, y_d)) \
            .batch(batch_size)
        if train:
            dataset = dataset.shuffle(self.shuffle_size)

        iterator = dataset.make_initializable_iterator()
        return dataset, iterator

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
                 bootstrap_sampling='random',
                 log_dir=None):
        super(FMPairwiseRanking, self).__init__(epochs=epochs,
                                                batch_size=batch_size,
                                                shuffle_size=shuffle_size,
                                                show_progress=show_progress,
                                                seed=seed,
                                                log_dir=log_dir)
        self.frac = frac
        self.bootstrap_sampling = bootstrap_sampling
        # Computational graph initialization
        self.core = BPR(n_factors=n_factors,
                        init_std=init_std,
                        dtype=dtype,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        l2_v=l2_v,
                        l2_w=l2_w)

    def init_input(self, pos, neg=None):
        if neg is not None:
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
        self.core.n_features = self.n_features
        self.core.define_graph()

    def fit(self, pos, neg=None):

        with self.graph.as_default():
            pos, neg = self.init_input(pos, neg)
            dataset = self.init_dataset(pos, neg, self.bootstrap_sampling)
            self.init_computational_graph()

        if not self.session.run(tf.is_variable_initialized(
                self.core.global_step)):
            self.session.run(self.core.init_all_vars)

        ops = self.core.ops
        for _ in tqdm(range(self.epochs), unit='epochs',
                      disable=not self.show_progress):
            for pos, neg in dataset.get_next():
                feed_dict = dataset.batch_to_feed_dict(pos, neg, self.core)
                logs = self.session.run(ops, feed_dict=feed_dict)
                self.log_summary(*logs)

    def predict(self, x):
        with self.graph.as_default():
            x = self.init_input(x)
            dataset = self.init_dataset(x)

        ops = self.core.pos_hat
        results = []
        for pos in dataset.get_next():
            feed_dict = dataset.batch_to_feed_dict(pos, core=self.core)
            hat = self.session.run(ops, feed_dict=feed_dict)
            results.append(hat)
        return np.concatenate(results).reshape(-1)

    def score(self, X, y=None, sample_weight=None):
        pass


