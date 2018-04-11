from abc import ABC, abstractmethod
import tensorflow as tf


class ComputationalGraph(ABC):

    def __init__(self,
                 dtype=tf.float32,
                 l2_w=0.001,
                 l2_v=0.001,
                 n_factors=10,
                 init_std=0.01):
        self.n_factors = n_factors
        self.dtype = dtype
        self.l2_v = l2_v
        self.l2_w = l2_w
        self.init_std = init_std
        self.init_all_vars = None
        self.n_features = None
        self.summary_op = None
        self.lambda_v = None
        self.lambda_w = None
        self.half = None
        self.bias = None
        self.params = None
        self.weights = None
        self.global_step = None
        self.l2_norm = None
        self.y_hat = None
        self.loss = None
        self.reduced_loss = None
        self.target = None
        self.checked_target = None
        self.trainer = None
        self.init_all_vars = None
        self.ops = None

    def define_graph(self):
        self.global_step = tf.train.create_global_step()
        with tf.name_scope('placeholders'):
            self.init_placeholder()
        with tf.name_scope('parameters'):
            self.init_params()
        with tf.name_scope('main_graph'):
            self.init_main_graph()
        with tf.name_scope('loss'):
            self.init_loss()
            self.init_regularization()
        with tf.name_scope('target'):
            self.init_target()
        with tf.name_scope('training'):
            self.init_trainer()

        self.init_all_vars = tf.global_variables_initializer()
        self.summary_op = tf.summary.merge_all()
        self.set_ops()

    def init_params(self):
        self.lambda_w = tf.constant(self.l2_w,
                                    dtype=self.dtype,
                                    name='lambda_w')
        self.lambda_v = tf.constant(self.l2_v,
                                    dtype=self.dtype,
                                    name='lambda_w')
        self.half = tf.constant(0.5,
                                dtype=self.dtype,
                                name='half')
        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(self.init_std,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        rnd_weights = tf.random_uniform([self.n_features],
                                        minval=-self.init_std,
                                        maxval=self.init_std,
                                        dtype=self.dtype)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights,
                        trainable=True,
                        name='weights'),
            msg='NaN or Inf in weights')
        rnd_params = tf.random_uniform([self.n_features, self.n_factors],
                                       minval=-self.init_std,
                                       maxval=self.init_std,
                                       dtype=self.dtype)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(rnd_params,
                        trainable=True,
                        name='params'),
            msg='NaN or Inf in parameters')
        tf.summary.scalar('bias', self.bias)

    @abstractmethod
    def set_ops(self):
        pass

    @abstractmethod
    def set_params(self, *args):
        pass

    @abstractmethod
    def init_placeholder(self):
        pass

    @abstractmethod
    def init_main_graph(self):
        pass

    @abstractmethod
    def init_loss(self):
        pass

    @abstractmethod
    def init_regularization(self):
        pass

    @abstractmethod
    def init_target(self):
        pass

    @abstractmethod
    def init_trainer(self):
        pass


class RegressionGraph(ComputationalGraph):

    def __init__(self,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 loss_function=tf.losses.mean_squared_error,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer):
        super(RegressionGraph, self).__init__(
            dtype=dtype,
            init_std=init_std,
            n_factors=n_factors,
            l2_w=l2_w,
            l2_v=l2_v)
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.optimizer_fun = self.optimizer(learning_rate=self.learning_rate)
        self.x = None
        self.y = None

    def set_ops(self):
        self.ops = [self.trainer, self.summary_op, self.global_step]

    def init_placeholder(self):
        pass

    def set_params(self, x, y):
        self.x = x
        self.y = y

    def init_main_graph(self):
        x = self.x
        assert x is not None, "x must be set before graph is defined"
        el_wise_mul = x * self.weights
        weighted_sum = tf.sparse_reduce_sum(el_wise_mul, 1, keep_dims=True)
        linear_terms = tf.add(self.bias, weighted_sum, name='linear_terms')
        dot = tf.sparse_tensor_dense_matmul(x, self.params)
        pow_sum = tf.pow(dot, 2)
        pow_x = x * tf.sparse_tensor_to_dense(x)
        pow_v = tf.pow(self.params, 2)
        sum_pow = tf.sparse_tensor_dense_matmul(pow_x, pow_v)
        sub = tf.subtract(pow_sum, sum_pow)
        sum_sub = tf.reduce_sum(sub, 1, keepdims=True)
        pair_interactions = tf.multiply(self.half, sum_sub)
        self.y_hat = tf.add(linear_terms, pair_interactions)

    def init_loss(self):
        y_true = self.y
        assert y_true is not None, "y must be set before graph is defined"
        self.loss = self.loss_function(self.y_hat, y_true)
        self.reduced_loss = tf.reduce_mean(self.loss)
        tf.summary.scalar('loss', self.reduced_loss)

    def init_regularization(self):
        self.l2_norm = tf.reduce_sum(
            tf.add(
                tf.multiply(self.lambda_w, tf.pow(self.weights, 2)),
                tf.multiply(self.lambda_v,
                            tf.transpose(tf.pow(self.params, 2)))))

    def init_target(self):
        self.target = self.reduced_loss + self.l2_norm
        self.checked_target = tf.verify_tensor_all_finite(
            self.target,
            msg='NaN or Inf in target value',
            name='target')
        tf.summary.scalar('target', self.checked_target)

    def init_trainer(self):
        self.trainer = self.optimizer_fun.minimize(self.checked_target,
                                                   global_step=self.global_step)


class BayesianPersonalizedRankingGraph(ComputationalGraph):

    def __init__(self,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer):
        super(BayesianPersonalizedRankingGraph, self).__init__(
            dtype=dtype,
            init_std=init_std,
            n_factors=n_factors,
            l2_w=l2_w,
            l2_v=l2_v
        )
        self.learning_rate = learning_rate
        self.optimizer = optimizer(learning_rate=self.learning_rate)
        self.pos_sample = None
        self.neg_sample = None
        self.pos_hat = None
        self.neg_hat = None

    def set_params(self, pos_sample, neg_sample):
        pass

    def set_ops(self):
            self.ops = [self.trainer, self.summary_op, self.global_step]

    def init_placeholder(self):
        self.pos_sample = tf.sparse_placeholder(self.dtype,
                                                shape=[None, self.n_features],
                                                name='pos')
        self.neg_sample = tf.sparse_placeholder(self.dtype,
                                                shape=[None, self.n_features],
                                                name='neg')

    def init_params(self):
        self.lambda_w = tf.constant(self.l2_w,
                                    dtype=self.dtype,
                                    name='lambda_w')
        self.lambda_v = tf.constant(self.l2_v,
                                    dtype=self.dtype,
                                    name='lambda_w')
        self.half = tf.constant(0.5,
                                dtype=self.dtype,
                                name='half')
        bias = tf.random_uniform([1],
                                 minval=-self.init_std,
                                 maxval=self.init_std,
                                 dtype=self.dtype)
        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(bias, name='bias'),
            msg='NaN or Inf in bias')
        rnd_weights = tf.random_uniform([self.n_features],
                                        minval=-self.init_std,
                                        maxval=self.init_std,
                                        dtype=self.dtype)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights, name='weights'),
            msg='NaN or Inf in weights')
        rnd_params = tf.random_uniform([self.n_features, self.n_factors],
                                       minval=-self.init_std,
                                       maxval=self.init_std,
                                       dtype=tf.float32)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(rnd_params, name='params'),
            msg='NaN or Inf in parameters')

    def init_main_graph(self):
        self.pos_hat = self.equation(self.pos_sample)
        self.neg_hat = self.equation(self.neg_sample)

    def equation(self, x):
        el_wise_mul = x * self.weights
        weighted_sum = tf.sparse_reduce_sum(el_wise_mul, 1, keep_dims=True)
        linear_terms = tf.add(self.bias, weighted_sum, name='linear_terms')
        dot = tf.sparse_tensor_dense_matmul(x, self.params)
        pow_sum = tf.pow(dot, 2)
        pow_x = x * tf.sparse_tensor_to_dense(x)
        pow_v = tf.pow(self.params, 2)
        sum_pow = tf.sparse_tensor_dense_matmul(pow_x, pow_v)
        sub = tf.subtract(pow_sum, sum_pow)
        sum_sub = tf.reduce_sum(sub, 1, keepdims=True)
        pair_interactions = tf.multiply(self.half, sum_sub)
        y_hat = tf.add(linear_terms, pair_interactions)
        return y_hat

    def init_loss(self):
        self.loss = tf.log(tf.sigmoid(tf.subtract(self.pos_hat, self.neg_hat)))
        self.reduced_loss = tf.reduce_mean(self.loss)
        tf.summary.scalar('loss', self.reduced_loss)

    def init_regularization(self):
        self.l2_norm = tf.reduce_sum(
            tf.add(
                tf.multiply(self.lambda_w, tf.pow(self.weights, 2)),
                tf.multiply(self.lambda_v,
                            tf.transpose(tf.pow(self.params, 2)))))

    def init_target(self):
        self.target = self.l2_norm - self.reduced_loss
        self.checked_target = tf.verify_tensor_all_finite(
            self.target,
            msg='NaN or Inf in target value',
            name='target')
        tf.summary.scalar('target', self.checked_target)

    def init_trainer(self):
        self.trainer = self.optimizer.minimize(self.checked_target,
                                               global_step=self.global_step)
