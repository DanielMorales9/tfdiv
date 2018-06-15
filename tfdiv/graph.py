from abc import ABC, abstractmethod
import tensorflow as tf
from tfdiv.utility import cond

MEAN = 0.


class ComputationalGraph(ABC):
    """
    Abstract Computational Graph

    Parameters
    ------
    dtype: `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    n_factors : int, optional (Default 10)
        The number of factors used to factorize
        pairwise interactions between variables.
    """

    def __init__(self,
                 dtype=tf.float32,
                 l2_w=0.001,
                 l2_v=0.001,
                 n_factors=10,
                 init_std=0.01,
                 learning_rate=0.01,
                 opt_kwargs=None,
                 optimizer=tf.train.AdamOptimizer):
        if opt_kwargs is None:
            opt_kwargs = {}
        self.n_factors = n_factors
        self.dtype = dtype
        self.l2_v = l2_v
        self.l2_w = l2_w
        self.init_std = init_std
        self.learning_rate = learning_rate
        self.opt_kwargs = opt_kwargs
        self.optimizer = optimizer(learning_rate=self.learning_rate, **self.opt_kwargs)
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
        self.batch_loss = None
        self.size = None
        self.ops = None
        self.x = None
        self.y = None

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
        self.fit_operations()

    def init_params(self):

        lambda_w = tf.constant(self.l2_w, dtype=self.dtype, name='lambda_w')
        lambda_v = tf.constant(self.l2_v, dtype=self.dtype, name='lambda_v')
        half = tf.constant(0.5, dtype=self.dtype, name='half')
        bias = tf.verify_tensor_all_finite(tf.Variable(self.init_std,
                                                       trainable=True,
                                                       name='bias'),
                                           msg='NaN or Inf in bias')

        rnd_weights = tf.random_normal(tf.expand_dims(self.n_features, 0),
                                       stddev=self.init_std,
                                       mean=MEAN,
                                       dtype=self.dtype)
        weights = tf.verify_tensor_all_finite(tf.Variable(rnd_weights,
                                                          trainable=True,
                                                          validate_shape=False,
                                                          name='weights'),
                                              msg='NaN or Inf in weights')
        tf_shape = tf.stack([self.n_features, self.n_factors])
        rnd_params = tf.random_normal(tf_shape,
                                      stddev=self.init_std,
                                      mean=MEAN,
                                      dtype=self.dtype)
        params = tf.verify_tensor_all_finite(tf.Variable(rnd_params,
                                                         trainable=True,
                                                         validate_shape=False,
                                                         name='params'),
                                             msg='NaN or Inf in parameters')
        self.lambda_w = cond(self.lambda_w, lambda_w)
        self.lambda_v = cond(self.lambda_v, lambda_v)
        self.half = cond(self.half, half)
        self.bias = cond(self.bias, bias)
        self.weights = cond(self.weights, weights)
        self.params = cond(self.params, params)
        tf.summary.scalar('bias', self.bias)

    def init_trainer(self):
        self.trainer = self.optimizer.minimize(self.checked_target,
                                               global_step=self.global_step)

    def init_regularization(self):
        self.l2_norm = tf.reduce_sum(
            tf.add(
                tf.multiply(self.lambda_w, tf.pow(self.weights, 2)),
                tf.multiply(self.lambda_v,
                            tf.transpose(tf.pow(self.params, 2)))))

    def init_target(self):
        self.target = self.l2_norm + self.reduced_loss
        self.checked_target = tf.verify_tensor_all_finite(
            self.target,
            msg='NaN or Inf in target value',
            name='target')
        tf.summary.scalar('target', self.checked_target)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def init_placeholder(self):
        self.n_features = tf.placeholder(shape=[],
                                         dtype=tf.int64,
                                         name='n_features')

    def fit_operations(self):
        self.ops = [self.trainer,
                    self.summary_op,
                    self.global_step,
                    self.batch_loss]

    @abstractmethod
    def init_main_graph(self):
        pass

    @abstractmethod
    def init_loss(self):
        pass


class PointwiseGraph(ComputationalGraph):
    """
    Pointwise Graph

    Parameters
    ------
    dtype: `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    n_factors : int, optional (Default 10)
        The number of factors used to factorize
        pairwise interactions between variables.
    """
    def __init__(self,
                 loss_function=tf.losses.mean_squared_error,
                 dtype=tf.float32,
                 l2_w=0.001,
                 l2_v=0.001,
                 n_factors=10,
                 init_std=0.01,
                 learning_rate=0.01,
                 opt_kwargs=None,
                 optimizer=tf.train.AdamOptimizer):
        super(PointwiseGraph, self).__init__(dtype=dtype,
                                             opt_kwargs=opt_kwargs,
                                             l2_w=l2_w,
                                             l2_v=l2_v,
                                             n_factors=n_factors,
                                             init_std=init_std,
                                             learning_rate=learning_rate,
                                             optimizer=optimizer)
        self.loss_function = loss_function

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
        self.size = tf.cast(tf.size(self.y_hat), dtype=self.dtype)
        self.batch_loss = self.reduced_loss * self.size
        tf.summary.scalar('loss', self.reduced_loss)


class RankingGraph(ComputationalGraph):
    """
    Abstract Ranking Computational Graph

    Parameters
    ------
    dtype: `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    n_factors : int, optional (Default 10)
        The number of factors used to factorize
        pairwise interactions between variables.
    """

    def __init__(self,
                 dtype=tf.float32,
                 l2_w=0.001,
                 l2_v=0.001,
                 n_factors=10,
                 init_std=0.01,
                 learning_rate=0.01,
                 opt_kwargs=None,
                 optimizer=tf.train.AdamOptimizer):
        super(RankingGraph, self).__init__(dtype=dtype,
                                           opt_kwargs=opt_kwargs,
                                           l2_w=l2_w,
                                           l2_v=l2_v,
                                           n_factors=n_factors,
                                           init_std=init_std,
                                           learning_rate=learning_rate,
                                           optimizer=optimizer)
        self.k = None
        self.n_users = None
        self.n_items = None
        self.pred = None
        self.ranking_results = None

    def init_placeholder(self):
        ComputationalGraph.init_placeholder(self)
        self.k = tf.placeholder(shape=[], dtype=tf.int32, name='k')
        self.n_users = tf.placeholder(dtype=tf.int64, shape=[], name='n_users')
        self.n_items = tf.placeholder(dtype=tf.int64, shape=[], name='n_items')
        self.pred = tf.placeholder(dtype=self.dtype, shape=[None], name='y_pred')

    def ranking_computation(self):
        shape = tf.stack([self.n_users, self.n_items])
        predictions = tf.reshape(self.pred, shape)
        self.ranking_results = tf.nn.top_k(predictions, k=self.k)


class PointwiseRankingGraph(PointwiseGraph, RankingGraph):

    """
    Pointwise Ranking Computational Graph


    Parameters
    ------
    dtype: `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    n_factors : int, optional (Default 10)
        The number of factors used to factorize
        pairwise interactions between variables.

    """

    def __init__(self,
                 dtype=tf.float32,
                 l2_w=0.001,
                 l2_v=0.001,
                 n_factors=10,
                 init_std=0.01,
                 learning_rate=0.01,
                 opt_kwargs=None,
                 optimizer=tf.train.AdamOptimizer):
        super(PointwiseRankingGraph, self).__init__(dtype=dtype,
                                                    l2_w=l2_w,
                                                    l2_v=l2_v,
                                                    opt_kwargs=opt_kwargs,
                                                    n_factors=n_factors,
                                                    init_std=init_std,
                                                    learning_rate=learning_rate,
                                                    optimizer=optimizer)


class BayesianPersonalizedRankingGraph(RankingGraph):
    """
    Bayesian Personalized Ranking Computational Graph

    Parameters
    ------
    dtype: `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    n_factors : int, optional (Default 10)
        The number of factors used to factorize
        pairwise interactions between variables.
    """
    def __init__(self,
                 dtype=tf.float32,
                 l2_w=0.001,
                 l2_v=0.001,
                 n_factors=10,
                 init_std=0.01,
                 learning_rate=0.01,
                 opt_kwargs=None,
                 optimizer=tf.train.AdamOptimizer):
        super(BayesianPersonalizedRankingGraph, self).__init__(dtype=dtype,
                                                               l2_w=l2_w,
                                                               l2_v=l2_v,
                                                               n_factors=n_factors,
                                                               opt_kwargs=opt_kwargs,
                                                               init_std=init_std,
                                                               learning_rate=learning_rate,
                                                               optimizer=optimizer)
        self.y_hat = None
        self.neg_hat = None

    def init_placeholder(self):
        RankingGraph.init_placeholder(self)
        self.x = tf.sparse_placeholder(self.dtype,
                                       shape=[None, self.n_features],
                                       name='pos')
        self.y = tf.sparse_placeholder(self.dtype,
                                       shape=[None, self.n_features],
                                       name='neg')

    def init_main_graph(self):
        self.y_hat = self.equation(self.x)
        self.neg_hat = self.equation(self.y)

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
        self.loss = tf.log(tf.sigmoid(tf.subtract(self.y_hat, self.neg_hat)))
        self.reduced_loss = -tf.reduce_mean(self.loss)
        self.size = tf.cast(tf.size(self.y_hat), dtype=self.dtype)
        self.batch_loss = self.reduced_loss * self.size
        tf.summary.scalar('loss', self.reduced_loss)


class LatentFactorPortfolioGraph(RankingGraph):

    """
    Abstract Latent Factor Portfolio Graph

    Parameters
    ------
    dtype: `tensorflow.dtype`, optional (Default tf.float32)
        Tensors dtype to use.
    l2_v : float, optional (Default 0.001)
        L2 Regularization value for factorized parameters.
    l2_w : float, optional (Default 0.001)
        L2 Regularization value for linear weights.
    learning_rate : float, optional (Default 0.001)
        Learning rate schedule for weight updates.
    optimizer : ``tf.train`` module, optional (Default tf.train.AdamOptimizer)
        The optimized for parameters optimization.
    n_factors : int, optional (Default 10)
        The number of factors used to factorize
        pairwise interactions between variables.
    """

    def __init__(self,
                 dtype=tf.float32,
                 l2_w=0.001,
                 l2_v=0.001,
                 n_factors=10,
                 init_std=0.01,
                 learning_rate=0.01,
                 opt_kwargs=None,
                 optimizer=tf.train.AdamOptimizer):
        super(LatentFactorPortfolioGraph, self).__init__(dtype=dtype,
                                                         l2_w=l2_w,
                                                         opt_kwargs=opt_kwargs,
                                                         l2_v=l2_v,
                                                         n_factors=n_factors,
                                                         init_std=init_std,
                                                         learning_rate=learning_rate,
                                                         optimizer=optimizer)
        self.variance = None
        self.unique_x = None
        self.init_variance_vars = None
        self.init_unique_vars = None

        self.delta_f = None
        self.predictions = None
        self.rankings = None
        self.b = None
        self.pk = None

    @staticmethod
    def two_dim_shape(x, rankings):
        return tf.stack([tf.reduce_prod(tf.shape(rankings)),
                         tf.shape(x)[-1]], axis=0)

    @staticmethod
    def three_dim_shape(x, rankings):
        return tf.concat([tf.shape(rankings),
                          tf.shape(x)[::-1]], axis=0)[:-1]

    @staticmethod
    def shape_cube_by_rank(x, rankings):
        cube_shape = LatentFactorPortfolioGraph.three_dim_shape(x, rankings)
        x_cube = tf.sparse_reshape(x, shape=cube_shape)
        return x_cube

    @staticmethod
    def swap_tensor_by_rank(x, rankings):
        inverse_ranking = tf.map_fn(tf.invert_permutation, rankings)
        new_items_indices = tf.gather_nd(tf.to_int64(inverse_ranking),
                                         x.indices[:, :2])
        new_indices = tf.stack([x.indices[:, 0],
                                new_items_indices,
                                x.indices[:, 2]], axis=1)
        new_x = tf.SparseTensor(new_indices, x.values, x.dense_shape)
        reordered_x = tf.sparse_reorder(new_x)
        return reordered_x

    @staticmethod
    def zero_users_columns(x, n_users, axis=1):
        assert axis > 0, "Axis must be greater then 0"
        gte = tf.greater_equal(x.indices[:, axis], n_users)
        lt = tf.less(x.indices[:, axis], n_users)
        new_x = tf.sparse_retain(x, gte)
        users_x = tf.sparse_retain(x, lt)
        return new_x, users_x

    @staticmethod
    def reshape_dataset(n_users, rankings, x):
        shaped_x = LatentFactorPortfolioGraph.shape_cube_by_rank(x, rankings)
        swapped_x = LatentFactorPortfolioGraph.swap_tensor_by_rank(shaped_x, rankings)
        zeroed_x, users_x = LatentFactorPortfolioGraph.zero_users_columns(swapped_x, n_users, axis=2)
        return zeroed_x, users_x

    @staticmethod
    def variance_lookup(users_indices, variance):
        unique_users = tf.unique(users_indices)[0]
        user_variance = tf.gather(variance, unique_users)
        return user_variance

    @staticmethod
    def ranking_coefficient(k, dtype=tf.float32):
        # Rank weighting function
        one = tf.ones(shape=[1], dtype=dtype)
        if dtype == tf.float32:
            two_k = tf.to_float(tf.pow(tf.constant([2]), k))
        else:
            two_k = tf.to_double(tf.pow(tf.constant([2]), k))
        return tf.div(one, two_k)

    @staticmethod
    def ranking_weights(k, dtype=tf.float32):
        ones = tf.ones(shape=k, dtype=dtype)
        two_k = tf.pow(tf.constant([2]),
                       tf.range(k))
        if dtype == tf.float32:
            two_k = tf.to_float(two_k)
        else:
            two_k = tf.to_double(two_k)
        pm = tf.div(ones, two_k)
        return pm

    @staticmethod
    def dot_product(params, rankings, x):
        two_dims = LatentFactorPortfolioGraph.two_dim_shape(x, rankings)
        x_two_dim = tf.sparse_reshape(x, shape=two_dims)
        dot_prod = tf.sparse_tensor_dense_matmul(x_two_dim, params)
        three_dims = LatentFactorPortfolioGraph.three_dim_shape(params, rankings)
        dot_prod_three_dim = tf.reshape(dot_prod, shape=three_dims)
        return dot_prod_three_dim

    @staticmethod
    def second_term(k, pk, variance, dot_prod):
        square_dot_prod = dot_prod ** 2
        var_square_dot_prod = tf.reduce_sum(
            tf.transpose(square_dot_prod[:, k:], perm=(1, 0, 2)) * variance, [2])
        weighted_var_dot = pk * var_square_dot_prod
        return tf.transpose(weighted_var_dot)

    @staticmethod
    def third_term(k, variance, dot_prod, dtype=tf.float32):
        pm = LatentFactorPortfolioGraph.ranking_weights(k, dtype=dtype)
        ranked_dot = dot_prod[:, :k]
        unranked_dot = dot_prod[:, k:]
        a = tf.tensordot(tf.transpose(ranked_dot, perm=(0, 2, 1)), pm, axes=[2, 0])
        b = tf.transpose(unranked_dot, perm=[1, 0, 2]) * variance
        mn = tf.transpose(tf.reduce_sum(a * b, axis=2), perm=[1, 0])
        return mn

    def init_placeholder(self):
        RankingGraph.init_placeholder(self)
        # Predictions and Rankings
        self.predictions = tf.placeholder(shape=[None, None],
                                          dtype=self.dtype,
                                          name='predictions')
        self.rankings = tf.placeholder(shape=[None, None],
                                       dtype=tf.int32,
                                       name='rankings')
        # System-level diversity
        self.b = tf.placeholder(shape=[], dtype=self.dtype, name='b')

    def variance_estimate(self):
        # Variables and tensors initialization
        tf_shape = tf.stack([self.n_users, self.n_factors])
        variance = tf.ones(tf_shape, dtype=self.dtype)
        init_var = tf.Variable(variance, name='variance',
                               validate_shape=False,
                               trainable=False)
        init_sum_of_square = tf.Variable(tf.zeros(shape=tf_shape,
                                                  dtype=self.dtype),
                                         name='sum_of_square',
                                         validate_shape=False,
                                         trainable=False)
        init_nu = tf.Variable(tf.zeros(shape=self.n_users, dtype=tf.int64),
                              name='n_items_per_user',
                              validate_shape=False,
                              trainable=False)
        ones = tf.ones(dtype=tf.int64, shape=tf.shape(self.x)[0])
        u_idx = self.x.indices[:, 1]
        lim_users = tf.expand_dims(self.n_users, axis=0)
        where = tf.less(u_idx, lim_users)
        indexes = tf.reshape(tf.where(where), shape=[-1])
        indexes = tf.nn.embedding_lookup(self.x.indices, indexes)[:, 1]

        # computes the square for the batch (batch_size, n_factors)
        # each row represent the square root for a user
        user_v = tf.nn.embedding_lookup(self.params, indexes)
        dot = tf.sparse_tensor_dense_matmul(self.x, self.params)
        dot = user_v - (dot - user_v)
        sq = tf.square(dot)

        sum_of_square = tf.scatter_add(init_sum_of_square, indexes, sq)
        nu = tf.scatter_add(init_nu, indexes, ones)
        nu = tf.tile(tf.expand_dims(tf.to_float(nu), 1), [1, self.n_factors])
        computed_variance = sum_of_square / nu
        self.variance = tf.assign(init_var, computed_variance)
        self.init_variance_vars = tf.variables_initializer([init_var,
                                                            init_nu,
                                                            init_sum_of_square])

    def unique_rows_sparse_tensor(self):
        n_users = self.n_users
        n_items = self.n_items
        max_allowed_features = n_users + n_items
        less_cond = tf.less(self.x.indices[:, 1], max_allowed_features)
        retained_x = tf.sparse_retain(self.x, less_cond)
        retained_idx = retained_x.indices
        tf_shape = tf.to_int64(tf.stack([tf.shape(self.x)[0], 4]))
        re_idx = tf.reshape(retained_idx, tf_shape)
        enc_ten = (re_idx[:, 1] + 1) * tf.reduce_max(re_idx[:, 3]) + re_idx[:, 3]
        nq, idx = tf.unique(enc_ten)
        num_partitions = tf.shape(nq)[0]
        sparse_rows = tf.unsorted_segment_min(re_idx[:, 0], idx, num_partitions)

        rng = tf.range(tf.shape(self.x.indices, out_type=tf.int64)[0])
        max_rows = tf.segment_max(rng, self.x.indices[:, 0]) + 1
        min_rows = tf.segment_min(rng, self.x.indices[:, 0])
        min_max = tf.stack([min_rows, max_rows], axis=1)
        ga = tf.gather(min_max, sparse_rows)
        num_rows = tf.shape(ga)[0]
        init_array = tf.TensorArray(tf.int64, size=num_rows, infer_shape=False)

        def loop_body(i, ta):
            return i + 1, ta.write(i, tf.range(ga[i, 0], ga[i, 1]))

        _, result_array = tf.while_loop(lambda i, ta: i < num_rows,
                                        loop_body, [0, init_array])
        rows = result_array.concat()
        trues = tf.ones(tf.shape(rows)[0], dtype=tf.bool)
        mask = tf.zeros((tf.shape(self.x.indices)[0]), dtype=tf.bool)
        init_mask = tf.Variable(mask, validate_shape=False, trainable=False)
        new_mask = tf.scatter_update(init_mask, rows, trues)

        new_x = tf.sparse_retain(self.x, new_mask)
        unq, idx = tf.unique(new_x.indices[:, 0])
        new_idx = tf.stack([tf.to_int64(idx), new_x.indices[:, 1]], axis=1)
        new_shape = tf.stack([tf.shape(unq, out_type=tf.int64)[0],
                              new_x.dense_shape[1]])
        self.unique_x = tf.SparseTensor(indices=new_idx,
                                        values=new_x.values,
                                        dense_shape=new_shape)
        self.init_unique_vars = tf.variables_initializer([init_mask])

    def delta_f_computation(self):
        zero_x, users_x = self.reshape_dataset(self.n_users, self.rankings, self.x)
        users = users_x.indices[:, 2]
        user_var = self.variance_lookup(users, self.variance)
        self.pk = self.ranking_coefficient(self.k, self.dtype)

        first_term = self.predictions[:, self.k:]
        dot_prod = LatentFactorPortfolioGraph.dot_product(self.params, self.rankings, zero_x)
        second_term = LatentFactorPortfolioGraph.second_term(self.k, self.pk, user_var, dot_prod)
        third_term = LatentFactorPortfolioGraph.third_term(self.k, user_var, dot_prod, self.dtype)
        second_n_third = second_term + 2 * third_term
        self.delta_f = self.pk * (first_term - self.b * second_n_third)


class PointwiseLFPGraph(PointwiseRankingGraph, LatentFactorPortfolioGraph):

    """
    Pointwise Latent Factor Portfolio Graph
    """

    def init_placeholder(self):
        PointwiseRankingGraph.init_placeholder(self)
        LatentFactorPortfolioGraph.init_placeholder(self)


class BPRLFPGraph(BayesianPersonalizedRankingGraph, LatentFactorPortfolioGraph):

    """
    Bayesian Personalized Ranking version of Latent Factor Portfolio Graph

    """

    def init_placeholder(self):
        BayesianPersonalizedRankingGraph.init_placeholder(self)
        LatentFactorPortfolioGraph.init_placeholder(self)
