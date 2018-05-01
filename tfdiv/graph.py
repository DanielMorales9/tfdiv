from abc import ABC, abstractmethod
import tensorflow as tf
from tfdiv.utility import cond


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
        self.optimizer = None
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
        self.saver = None
        self.ops = None
        self.x = None
        self.y = None

    def define_graph(self, n_features):
        self.n_features = n_features
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
        lambda_w = tf.constant(self.l2_w, dtype=self.dtype, name='lambda_w')
        lambda_v = tf.constant(self.l2_v, dtype=self.dtype, name='lambda_w')
        half = tf.constant(0.5, dtype=self.dtype, name='half')
        bias = tf.verify_tensor_all_finite(tf.Variable(self.init_std,
                                                       trainable=True,
                                                       name='bias'),
                                           msg='NaN or Inf in bias')
        rnd_weights = tf.random_uniform([self.n_features],
                                        minval=-self.init_std,
                                        maxval=self.init_std,
                                        dtype=self.dtype)
        weights = tf.verify_tensor_all_finite(tf.Variable(rnd_weights,
                                                          trainable=True,
                                                          name='weights'),
                                              msg='NaN or Inf in weights')
        rnd_params = tf.random_uniform([self.n_features, self.n_factors],
                                       minval=-self.init_std,
                                       maxval=self.init_std,
                                       dtype=self.dtype)
        params = tf.verify_tensor_all_finite(tf.Variable(rnd_params,
                                                         trainable=True,
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

    @abstractmethod
    def set_ops(self):
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


class PointwiseGraph(ComputationalGraph):

    def __init__(self,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 loss_function=tf.losses.mean_squared_error,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer):
        super(PointwiseGraph, self).__init__(
            dtype=dtype,
            init_std=init_std,
            n_factors=n_factors,
            l2_w=l2_w,
            l2_v=l2_v)
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer(learning_rate=self.learning_rate)

    def set_ops(self):
        self.ops = [self.trainer,
                    self.summary_op,
                    self.global_step,
                    self.batch_loss]

    def init_placeholder(self):
        pass

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
        self.y_hat = None
        self.neg_hat = None

    def set_params(self, x, y):
        pass

    def set_ops(self):
        self.ops = [self.trainer,
                    self.summary_op,
                    self.global_step,
                    self.batch_loss]

    def init_placeholder(self):
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


class LFPGraph(ComputationalGraph):

    def __init__(self, **kwargs):
        super(LFPGraph, self).__init__(**kwargs)
        self.variance = None
        self.init_variance_vars = None
        self.init_unique_vars = None

        self.n_items = None
        self.n_users = None
        self.ranking_results = None
        self.predictions = None
        self.rankings = None
        self.b = None
        self.k = None
        self.pk = None

        self.hash_coo = None
        self.csr = None

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
        cube_shape = LFPGraph.three_dim_shape(x, rankings)
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
        shaped_x = LFPGraph.shape_cube_by_rank(x, rankings)
        swapped_x = LFPGraph.swap_tensor_by_rank(shaped_x, rankings)
        zeroed_x, users_x = LFPGraph.zero_users_columns(swapped_x, n_users, axis=2)
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
        two_dims = LFPGraph.two_dim_shape(x, rankings)
        x_two_dim = tf.sparse_reshape(x, shape=two_dims)
        dot_prod = tf.sparse_tensor_dense_matmul(x_two_dim, params)
        three_dims = LFPGraph.three_dim_shape(params, rankings)
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
        pm = LFPGraph.ranking_weights(k, dtype=dtype)
        ranked_dot = dot_prod[:, :k]
        unranked_dot = dot_prod[:, k:]
        a = tf.tensordot(tf.transpose(ranked_dot, perm=(0, 2, 1)), pm, axes=[2, 0])
        b = tf.transpose(unranked_dot, perm=[1, 0, 2]) * variance
        mn = tf.transpose(tf.reduce_sum(a * b, axis=2), perm=[1, 0])
        return mn

    def init_placeholder(self):

        # Training samples and labels
        self.x = tf.sparse_placeholder(shape=[None, self.n_features],
                                       dtype=self.dtype, name='x')
        self.y = tf.placeholder(shape=[None], dtype=self.dtype, name='y')

        # Predictions and Rankings
        self.predictions = tf.placeholder(shape=[None, None], dtype=self.dtype,
                                          name='predictions')
        self.rankings = tf.placeholder(shape=[None, None], dtype=tf.int32,
                                       name='rankings')
        # System-level diversity
        self.b = tf.placeholder(shape=[], dtype=self.dtype, name='b')

        # Rank level
        self.k = tf.placeholder(shape=[], dtype=tf.int32, name='k')

        # Number of users
        self.n_users = tf.placeholder(dtype=tf.int64, shape=[], name='n_users')

        # Number of items
        self.n_items = tf.placeholder(shape=[], dtype=tf.int64, name='n_items')

    def variance_estimate(self, n_users):
        # Variables and tensors initialization
        variance = tf.ones([n_users, self.n_factors], dtype=self.dtype)
        init_var = tf.Variable(variance, name='variance',
                               validate_shape=False,
                               trainable=False)
        init_sum_of_square = tf.Variable(tf.zeros(shape=[n_users, self.n_factors],
                                                  dtype=self.dtype),
                                         name='sum_of_square',
                                         validate_shape=False,
                                         trainable=False)
        init_nu = tf.Variable(tf.zeros(shape=n_users, dtype=tf.int64),
                              name='n_items_per_user',
                              validate_shape=False,
                              trainable=False)
        ones = tf.ones(dtype=tf.int64, shape=tf.shape(self.x)[0])
        u_idx = self.x.indices[:, 1]
        lim_users = tf.constant(n_users, dtype=tf.int64, shape=[1])
        where = tf.less(u_idx, lim_users)
        indexes = tf.reshape(tf.where(where), shape=[-1])
        indexes = tf.nn.embedding_lookup(self.x.indices, indexes)[:, 1]

        # computes the square for the batch (batch_size, n_factors)
        # each row represent the square root for a user
        user_v = tf.nn.embedding_lookup(self.params, indexes)
        dot = tf.sparse_tensor_dense_matmul(self.x, self.params)
        dot = user_v - (dot - user_v)
        sq = tf.square(dot)

        # Nice it should be working
        sum_of_square = tf.scatter_add(init_sum_of_square, indexes, sq)
        nu = tf.scatter_add(init_nu, indexes, ones)
        nu = tf.tile(tf.expand_dims(tf.to_float(nu), 1), [1, self.n_factors])
        computed_variance = sum_of_square / nu
        self.variance = tf.assign(init_var, computed_variance)
        self.init_variance_vars = tf.variables_initializer([init_var,
                                                            init_nu,
                                                            init_sum_of_square])

    def unique_rows_sp_matrix(self, n_rows):
        with tf.name_scope('unique_rows_sparse_matrix'):
            # Placeholders init
            indptr = tf.placeholder(dtype=tf.int64, shape=[None])
            indices = tf.placeholder(dtype=tf.int64, shape=[None])
            data = tf.placeholder(dtype=self.dtype, shape=[None])

            self.csr = (indptr, indices, data)

            # Variables init
            init_i = tf.Variable(tf.constant(0), trainable=False)
            init_col = tf.Variable(tf.constant('', shape=[n_rows, 1]),
                                   name='init_col', trainable=False)
            init_dat = tf.Variable(tf.constant('', shape=[n_rows, 1]),
                                   name='tf_string_data', trainable=False)

            def conditions(_):
                return True

            def body(i):
                min_i = tf.gather(indptr, i)
                max_i = tf.gather(indptr, i + 1)
                tf_slice = tf.range(min_i, max_i)
                string_col = tf.as_string(tf.gather(indices, tf_slice))
                string_dat = tf.as_string(tf.gather(data, tf_slice))
                cols = tf.reduce_join(string_col, separator=':',
                                      keep_dims=True)
                dat = tf.reduce_join(string_dat, separator=':',
                                     keep_dims=True)
                update_col = tf.scatter_update(init_col, i, cols)
                update_dat = tf.scatter_update(init_dat, i, dat)
                with tf.control_dependencies([update_col, update_dat]):
                    return i + 1

            loop = tf.while_loop(conditions, body, [init_i],
                                 maximum_iterations=n_rows,
                                 back_prop=False)
            with tf.control_dependencies([loop]):
                reshape_col = tf.reshape(init_col, shape=[-1])
                reshape_dat = tf.reshape(init_dat, shape=[-1])
            tf_stack_coo = tf.stack([reshape_col, reshape_dat])
            tf_hash_coo = tf.reduce_join(tf_stack_coo, axis=0, separator=',')

            self.init_unique_vars = tf.variables_initializer([init_i,
                                                              init_col,
                                                              init_dat])
            self.hash_coo = tf_hash_coo

    def ranking_computation(self):
        shape = tf.stack([self.n_users, self.n_items])
        predictions = tf.reshape(self.y_hat, shape)
        self.ranking_results = tf.nn.top_k(predictions, k=self.k)

    def delta_f_computation(self):
        zero_x, users_x = self.reshape_dataset(self.n_users, self.rankings, self.x)
        users = users_x.indices[:, 2]
        user_var = self.variance_lookup(users, self.variance)
        self.pk = self.ranking_coefficient(self.k, self.dtype)

        first_term = self.predictions[:, self.k:]
        dot_prod = LFPGraph.dot_product(self.params, self.rankings, zero_x)
        second_term = LFPGraph.second_term(self.k, self.pk, user_var, dot_prod)
        third_term = LFPGraph.third_term(self.k, user_var, dot_prod, self.dtype)
        second_n_third = second_term + 2 * third_term
        delta = self.pk * (first_term - self.b * second_n_third)
        return delta


class BPRLFPGraph(BayesianPersonalizedRankingGraph, LFPGraph):

    def init_placeholder(self):
        BayesianPersonalizedRankingGraph.init_placeholder(self)
        LFPGraph.init_placeholder(self)


class PointwiseLFPGraph(PointwiseGraph, LFPGraph):

    def init_placeholder(self):
        PointwiseGraph.init_placeholder(self)
        LFPGraph.init_placeholder(self)
