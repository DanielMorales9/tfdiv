from tfdiv.utility import ranked_relevance_feedback
from collections import defaultdict
import numpy as np


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def dcg_at_k(relevance, topics):

    scores = np.zeros(relevance.shape[0], dtype=np.float)
    for u, top in enumerate(relevance):
        tops = set([])

        for i, item in enumerate(top):
            wi = 1/(2**i)
            if topics[item] is not None:
                scores[u] += len(set(topics[item]).difference(tops))*wi
                tops.update(topics[item])
    return np.mean(scores)


def dcg_at_k_with_importance(relevance, topics, importance):

    scores = np.zeros(relevance.shape[0], dtype=np.float)
    for u, top in enumerate(relevance):
        tops = []
        for i, item in enumerate(top):
            wi = 1/(2**i)
            if topics[item] is not None:
                scores[u] += (importance[item]**tops.count(topics[item])) * wi
                tops.append(topics[item])
    return np.mean(scores)


def alpha_ndcg_at_k(alpha, rankings, relevance, topics):
    k = rankings.shape[1]
    ranked_rel = ranked_relevance_feedback(rankings, relevance)
    _alpha_dcg_at_k = alpha_dcg_at_k(alpha, ranked_rel, rankings, topics)
    _alpha_ideal_dcg_at_k = alpha_ideal_dcg_at_k(alpha, relevance, topics, k)
    return _alpha_dcg_at_k / _alpha_ideal_dcg_at_k


def alpha_dcg_at_k(alpha, ranked_rel, rankings, topics):
    _alpha_dcg_at_k = np.zeros(rankings.shape[0], np.float)
    for u, (rel, rank) in enumerate(zip(ranked_rel, rankings)):
        cum = 0.0
        q = defaultdict(lambda : 1)
        for k, (r, i) in enumerate(zip(rel, rank)):
            j = 0.0
            for g in topics[i]:
                j += r * (1 - alpha) ** q[g]
                q[g] += 1
            cum += (j / np.log2(k+2))
        _alpha_dcg_at_k[u] = cum
    return _alpha_dcg_at_k


def alpha_ideal_dcg_at_k(alpha, relevance, topics, k):
    # Greedy approach
    dcgs = np.zeros(relevance.shape[0], np.float)
    for user, user_relevance in enumerate(relevance):
        topic_gain = defaultdict(lambda: 1.0)
        dropped_out = []
        for rank in range(k):
            # Compute gain
            g = [ideal_gain(topics[id], topic_gain)
                 if item == 1 and id not in dropped_out
                 else 0.0 for id, item in enumerate(user_relevance)]
            item_id = np.argmax(g)
            max_g = max(g)

            # Increment topic ids
            if max_g > 0:
                for tid in topics[item_id]:
                    topic_gain[tid] *= 1.0 - alpha
            dropped_out.append(item_id)
            dg = max_g / np.log2(k+2)
            dcgs[user] += dg
    return dcgs


def ideal_gain(topics, topic_gain):
    _gain = 0.0
    for tid in topics:
        _gain += topic_gain[tid]
    return _gain
