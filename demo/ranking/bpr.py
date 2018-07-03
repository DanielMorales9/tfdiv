from sklearn.preprocessing import OneHotEncoder
from tfdiv.fm import BayesianPersonalizedRanking
from tfdiv.utility import cartesian_product
import pandas as pd
import numpy as np
import sys

PATH = "~/PycharmProjects/DivMachines/data/ua.base"
columns = ['user', 'item', 'rating', 'timestamp']

train = pd.read_csv(PATH, delimiter='\t', names=columns).sample(10000)[columns[:-2]]

n_users = train.user.unique().shape[0]
n_items = train.item.unique().shape[0]

enc = OneHotEncoder(dtype=np.float32)

pos = train

df = {'user': [], 'item': []}
cat = set(train.item.unique())

for u in train.user.unique():
    items = set(train.loc[train['user'] == u, 'item'])
    neg = cat - items
    size = len(neg)
    df['user'].append(np.repeat(u, size))
    df['item'].append(list(neg))

df['user'] = np.concatenate(df['user'])
df['item'] = np.concatenate(df['item'])

neg = pd.DataFrame(df)

pos = pos[['user', 'item']].values
neg = neg[['user', 'item']].values

train_x = train.values
x = enc.fit(train_x).transform(train_x)
_, n_features = x.shape
pos = enc.transform(pos)
neg = enc.transform(neg)
x.sort_indices()
pos.sort_indices()
neg.sort_indices()

epochs = 100
batch_size = 100000
k = 10

import tensorflow as tf
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 10
config.inter_op_parallelism_threads = 10

fm = BayesianPersonalizedRanking(epochs=epochs,
                                 frac=0.5,
                                 shuffle_size=10,
                                 n_threads=10,
                                 bootstrap_sampling='uniform_user',
                                 batch_size=batch_size,
                                 l2_w=0.01, l2_v=0.01, init_std=0.01)
fm.fit(pos, neg)

x = enc.transform(cartesian_product(train.user.unique(), train.item.unique()))

a, b = fm.predict(x, n_users, n_items)

print(a)
print(b)
