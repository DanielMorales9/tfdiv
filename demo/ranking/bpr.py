from sklearn.preprocessing import OneHotEncoder
from tfdiv.fm import BayesianPersonalizedRanking
from tfdiv.utility import cartesian_product
import pandas as pd
import numpy as np
import sys

PATH = "~/PycharmProjects/DivMachines/data/ua.base"
columns = ['user', 'item', 'rating', 'timestamp']

train = pd.read_csv(PATH, delimiter='\t', names=columns)[columns[:-2]]

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

epochs = int(sys.argv[1])
batch_size = 10000
k = int(sys.argv[3])

import tensorflow as tf
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 2
config.inter_op_parallelism_threads = 2

fm = BayesianPersonalizedRanking(epochs=epochs,
                                 n_threads=2,
                                 shuffle_size=1000000,
                                 frac=0.7,
                                 bootstrap_sampling='uniform_user',
                                 batch_size=batch_size, tol=1e-4,
                                 l2_w=0.01, l2_v=0.01, init_std=0.01)
fm.fit(pos, neg)

x = enc.transform(cartesian_product(train.user.unique(), train.item.unique()))

a, b = fm.predict(x, n_users, n_items)

print(a)
print(b)
