from sklearn.preprocessing import OneHotEncoder
from tfdiv.fm import BayesianPersonalizedRankingLFP
from tfdiv.utility import cartesian_product
import pandas as pd
import numpy as np
import sys

PATH = "~/PycharmProjects/DivMachines/data/ua.base"
columns = ['user', 'item', 'rating', 'timestamp']

train = pd.read_csv(PATH, delimiter='\t', names=columns).sample(1000)[columns[:-2]]

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

x = pd.merge(pos[['user', 'item']], neg, on='user')

pos = x[['user', 'item_x']].values
neg = x[['user', 'item_y']].values

train_x = train.values
x = enc.fit(train_x).transform(train_x)
_, n_features = x.shape
pos = enc.transform(pos)
neg = enc.transform(neg)
x.sort_indices()
pos.sort_indices()
neg.sort_indices()

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

fm = BayesianPersonalizedRankingLFP(epochs=epochs,
                                    bootstrap_sampling='uniform_user',
                                    log_dir="../logs/bpr-"+str(epochs)+"_size-"+str(batch_size),
                                    batch_size=batch_size, tol=1e-4, frac=0.7,
                                    l2_w=0.01, l2_v=0.01, init_std=0.01)
fm.fit(pos, neg, n_users, n_items)

x = enc.transform(cartesian_product(train.user.unique(), train.item.unique()))

a = fm.predict(x, n_users, n_items)

print(a)