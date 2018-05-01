from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.preprocessing import OneHotEncoder
from tfdiv.fm import FMPairwiseRanking
from tfdiv.scorer import MAPScorer
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

users = x.user.values
pos = x[['user', 'item_x']].values
neg = x[['user', 'item_y']].values

train_x = train.values
x = enc.fit(train_x).transform(train_x)
_, n_features = x.shape
csr_pos = enc.transform(pos)
csr_neg = enc.transform(neg)
x.sort_indices()
csr_pos.sort_indices()
csr_neg.sort_indices()

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

fm = FMPairwiseRanking(epochs=epochs,
                       bootstrap_sampling='uniform_user',
                       log_dir="../logs/bpr-epochs-"+str(epochs)+"_size-"+str(batch_size),
                       batch_size=batch_size, tol=1e-10,
                       l2_w=0.01, l2_v=0.01, init_std=0.01)


scorer = MAPScorer(x)

groups = users

print(cross_validate(fm, csr_pos, csr_neg,
                     groups=groups, scoring=scorer,
                     fit_params={'n_features': n_features},
                     cv=GroupKFold(n_splits=3)))

