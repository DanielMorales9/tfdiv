from sklearn.preprocessing import OneHotEncoder
from tfdiv.fm import FMClassificationLFP
from tfdiv.utility import cartesian_product
import pandas as pd
import numpy as np
import sys

PATH = "~/PycharmProjects/DivMachines/data/ua.base"
columns = ['user', 'item', 'rating', 'timestamp']

train = pd.read_csv(PATH, delimiter='\t', names=columns)[columns[:-1]].sample(100)

n_users = train.user.unique().shape[0]
n_items = train.item.unique().shape[0]

enc = OneHotEncoder(categorical_features=[0, 1],
                    dtype=np.float32)

x = train.values[:, :-1]
train.loc[train['rating'] < 4, 'rating'] = 0
train.loc[train['rating'] > 3, 'rating'] = 1
y = train.values[:, -1]

csr = enc.fit(x).transform(x)

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

fm = FMClassificationLFP(epochs=epochs,
                         log_dir="../logs/classification-"+str(epochs)+"_size-"+str(batch_size),
                         batch_size=batch_size, tol=1e-10,
                         l2_w=0.01, l2_v=0.01, init_std=0.01)

fm.fit(csr, y, n_users, n_items)

x = enc.transform(cartesian_product(train.user.unique(), train.item.unique()))

a = fm.predict(x, n_users, n_items)

print(a)
