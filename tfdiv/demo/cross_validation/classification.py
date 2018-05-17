from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import OneHotEncoder
from tfdiv.fm import Classification
from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
import numpy as np
import sys

PATH = "~/PycharmProjects/DivMachines/data/ua.base"
columns = ['user', 'item', 'rating', 'timestamp']

train = pd.read_csv(PATH, delimiter='\t', names=columns)[columns[:-1]].sample(1000)

n_users = train.user.unique().shape[0]
n_items = train.item.unique().shape[0]

enc = OneHotEncoder(categorical_features=[0, 1],
                    dtype=np.float32)

x = train.values[:, :-1]
train.loc[train['rating'] < 4, 'rating'] = 0
train.loc[train['rating'] > 3, 'rating'] = 1
y = train.values[:, -1]

csr = enc.fit(x).transform(x)
csr.sort_indices()

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

fm = Classification(epochs=epochs,
                    log_dir="../logs/classifier-"+str(epochs)+"_size-"+str(batch_size),
                    batch_size=batch_size, tol=1e-10,
                    l2_w=0, l2_v=0, init_std=0.01)
mse = make_scorer(mean_squared_error)
print(cross_validate(fm, csr, y, scoring=mse, cv=KFold(n_splits=3)))
