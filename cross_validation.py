from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.scorer import mean_squared_error_scorer
from sklearn.model_selection import cross_validate, KFold
from fm import FMRegression
import pandas as pd
import numpy as np
import sys

PATH = "~/PycharmProjects/DivMachines/data/ua.base"
columns = ['user', 'item', 'rating', 'timestamp']

train = pd.read_csv(PATH, delimiter='\t', names=columns)[columns[:-1]]

n_users = train.user.unique().shape[0]
n_items = train.item.unique().shape[0]

enc = OneHotEncoder(categorical_features=[0, 1],
                    dtype=np.float32)

train.loc[train['rating'] < 4, 'rating'] = 0
train.loc[train['rating'] > 3, 'rating'] = 1
x = train.values[:, :-1]
y = train.values[:, -1]

csr = enc.fit(x).transform(x)
csr.sort_indices()

print(csr.shape[0])
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

fm = FMRegression(epochs=epochs,
                  log_dir="./logs/epochs-"+str(epochs)+"_size-"+str(batch_size),
                  batch_size=batch_size,
                  l2_w=0.01, l2_v=0.01, init_std=0.1)

print(cross_validate(fm,
                     X=csr,
                     y=y,
                     scoring=mean_squared_error_scorer,
                     n_jobs=1,
                     cv=KFold(n_splits=5)))
