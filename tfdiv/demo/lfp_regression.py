import numpy as np
import pandas as pd
from tfdiv.fm import FMRegressionLFP
from sklearn.preprocessing import OneHotEncoder

PATH = "~/PycharmProjects/DivMachines/data/ua.base"
columns = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv(PATH, delimiter='\t', names=columns)[columns[:-1]]
n_users = train.user.unique().shape[0]
n_items = train.item.unique().shape[0]


train.loc[train['rating'] < 4, 'rating'] = 0
train.loc[train['rating'] > 3, 'rating'] = 1
enc = OneHotEncoder(categorical_features=[0, 1],
                    dtype=np.float32)
x = train.values[:, :-1]
y = train.values[:, -1]
csr = enc.fit(x).transform(x)

FMRegressionLFP(epochs=1, batch_size=32).fit(csr, y)
