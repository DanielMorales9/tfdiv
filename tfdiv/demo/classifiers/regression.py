from sklearn.preprocessing import OneHotEncoder
from tfdiv.fm import FMRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import sys

PATH = "~/PycharmProjects/DivMachines/data/ua.base"
columns = ['user', 'item', 'rating', 'timestamp']

train = pd.read_csv(PATH, delimiter='\t', names=columns)[columns[:-1]]

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

fm = FMRegression(epochs=epochs,
                  log_dir="../logs/regression-"+str(epochs)+"_size-"+str(batch_size),
                  batch_size=batch_size, tol=1e-10,
                  l2_w=0.01, l2_v=0.01, init_std=0.01)

fm.fit(csr, y)

y_hat = fm.predict(csr)

print(mean_squared_error(y, y_hat))

diff = np.zeros((y.shape[0], 2))
diff[:, 0] = y
diff[:, 1] = y_hat

print(diff)
