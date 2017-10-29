import numpy as np
import scipy.misc # to visualize only
import pandas as pd

dfmaster = pd.read_csv('train_x.csv', nrows = 10, header = None)


def cleaner(anydf):
    cutoff = 254
    for col in range(len(anydf.columns)):
        anydf.ix[anydf.loc[:,col]>cutoff,col] = 255
        anydf.ix[anydf.loc[:,col]<cutoff,col] = 0
    return anydf


dfmaster2 = dfmaster
df = cleaner(dfmaster2)
df2 = pd.read_csv('train_x.csv', nrows = 10, header = None)


choice = 3
x = np.array(df.iloc[choice,0:4096])
x = x.reshape(-1,64,64)
scipy.misc.imshow(x[0])

y = np.array(df2.iloc[choice,0:4096])
y = y.reshape(-1,64,64)
scipy.misc.imshow(y[0])
