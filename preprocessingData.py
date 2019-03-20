import numpy as np
import sys
import pandas as pd

MAX_ROWS_TO_USE = -1
DELAY_CUTOFF = 15.0

def normalize(col):
    if col.std() != 0:
        col = (col - col.mean()) / col.std()
    return col

def preprocess(filename):
    data = pd.read_csv(filename)
    data = data.fillna(0)
    values = np.copy(data.values)
    # the following columns are normalized using zero-mean, unit-variance
    weathers = ['time', 'tmpc', 'dwpc', 'relh', 'feel', 'drct', 'sped', 'alti', 'mslp', 'p01m', 'vsby', 'gust_mph', 'skyl1']
    # we also normalize the corresponding arrival versions of these columns
    arrivalWeathers = list(['a_' + x for x in weathers])
    toNormalize = ['year', 'day', 'time', 'a_time', 'OP_CARRIER_FL_NUM', 'DEP_DELAY']
    toNormalize += weathers
    toNormalize += arrivalWeathers
    # toNormalize = ['year', 'day', 'time', 'tmpc', 'dwpc', 'relh', 'feel', 'drct', 'sped', 'alti', 'mslp', 'p01m', 'vsby', 'gust_mph', 'skyl1', 'join_time', 'OP_CARRIER_FL_NUM', 'DEP_DELAY', 'CRS_ARR_TIME']
    indices = list(data.columns.values)
    indexOfArrDelay = indices.index('ARR_DELAY')
    # we convert flight delays to a category variable based on whether the delay was greater than or equal to 15 minutes
    values[:, indexOfArrDelay] = np.vectorize(lambda x: 1 if x >= DELAY_CUTOFF else 0)(data.values[:, indexOfArrDelay])
    for n in toNormalize:
        i = indices.index(n)
        values[:, i] = normalize(data.values[:, i])
    values = pd.DataFrame(data=values, index=data.index, columns=data.columns)
    return values

assert(len(sys.argv) > 2)
filename = sys.argv[1]
data = preprocess(filename)
data.to_csv(sys.argv[2]) 
