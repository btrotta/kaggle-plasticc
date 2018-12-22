"""Split the test data into chunks."""

import numpy as np
import pandas as pd
import os

n_chunks = 100
if not os.path.exists(os.path.join('data', 'split_{}'.format(n_chunks))):
    os.mkdir(os.path.join('data', 'split_{}'.format(n_chunks)))

col_dict = {'mjd': np.float64, 'flux': np.float32, 'flux_err': np.float32, 'object_id': np.int32, 'passband': np.int8,
            'detected': np.int8}
test = pd.read_csv(os.path.join('data', 'test_set.csv'), dtype=col_dict)
test.sort_values('object_id', inplace=True)
test = test.reset_index()
test_len = len(test)

id_diff = test.loc[test['object_id'].diff() != 0].index
chunk_starts = [id_diff[int(len(id_diff) * i / n_chunks)] for i in range(n_chunks)]
for i in range(n_chunks):
    if i == n_chunks - 1:
        end = len(test)
    else:
        end = chunk_starts[i + 1]
    test.iloc[chunk_starts[i]: end - 1].to_hdf(os.path.join('data', 'split_{}'.format(n_chunks),
                                                            'chunk_{}.hdf5'.format(i)), key='file0')

