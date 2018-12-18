"""Scale raw predictions to add to 1 and apply regularisation."""
import pandas as pd
import numpy as np
import datetime as dt


# read raw predictions
prediction = pd.read_csv('Prediction_raw_181215_1818.csv', dtype={'object_id': np.int32})
col_dict = {'mjd': np.float64, 'flux': np.float32, 'flux_err': np.float32, 'object_id': np.int32, 'passband': np.int8,
            'detected': np.int8}
test_meta = pd.read_csv('data\\test_set_metadata.csv', dtype=col_dict)
test_meta['galactic'] = test_meta['hostgal_photoz'] == 0
test_meta['exact'] = test_meta['hostgal_specz'].notnull()
prediction = pd.merge(prediction, test_meta[['object_id', 'galactic', 'exact']], 'left', 'object_id')

# Regularise class 99 prediction
# Use separate mean for galactic/non-galactic and approx/exact, since the predicted averages are different
# (which also accords with the claims in the following thread that class 99 occurs much less frequently for galactic
# objects compared to extra-galactic):
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/68943
alpha = 0.5  # regularisation parameter, between 0 and 1; small alpha = more regularisation
mean_99 = prediction.groupby(['exact', 'galactic'])['class_99'].transform('mean')
prediction['class_99'] \
    = mean_99 + alpha * (prediction['class_99'] - mean_99)
prediction.drop(['exact', 'galactic'], axis=1, inplace=True)

# scale so remaining columns sum to 1 - Pr(class_99)
predict_cols = [c for c in prediction.columns if (c not in ['object_id', 'class_99'])]
predict_sum = prediction[predict_cols].sum(axis=1)
for c in predict_cols:
    prediction[c] *= (1 - prediction['class_99']) / predict_sum

# calculate the weights
# losses from https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
losses = [32.62, 30.702, 32.62, 32.62, 32.62, 32.622, 32.62, 30.692, 32.62, 32.62, 32.62, 32.62, 32.62, 32.62, 30.701]
# Assume WLOG that the weight of class 0 is 1, and use the above losses to calculate the other weights
p_min = np.log(10 ** -15)
# It can be shown that if classes c, d have losses L_c, L_d, then the ratio of their weights is
# w_c / w_d = (L_c + p_min) / (L_d + p_max)
w = [(loss + p_min) / (32.62 + p_min) for loss in losses]
weights = pd.DataFrame(w, columns=['W'])
# (As described in the following post, a close approximation is given by setting all weights to 1, except for
# classes 15, 64, 99 which have weight 2; this is consistent with the weights found above)
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194#397153

# estimate proportions of class 99
weights['N'] = prediction[predict_cols + ['class_99']].mean().values
weights['N'] /= weights['N'].sum()

# scale using the class weights
for i, c in enumerate(predict_cols + ['class_99']):
    prediction[c] *= weights.loc[i, 'W'] / weights.loc[i, 'N']

# scale so columns sum to 1
predict_sum = prediction[predict_cols + ['class_99']].sum(axis=1)
for c in predict_cols + ['class_99']:
    prediction[c] /= predict_sum

# write output
prediction[predict_cols + ['class_99']] = prediction[predict_cols + ['class_99']].astype(np.float16)
filename = 'Submission_alpha_{}_{}.csv'.format(alpha, dt.datetime.now().strftime('%y%m%d_%H%M'))
prediction[['object_id'] + predict_cols + ['class_99']].to_csv(filename, index=False, header=True)
