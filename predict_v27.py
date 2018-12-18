import pandas as pd
import numpy as np
from sklearn import metrics, model_selection
import lightgbm as lgb
import datetime as dt


test_mode = False

all_meta = pd.read_hdf('data\\features_181210_2155\\all_data.hdf5', key='file0')
train_meta_approx = pd.read_hdf('data\\features_181210_2155\\train_meta_approx.hdf5', key='file0')
train_meta_exact = pd.read_hdf('data\\features_181210_2155\\train_meta_exact.hdf5', key='file0')

# map classes to range [0, 14]
classes = np.sort(all_meta.loc[all_meta['target'].notnull(), 'target'].unique().astype(int))
# Train separate models for galatic and extra-galactic, since these classes contain disjoint sets of objects,
# and can be distinguished by whether hostgl_photoz == 0, as observed here:
# https://www.kaggle.com/kyleboone/naive-benchmark-galactic-vs-extragalactic?scriptVersionId=6104036#
galactic_bool = all_meta['hostgal_photoz'] == 0
exact_bool = all_meta['hostgal_photoz'].notnull().astype(int)
galactic_classes = np.sort(all_meta.loc[all_meta['target'].notnull() & galactic_bool, 'target'].unique().astype(int))
non_galactic_classes = np.sort(
    all_meta.loc[all_meta['target'].notnull() & ~galactic_bool, 'target'].unique().astype(int))
for df in [all_meta, train_meta_approx, train_meta_exact]:
    df['target_trans'] = np.nan
    df['target_trans_galactic'] = np.nan
    df['target_trans_non_galactic'] = np.nan
    for k, class_list in enumerate([classes, galactic_classes, non_galactic_classes]):
        if k == 0:
            suffix = ''
        elif k == 1:
            suffix = '_galactic'
        else:
            suffix = '_non_galactic'
        for i in range(len(class_list)):
            df.loc[df['target'] == class_list[i], 'target_trans' + suffix] = i

# train 2 models for each class, one for when we have exact redshift, and another for when we don't
train_cols_exact_redshift \
    = [c for c in train_meta_exact.columns if
      c not in ['object_id', 'ra', 'decl', 'gal_l', 'gal_b', 'target', 'target_trans', 'target_trans_galactic',
                 'target_trans_non_galactic', 'ddf',
                 'distmod', 'mwebv', 'hostgal_photoz', 'hostgal_photoz_err', 'index']]
train_cols_approx_redshift \
    = [c for c in train_meta_approx.columns if
      c not in ['object_id', 'ra', 'decl', 'gal_l', 'gal_b', 'target', 'target_trans', 'target_trans_galactic',
                 'target_trans_non_galactic', 'ddf',
                 'distmod', 'mwebv', 'hostgal_specz', 'index']]

# separate parameters for galactic and non-galactic
params_galactic = {'boosting_type': 'gbdt', 'application': 'binary',
                  'num_leaves': 32, 'seed': 0, 'verbose': -1, 'min_data_in_leaf': 1,
                  'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 1, 'learning_rate': 0.02}
params_non_galactic = {'boosting_type': 'gbdt', 'application': 'binary',
                      'num_leaves': 16, 'seed': 0, 'verbose': -1, 'min_data_in_leaf': 1,
                      'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 1,
                      'learning_rate': 0.02}

num_rounds = 3000

# cross-validate on train set, and measure distribution of out-of-sample predicted values
train_err_exact = []
test_err_exact = []
train_err_approx = []
test_err_approx = []
cv = model_selection.KFold(5, shuffle=True, random_state=4)
galactic_bool_train = train_meta_exact['hostgal_photoz'] == 0
train_meta_exact['predict_max_exact'] = 0
train_meta_exact['predict_max_approx'] = 0
train_meta_approx['predict_max_exact'] = 0
train_meta_approx['predict_max_approx'] = 0
predict_cols = ['class_' + str(c) for c in classes]
train_prediction_exact \
    = pd.DataFrame(np.zeros((len(train_meta_exact), 14)), index=train_meta_exact.index, columns=predict_cols)
train_prediction_approx \
    = pd.DataFrame(np.zeros((len(train_meta_exact), 14)), index=train_meta_exact.index, columns=predict_cols)
eval_prediction_exact \
    = pd.DataFrame(np.zeros((len(train_meta_exact), 14)), index=train_meta_exact.index, columns=predict_cols)
eval_prediction_approx \
    = pd.DataFrame(np.zeros((len(train_meta_exact), 14)), index=train_meta_exact.index, columns=predict_cols)
importance = {}
best_iter_exact = {c: [] for c in classes}
best_iter_approx = {c: [] for c in classes}
# Evaluate accuracy on resampled training set having similar distribution to test. The data note says
# "The training data are mostly composed of nearby, low-redshift, brighter objects while the  test data contain
# more distant (higher redshift) and fainter objects."  So we resample to acheive a similar distribution of
# hostgal_photoz.
train_bool = all_meta['target'].notnull()
ddf = all_meta['ddf'] == 1
w = pd.DataFrame(index=train_meta_exact.index, columns=['galactic', 'non_galactic'])
w['galactic'] = galactic_bool_train.astype(int)
w['non_galactic'] = np.nan
bands = np.arange(all_meta.loc[~train_bool, 'hostgal_photoz'].min(),
                  all_meta.loc[~train_bool, 'hostgal_photoz'].max() + 0.00001, 0.1)
for i in range(len(bands[:-1])):
    band_bool = ~galactic_bool_train & ~ddf & (train_meta_exact['hostgal_photoz'] >= bands[i]) \
                & (train_meta_exact['hostgal_photoz'] <= bands[i + 1])
    train_prop = band_bool.sum() / (~galactic_bool_train & ~ddf).sum()
    test_prop = ((all_meta.loc[~train_bool & ~galactic_bool, 'hostgal_photoz'] >= bands[i])
                 & (all_meta.loc[~train_bool & ~galactic_bool, 'hostgal_photoz'] <= bands[i + 1])).sum() \
                / (~train_bool & ~galactic_bool).sum()
    w.loc[band_bool, 'non_galactic'] = test_prop / train_prop
w.loc[ddf] = 0
for train_ind, test_ind in list(cv.split(train_meta_exact.index, train_meta_exact['target_trans'])):
    train_bool = train_meta_exact.index.isin(train_ind)
    ddf = train_meta_exact['ddf'] == 1

    for i, c in enumerate(classes):
        g = c in galactic_classes
        gal_bool_train_curr = galactic_bool_train == g
        params = params_galactic if g else params_non_galactic
        col = 'class_' + str(c)
        weight_col = 'galactic' if g else 'non_galactic'

        # exact redshift model
        lgb_train = lgb.Dataset(train_meta_exact.loc[train_bool & gal_bool_train_curr, train_cols_exact_redshift],
                                label=(train_meta_exact.loc[train_bool & gal_bool_train_curr, 'target'] == c).astype(int))
        lgb_valid = lgb.Dataset(train_meta_exact.loc[(~train_bool) & gal_bool_train_curr & ~ddf, train_cols_exact_redshift],
                                label=(train_meta_exact.loc[(~train_bool) & gal_bool_train_curr & ~ddf, 'target'] == c).astype(int),
                                weight=w.loc[(~train_bool) & gal_bool_train_curr & ~ddf, weight_col])
        est = lgb.train(train_set=lgb_train, valid_sets=[lgb_train, lgb_valid], valid_names=['train', 'valid'],
                        params=params, num_boost_round=num_rounds, early_stopping_rounds=100)
        best_iter_exact[c].append(est.best_iteration)
        train_prediction_exact.loc[~train_bool & gal_bool_train_curr, col] = est.predict(
            train_meta_exact.loc[(~train_bool) & gal_bool_train_curr, train_cols_exact_redshift],
            num_iteration=est.best_iteration)
        # measure errors on train and test
        eval_prediction_exact.loc[gal_bool_train_curr, col] \
            = est.predict(train_meta_exact.loc[gal_bool_train_curr, train_cols_exact_redshift],
                          num_iteration=est.best_iteration)

        # approx redshift models
        lgb_train = lgb.Dataset(train_meta_approx.loc[train_bool & gal_bool_train_curr, train_cols_approx_redshift],
                                label=(train_meta_approx.loc[train_bool & gal_bool_train_curr, 'target'] == c).astype(int))
        lgb_valid = lgb.Dataset(train_meta_approx.loc[(~train_bool) & gal_bool_train_curr & ~ddf, train_cols_approx_redshift],
                                label=(train_meta_approx.loc[(~train_bool) & gal_bool_train_curr & ~ddf, 'target'] == c).astype(int),
                                weight=w.loc[(~train_bool) & gal_bool_train_curr & ~ddf, weight_col])
        est = lgb.train(train_set=lgb_train, valid_sets=[lgb_train, lgb_valid], valid_names=['train', 'valid'],
                        params=params, num_boost_round=num_rounds, early_stopping_rounds=100)
        best_iter_approx[c].append(est.best_iteration)
        train_prediction_approx.loc[(~train_bool) & gal_bool_train_curr, col] = est.predict(
            train_meta_approx.loc[(~train_bool) & gal_bool_train_curr, train_cols_approx_redshift],
            num_iteration=est.best_iteration)
        # measure errors on train and test
        eval_prediction_approx.loc[gal_bool_train_curr, col] \
            = est.predict(train_meta_approx.loc[gal_bool_train_curr, train_cols_approx_redshift],
                          num_iteration=est.best_iteration)

        imp_arr = est.feature_importance()
        importance[g] = {c: imp_arr[i] for i, c in enumerate(train_cols_approx_redshift)}

    # fill nulls
    for df in [train_prediction_exact, train_prediction_approx, eval_prediction_exact, eval_prediction_approx]:
        df[predict_cols] = df[predict_cols].fillna(0)

    # scale so columns add to 1
    for df in [eval_prediction_exact, eval_prediction_approx]:
        col_sum = df.sum(axis=1)
        for c in df.columns:
            df[c] /= col_sum

    train_err_exact.append(metrics.log_loss(train_meta_exact.loc[train_bool & ~ddf, 'target_trans'],
                                      eval_prediction_exact.loc[train_bool & ~ddf, predict_cols]))
    test_err_exact.append(metrics.log_loss(train_meta_exact.loc[~train_bool & ~ddf, 'target_trans'],
                                     eval_prediction_exact.loc[~train_bool & ~ddf, predict_cols]))
    print('Train exact error: ', train_err_exact)
    print('Test exact error: ', test_err_exact)
    train_err_approx.append(metrics.log_loss(train_meta_approx.loc[train_bool & ~ddf, 'target_trans'],
                                      eval_prediction_approx.loc[train_bool & ~ddf, predict_cols]))
    test_err_approx.append(metrics.log_loss(train_meta_approx.loc[~train_bool & ~ddf, 'target_trans'],
                                     eval_prediction_approx.loc[~train_bool & ~ddf, predict_cols]))
    print('Train approx error: ', train_err_approx)
    print('Test approx error: ', test_err_approx)

with open('log.txt', 'w') as f:
    f.write('Train approx error: {}'.format(train_err_approx))
    f.write('\nTest approx error: {}'.format(test_err_approx))
    f.write('\nMean exact train error: {}'.format(np.mean(train_err_exact)))
    f.write('\nMean exact test error: {}'.format(np.mean(test_err_exact)))
    f.write('\nMean approx train error: {}'.format(np.mean(train_err_approx)))
    f.write('\nMean approx test error: {}'.format(np.mean(test_err_approx)))

if not test_mode:
    prediction = pd.DataFrame(np.zeros((len(all_meta), 14)), columns=predict_cols, index=all_meta.index)
    exact_bool = all_meta['hostgal_specz'].notnull()
    train_bool = all_meta['target'].notnull()

    for i, c in enumerate(classes):
        g = c in galactic_classes
        gal_bool_train_curr = galactic_bool_train == g
        galactic_bool_curr = galactic_bool == g
        params = params_galactic if g else params_non_galactic
        col = 'class_' + str(c)

        if not g:
            # model for exact redshift, only needed for non-galactic objects, since all galactic objects in the
            # test set have approx data
            lgb_train = lgb.Dataset(train_meta_exact.loc[gal_bool_train_curr, train_cols_exact_redshift],
                                    label=(train_meta_exact.loc[gal_bool_train_curr, 'target'] == c).astype(int))
            est_exact_redshift = lgb.train(train_set=lgb_train, valid_sets=[lgb_train], valid_names=['train'],
                                          params=params, num_boost_round=int(np.max(best_iter_exact[c])))
            prediction.loc[exact_bool & galactic_bool_curr, col] \
                = est_exact_redshift.predict(all_meta.loc[exact_bool & galactic_bool_curr, train_cols_exact_redshift])

        # model for approx redshift
        lgb_train = lgb.Dataset(train_meta_approx.loc[gal_bool_train_curr, train_cols_approx_redshift],
                                label=(train_meta_approx.loc[gal_bool_train_curr, 'target'] == c).astype(int))
        est_approx_redshift = lgb.train(train_set=lgb_train, valid_sets=[lgb_train], valid_names=['train'],
                                        params=params, num_boost_round=int(np.max(best_iter_approx[c])))
        prediction.loc[~exact_bool & galactic_bool_curr, col] \
            = est_approx_redshift.predict(all_meta.loc[~exact_bool & galactic_bool_curr, train_cols_approx_redshift])

    # fill nulls
    prediction[predict_cols] = prediction[predict_cols].fillna(0)

    # We will calculate the probability that the object is class 99 using 1 - max(other columns). But this is an overestimate,
    # since the max is always less than 1, even in the training set. So adjust for this by comparing to the max
    # in the training set, resampled as before to account for the different distribution.
    # Different distributions for each combination of galactic/non-galactic and approx/exact. All galactic objects
    # have approx redshift, so only need three combinations. First add the target column.
    train_prediction_exact['target'] = train_meta_exact['target']
    train_prediction_approx['target'] = train_meta_approx['target']
    train_approx_galactic_resample \
        = train_prediction_approx.sample(n=10000, weights=w['galactic'], replace=True, random_state=0)
    train_exact_non_galactic_resample \
        = train_prediction_exact.sample(n=10000, weights=w['non_galactic'], replace=True, random_state=0)
    train_approx_non_galactic_resample \
        = train_prediction_approx.sample(n=10000, weights=w['non_galactic'], replace=True, random_state=0)

    # predict class 99
    prediction['class_99'] = 1 - prediction[predict_cols].max(axis=1)
    # adjust as described above
    for exact in [True, False]:
        for g in [True, False]:
            if exact and g:
                continue
            prediction_ind = ~train_bool & (exact_bool == exact) & (galactic_bool == g)
            if exact:
                train_avg_max = train_exact_non_galactic_resample[predict_cols].max(axis=1).mean()
            else:
                if g:
                    train_avg_max = train_approx_galactic_resample[predict_cols].max(axis=1).mean()
                else:
                    train_avg_max = train_approx_non_galactic_resample[predict_cols].max(axis=1).mean()
            if (1 - train_avg_max) < prediction.loc[prediction_ind, 'class_99'].mean():
                old_avg = prediction.loc[prediction_ind, 'class_99'].mean()
                new_avg = old_avg - (1 - train_avg_max)
                prediction.loc[prediction_ind, 'class_99'] *= new_avg / old_avg

    # write output
    prediction[predict_cols + ['class_99']] = prediction[predict_cols + ['class_99']]
    prediction['object_id'] = all_meta['object_id']
    filename = 'Prediction_raw_{}.csv'.format(dt.datetime.now().strftime('%y%m%d_%H%M'))
    prediction.loc[~train_bool, ['object_id'] + predict_cols + ['class_99']].to_csv(filename, index=False, header=True)

# Train exact error:  [0.07335417740378421, 0.07100410182122908, 0.07059803460300715, 0.07224002015707873, 0.06944762026486369]
# Test exact error:  [0.3748601238430611, 0.36556888133295085, 0.39131388860123373, 0.38304939098733926, 0.41699428530911387]
# Train approx error:  [0.09506613940490972, 0.0933585068779037, 0.09334975604237751, 0.09505441312509609, 0.09147455895104299]
# Test approx error:  [0.4480470146597202, 0.4384931283708287, 0.47894630719969605, 0.47405030414662697, 0.48429064624925044]
# Mean exact train error:  0.07132879084999257
# Mean exact test error:  0.3863573140147397
# Mean approx train error:  0.093660674880266
# Mean approx test error:  0.46476548012522445