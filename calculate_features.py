import pandas as pd
import numpy as np
import gc
import os


# read data
col_dict = {'mjd': np.float64, 'flux': np.float32, 'flux_err': np.float32, 'object_id': np.int32, 'passband': np.int8,
            'detected': np.int8}
train_meta = pd.read_csv(os.path.join('data', 'training_set_metadata.csv'))
train = pd.read_csv(os.path.join('data', 'training_set.csv'), dtype=col_dict)


def calc_aggs(all_data, exact):

    # Normalise the flux, following the Bayesian approach here:
    # https://www.statlect.com/fundamentals-of-statistics/normal-distribution-Bayesian-estimation
    # Similar idea (but not the same) as the normalisation done in the Starter Kit
    # https://www.kaggle.com/michaelapers/the-plasticc-astronomy-starter-kit?scriptVersionId=6040398
    prior_mean = all_data.groupby(['object_id', 'passband'])['flux'].transform('mean')
    prior_std = all_data.groupby(['object_id', 'passband'])['flux'].transform('std')
    prior_std.loc[prior_std.isnull()] = all_data.loc[prior_std.isnull(), 'flux_err']
    obs_std = all_data['flux_err']  # since the above kernel tells us that the flux error is the 68% confidence interval
    all_data['bayes_flux'] = (all_data['flux'] / obs_std**2 + prior_mean / prior_std**2) \
                             / (1 / obs_std**2 + 1 / prior_std**2)
    all_data.loc[all_data['bayes_flux'].notnull(), 'flux'] \
        = all_data.loc[all_data['bayes_flux'].notnull(), 'bayes_flux']

    # Estimate the flux at source, using the fact that light is proportional
    # to inverse square of distance from source.
    # This is hinted at here: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70725#417195
    redshift = all_meta.set_index('object_id')[['hostgal_specz', 'hostgal_photoz']]
    if exact:
        redshift['redshift'] = redshift['hostgal_specz']
        redshift.loc[redshift['redshift'].isnull(), 'redshift'] \
            = redshift.loc[redshift['redshift'].isnull(), 'hostgal_photoz']
    else:
        redshift['redshift'] = redshift['hostgal_photoz']
    all_data = pd.merge(all_data, redshift, 'left', 'object_id')
    nonzero_redshift = all_data['redshift'] > 0
    all_data.loc[nonzero_redshift, 'flux'] = all_data.loc[nonzero_redshift, 'flux'] \
                                             * all_data.loc[nonzero_redshift, 'redshift']**2

    # aggregate features
    band_aggs = all_data.groupby(['object_id', 'passband'])['flux'].agg(['mean', 'std', 'max', 'min']).unstack(-1)
    band_aggs.columns = [x + '_' + str(y) for x in band_aggs.columns.levels[0]
                          for y in band_aggs.columns.levels[1]]
    all_data.sort_values(['object_id', 'passband', 'flux'], inplace=True)
    # this way of calculating quantiles is faster than using the pandas quantile builtin on the groupby object
    all_data['group_count'] = all_data.groupby(['object_id', 'passband']).cumcount()
    all_data['group_size'] = all_data.groupby(['object_id', 'passband'])['flux'].transform('size')
    q_list = [0.25, 0.75]
    for q in q_list:
        all_data['q_' + str(q)] = all_data.loc[
            (all_data['group_size'] * q).astype(int) == all_data['group_count'], 'flux']
    quantiles = all_data.groupby(['object_id', 'passband'])[['q_' + str(q) for q in q_list]].max().unstack(-1)
    quantiles.columns = [str(x) + '_' + str(y) + '_quantile' for x in quantiles.columns.levels[0]
                         for y in quantiles.columns.levels[1]]

    # max detected flux
    max_detected = all_data.loc[all_data['detected'] == 1].groupby('object_id')['flux'].max().to_frame('max_detected')

    def most_extreme(df_in, k, positive=True, suffix='', include_max=True, include_dur=True, include_interval=False):
        # find the "most extreme" time for each object, and for each band, retrieve the k data points on either side
        # k points before
        df = df_in.copy()
        df['object_passband_mean'] = df.groupby(['object_id', 'passband'])['flux'].transform('median')
        if positive:
            df['dist_from_mean'] = (df['flux'] - df['object_passband_mean'])
        else:
            df['dist_from_mean'] = -(df['flux'] - df['object_passband_mean'])

        max_time = df.loc[df['detected'] == 1].groupby('object_id')['dist_from_mean'].idxmax().to_frame(
            'max_ind')
        max_time['mjd_max' + suffix] = df.loc[max_time['max_ind'].values, 'mjd'].values
        df = pd.merge(df, max_time[['mjd_max' + suffix]], 'left', left_on=['object_id'], right_index=True)
        df['time_after_mjd_max'] = df['mjd'] - df['mjd_max' + suffix]
        df['time_before_mjd_max'] = -df['time_after_mjd_max']

        # first k after event
        df.sort_values(['object_id', 'passband', 'time_after_mjd_max'], inplace=True)
        df['row_num_after'] = df.loc[df['time_after_mjd_max'] >= 0].groupby(
            ['object_id', 'passband']).cumcount()
        first_k_after = df.loc[(df['row_num_after'] < k) & (df['time_after_mjd_max'] <= 50),
                              ['object_id', 'passband', 'flux', 'row_num_after']]
        first_k_after.set_index(['object_id', 'passband', 'row_num_after'], inplace=True)
        first_k_after = first_k_after.unstack(level=-1).unstack(level=-1)
        first_k_after.columns = [str(x) + '_' + str(y) + '_after' for x in first_k_after.columns.levels[1]
                                 for y in first_k_after.columns.levels[2]]
        extreme_data = first_k_after
        time_bands = [[-50, -20], [-20, -10], [-10, 0], [0, 10], [10, 20], [20, 50], [50, 100], [100, 200], [200, 500]]
        if include_interval:
            interval_arr = []
            for start, end in time_bands:
                band_data = df.loc[(start <= df['time_after_mjd_max']) & (df['time_after_mjd_max'] <= end)]
                interval_agg = band_data.groupby(['object_id', 'passband'])['flux'].mean().unstack(-1)
                interval_agg.columns = ['{}_start_{}_end_{}'.format(c, start, end) for c in interval_agg.columns]
                interval_arr.append(interval_agg)
            interval_data = pd.concat(interval_arr, axis=1)
            extreme_data = pd.concat([extreme_data, interval_data], axis=1)
        if include_dur:
            # detection duration in each passband after event
            duration_after = df.loc[(df['time_after_mjd_max'] >= 0) & (df['detected'] == 0)] \
                .groupby(['object_id', 'passband'])['time_after_mjd_max'].first().unstack(-1)
            duration_after.columns = ['dur_after_' + str(c) for c in range(6)]
            extreme_data = pd.concat([extreme_data, duration_after], axis=1)

        # last k before event
        df.sort_values(['object_id', 'passband', 'time_before_mjd_max'], inplace=True)
        df['row_num_before'] = df.loc[df['time_before_mjd_max'] >= 0].groupby(
            ['object_id', 'passband']).cumcount()
        first_k_before = df.loc[(df['row_num_before'] < k) & (df['time_after_mjd_max'] <= 50),
                                ['object_id', 'passband', 'flux', 'row_num_before']]
        first_k_before.set_index(['object_id', 'passband', 'row_num_before'], inplace=True)
        first_k_before = first_k_before.unstack(level=-1).unstack(level=-1)
        first_k_before.columns = [str(x) + '_' + str(y) + '_before' for x in first_k_before.columns.levels[1]
                                  for y in first_k_before.columns.levels[2]]
        extreme_data = pd.concat([extreme_data, first_k_before], axis=1)
        if include_dur:
            # detection duration in each passband before event
            duration_before = df.loc[(df['time_before_mjd_max'] >= 0) & (df['detected'] == 0)] \
                .groupby(['object_id', 'passband'])['time_before_mjd_max'].first().unstack(-1)
            duration_before.columns = ['dur_before_' + str(c) for c in range(6)]
            extreme_data = pd.concat([extreme_data, duration_before], axis=1)

        if include_max:
            # passband with maximum detected flux for each object
            max_pb = df.loc[max_time['max_ind'].values].groupby('object_id')['passband'].max().to_frame(
                'max_passband')
            # time of max in each passband, relative to extreme max
            band_max_ind = df.groupby(['object_id', 'passband'])['flux'].idxmax()
            band_mjd_max = df.loc[band_max_ind.values].groupby(['object_id', 'passband'])['mjd'].max().unstack(-1)
            cols = ['max_time_' + str(i) for i in range(6)]
            band_mjd_max.columns = cols
            band_mjd_max = pd.merge(band_mjd_max, max_time, 'left', 'object_id')
            for c in cols:
                band_mjd_max[c] -= band_mjd_max['mjd_max' + suffix]
            band_mjd_max.drop(['mjd_max' + suffix, 'max_ind'], axis=1, inplace=True)
            extreme_data = pd.concat([extreme_data, max_pb, band_mjd_max], axis=1)

        extreme_data.columns = [c + suffix for c in extreme_data.columns]
        return extreme_data

    extreme_max = most_extreme(all_data, 1, positive=True, suffix='', include_max=True, include_dur=True,
                               include_interval=True)
    extreme_min = most_extreme(all_data, 1, positive=False, suffix='_min', include_max=False, include_dur=True)

    # add the feature mentioned here, attempts to identify periodicity:
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    time_between_detections = all_data.loc[all_data['detected'] == 1].groupby('object_id')['mjd'].agg(['max', 'min'])
    time_between_detections['det_period'] = time_between_detections['max'] - time_between_detections['min']
    # same feature but grouped by passband
    time_between_detections_pb \
        = all_data.loc[all_data['detected'] == 1].groupby(['object_id', 'passband'])['mjd'].agg(['max', 'min'])
    time_between_detections_pb['det_period'] = time_between_detections_pb['max'] - time_between_detections_pb['min']
    time_between_detections_pb = time_between_detections_pb['det_period'].unstack(-1)
    time_between_detections_pb.columns = ['det_period_pb_' + str(i) for i in range(6)]
    # similar feature based on high values
    all_data['threshold'] = all_data.groupby(['object_id'])['flux'].transform('max') * 0.75
    all_data['high'] = ((all_data['flux'] >= all_data['threshold']) & (all_data['detected'] == 1)).astype(int)
    time_between_highs = all_data.loc[all_data['high'] == 1].groupby('object_id')['mjd'].agg(['max', 'min'])
    time_between_highs['det_period_high'] = time_between_highs['max'] - time_between_highs['min']

    # aggregate values of the features during the detection period
    all_data = pd.merge(all_data, time_between_detections, 'left', 'object_id')
    det_data = all_data.loc[(all_data['mjd'] >= all_data['min']) & (all_data['mjd'] <= all_data['max'])]
    det_aggs = det_data.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max', 'std', 'median'])
    det_aggs['prop_detected'] = det_data.groupby(['object_id', 'passband'])['detected'].mean()
    det_aggs = det_aggs.unstack(-1)
    det_aggs.columns = [x + '_' + str(y) + '_det_period' for x in det_aggs.columns.levels[0]
                          for y in det_aggs.columns.levels[1]]

    # time distribution of detections in each band
    detection_time_dist \
        = all_data.loc[all_data['detected'] == 1].groupby(['object_id', 'passband'])['mjd'].std().unstack(-1)
    detection_time_dist.columns = ['time_dist_' + str(i) for i in range(6)]
    detection_time_dist_all \
        = all_data.loc[all_data['detected'] == 1].groupby(['object_id'])['mjd'].std().to_frame('time_dist')

    # scale data and recalculate band aggs
    all_data['abs_flux'] = all_data['flux'].abs()
    all_data['flux'] = (all_data['flux']) / all_data.groupby('object_id')['abs_flux'].transform('max')
    band_aggs_s = all_data.groupby(['object_id', 'passband'])['flux'].agg(['mean', 'std', 'max', 'min']).unstack(-1)
    band_aggs_s.columns = [x + '_' + str(y) + '_scaled' for x in band_aggs_s.columns.levels[0]
                          for y in band_aggs_s.columns.levels[1]]
    all_data.sort_values(['object_id', 'passband', 'flux'], inplace=True)
    for q in q_list:
        all_data['q_' + str(q)] = all_data.loc[
            (all_data['group_size'] * q).astype(int) == all_data['group_count'], 'flux']
    quantiles_s = all_data.groupby(['object_id', 'passband'])[['q_' + str(q) for q in q_list]].max().unstack(-1)
    quantiles_s.columns = [str(x) + '_' + str(y) + '_quantile_s' for x in quantiles_s.columns.levels[0]
                          for y in quantiles_s.columns.levels[1]]

    extreme_max_s = most_extreme(all_data, 1, positive=True, suffix='_s', include_max=False, include_dur=False,
                                 include_interval=True)
    extreme_min_s = most_extreme(all_data, 1, positive=False, suffix='_min_s', include_max=False, include_dur=False)

    new_data = pd.concat([band_aggs, quantiles, band_aggs_s, max_detected, time_between_detections[['det_period']],
                          time_between_detections_pb, extreme_max, extreme_min, extreme_max_s, extreme_min_s,
                          time_between_highs[['det_period_high']], quantiles_s, detection_time_dist,
                          detection_time_dist_all, det_aggs], axis=1)
    return new_data


# get the metadata
test_meta = pd.read_csv(os.path.join('data', 'test_set_metadata.csv'))
all_meta = pd.concat([train_meta, test_meta], axis=0, ignore_index=True, sort=True).reset_index()
all_meta.drop('index', axis=1, inplace=True)
n_chunks = 100

# calculate features
new_data_exact = calc_aggs(train.copy(), True)
new_data_approx = calc_aggs(train.copy(), False)
train_meta_exact = pd.merge(train_meta, new_data_exact, 'left', left_on='object_id', right_index=True)
train_meta_approx = pd.merge(train_meta, new_data_approx, 'left', left_on='object_id', right_index=True)

# process training set (not actually used, just to get right shape of dataframe)
new_data_arr = []
new_data_arr.append(calc_aggs(train.copy(), True))
# process test set
for i in range(n_chunks):
    df = pd.read_hdf(os.path.join('data', 'split_{}'.format(n_chunks), 'chunk_{}.hdf5'.format(i)), key='file0')
    df.drop('index', axis=1, inplace=True)
    print('Read chunk {}'.format(i))
    new_data_arr.append(calc_aggs(df.copy(), True))
    print('Calculated features for chunk {}'.format(i))
del df
gc.collect()
new_data = pd.concat(new_data_arr, axis=0, sort=True)

# merge
all_meta = pd.merge(all_meta, new_data, 'left', left_on='object_id', right_index=True)

# write output
dir_name = 'features'
if not os.path.exists(os.path.join('data', dir_name)):
    os.mkdir(os.path.join('data', dir_name))
all_meta.to_hdf(os.path.join('data', dir_name, 'all_data.hdf5'), key='file0')
train_meta_exact.to_hdf(os.path.join('data', dir_name, 'train_meta_exact.hdf5'), key='file0')
train_meta_approx.to_hdf(os.path.join('data', dir_name, 'train_meta_approx.hdf5'), key='file0')
