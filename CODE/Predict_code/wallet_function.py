# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn import preprocessing
from logging import getLogger, FileHandler, StreamHandler, Formatter, DEBUG, INFO
from xgboost import plot_importance
import os


def read_csv(file_path):
    os.chdir(file_path)    
    file_chdir = os.getcwd()
    
    sale_list = []
    for root, dirs, files in os.walk(file_chdir):
        for file in files:
            if os.path.splitext(file)[1] == '.CSV':
                sale_list.append(file)
    sale_data = pd.DataFrame()
    for csv in sale_list:
        sale_data = sale_data.append(pd.read_csv(file_chdir + '\\' + csv,header = -1,sep=None,engine='python',encoding='cp932',usecols=list(range(0,35))))
    return sale_data



def diff_POS(df, group_by, on, ndiff):
    df = df.reset_index(drop=True).sort_values(group_by + ['yearweek'])
    groups = df[[on]+group_by+['yearweek']].groupby(group_by, sort=False)
    
    values = []
    for _, group in tqdm(groups):
        diff_pos = group[on].pct_change(ndiff)
        values.extend(diff_pos.fillna(method='bfill').fillna(method="ffill").values)
        
    assert len(df)==len(values), f'{df.shape} vs {len(values)}'
    suffix = '_&_'.join(group_by)
    df.loc[:, f'{on}_{ndiff}_pct_change_by_{suffix}'] = values
    
    df[f'{on}_{ndiff}_pct_change_by_{suffix}'] = df[f'{on}_{ndiff}_pct_change_by_{suffix}'].astype(str)
    df[f'{on}_{ndiff}_pct_change_by_{suffix}'] = df[f'{on}_{ndiff}_pct_change_by_{suffix}'].replace({'inf':0,'-inf':0})
    df[f'{on}_{ndiff}_pct_change_by_{suffix}'] = df[f'{on}_{ndiff}_pct_change_by_{suffix}'].astype(float)
    return df



def ALL_STATISTICS_year_team(df,sales_cnt,on):

    tmp = df.groupby([f'{on}','JAN'], as_index=False).agg({
                sales_cnt: ['mean', 'median', 'std', 'max', 'min', 'skew']
            })
    tmp.columns = [f'{on}','JAN',
                   sales_cnt+'_'+f'{on}_平均', sales_cnt+'_'+f'{on}_中央値', 
                   sales_cnt+'_'+f'{on}_標準偏差', sales_cnt+'_'+f'{on}_最大值', sales_cnt+'_'+f'{on}_最小值',
                   sales_cnt+'_'+f'{on}_偏度']
    
    df = pd.merge(df.reset_index(drop=True), tmp, how='left', 
                  on=[f'{on}','JAN'])
    
    df.sort_values([f'{on}','JAN'], inplace=True)
#    df.set_index('date', inplace=True)
    return df


#def extract_precedent_statistics(df, on, group_by, day_window):
#    df = df.reset_index(drop=True).sort_values(group_by + ['yearweek'])
#    groups = df[[on]+group_by+['yearweek']].groupby(group_by, sort=False)
#    
#    stats = {
#        'mean': [],
#        'median': [],
#        'std': [],
#        'skew': []
#    }
#    
#    exp_alphas = [0.5]
#    stats.update({'exp_{}_mean'.format(alpha): [] for alpha in exp_alphas})
#    
#    for _, group in tqdm(groups):
#        roll = group[on].rolling(window=day_window, min_periods=1)
#        
#        mean_value = roll.mean().fillna(method='ffill')
#        median_value = roll.median().fillna(method='ffill')
#        std_value = roll.std().fillna(method='ffill')
#        
##        count_value = roll.count().fillna(method='ffill')
#        skew_value = roll.skew().fillna(method='ffill')
#        
#        stats['mean'].extend(mean_value.fillna(mean_value.mean()).values)
#        stats['median'].extend(median_value.fillna(median_value.mean()).values)
#        stats['std'].extend(std_value.fillna(std_value.mean()).values)
#        stats['skew'].extend(skew_value.fillna(skew_value.mean()).values)
#        
#        for alpha in exp_alphas:
#            exp = group[on].ewm(alpha=alpha, adjust=False)
#            ewm_value = exp.mean()
#            stats['exp_{}_mean'.format(alpha)].extend(ewm_value.fillna(ewm_value))
#    
#    suffix = '_&_'.join(group_by)
#    
#    for stat_name, values in tqdm(stats.items()):
#        df['{}_{}_by_{}periods_{}'.format(on, stat_name, day_window, suffix)] = values
#        
#    return df
def extract_precedent_statistics(df, on, group_by, day_window, exp_alphas):
    df = df.reset_index(drop=True).sort_values(group_by + ['yearweek'])
    groups = df[[on]+group_by+['yearweek']].groupby(group_by, sort=False)
    
    stats = {
        'mean': [],
        'median': [],
        'std': [],
        'skew': []
#        'sum': []
#        'min': []
#        'max': []
    }
    
#    exp_alphas = [0.6]
#    stats.update({'exp_{}_mean'.format(alpha): [] for alpha in exp_alphas})
#    stats.update({'exp_{}_var'.format(alpha): [] for alpha in exp_alphas})
    
    for _, group in tqdm(groups):
        roll = group[on].rolling(window=day_window, min_periods=1)
        
        mean_value = roll.mean().fillna(method='ffill')
        median_value = roll.median().fillna(method='ffill')
        std_value = roll.std().fillna(method='ffill')
#        sum_value = roll.sum().fillna(method='ffill')
#        min_value= roll.min().fillna(method='ffill')
#        max_value= roll.max().fillna(method='ffill')
        
#        count_value = roll.count().fillna(method='ffill')
        skew_value = roll.skew().fillna(method='ffill')
        
        stats['mean'].extend(mean_value.fillna(mean_value.mean()).values)
        stats['median'].extend(median_value.fillna(median_value.mean()).values)
        stats['std'].extend(std_value.fillna(std_value.mean()).values)
        stats['skew'].extend(skew_value.fillna(skew_value.mean()).values)
#        stats['sum'].extend(sum_value.fillna(sum_value.mean()).values)
#        stats['min'].extend(min_value.fillna(min_value.mean()).values)
#        stats['max'].extend(max_value.fillna(max_value.mean()).values)
        
#        for alpha in exp_alphas:
#            exp = group[on].ewm(alpha=alpha, adjust=False)
#            ewm_value = exp.mean()
#            stats['exp_{}_mean'.format(alpha)].extend(ewm_value.fillna(ewm_value))
#            ewm_var_value = exp.var()
#            stats['exp_{}_var'.format(alpha)].extend(ewm_var_value.fillna(ewm_var_value))
    
    suffix = '_&_'.join(group_by)
    
    for stat_name, values in tqdm(stats.items()):
        df['{}_{}_by_{}periods_{}'.format(on, stat_name, day_window, suffix)] = values
        
    return df


def target_encoding_to_df(df_train, df_test, label='JAN', target='sales_cnt'):
    trn, sub = target_encode(
                    df_train[label],
                    df_test[label],
                    target=df_train[target],
                    min_samples_leaf=100,
                    smoothing=10,
                    noise_level=0.01
               )
    df_train.loc[:, label] = trn.values
    df_test.loc[:, label] = sub.values
    
    return df_train, df_test


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


#均方根平方对数误差
def rmsle(pre,act):
    return (((np.log(pre + 1) - np.log(act + 1)) ** 2).mean()) ** 0.5



def get_weather():
    connection = psycopg2.connect(
        "host=10.2.5.50 port=5432 dbname=NumericalAnalysis "  "user=analyst password=TRanalyst")
    cur = connection.cursor()

    sql = "SELECT day,store,kionmax,kousuryosum,fusokumaxspeed            FROM webapi_weather as A,webapi_weather_store as B              WHERE A.prec_no=B.prec_no                    and A.block_no=B.block_no                    and A.day>='2015-01-01'                    and B.store in (%d)             union             SELECT day,store,kionmax,kousuryosum,fusokumaxspeed              FROM webapi_weather_future             " % (
    178, 178)
    cur.execute(sql)
    result = []
    for irow in cur:
        result.append([irow[0], irow[1], irow[2], irow[3], irow[4]])
    df_tenki = pd.DataFrame(result, columns=['date', 'store_cd', 'temp_max', 'rain_sum', 'speed_max'])
    df_tenki = df_tenki.sort_values(["date"])
    return df_tenki
