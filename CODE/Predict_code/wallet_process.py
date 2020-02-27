# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from xgboost import plot_importance
from matplotlib import pyplot
import os
from datetime import datetime
import time
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
import random
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import clothes_function as cf
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import lightgbm as lgb


class MakeFunction(object):

    def __init__(self,pre_date,path):
        
        self.pre_date = pre_date
        self.path = path


    def get_clothes_data(self,part):
        
        sale_data = cf.read_csv(file_path=self.path + f'/{part}/')   
        sale_data['店CD'] = 99999  
        sale_data_bk = sale_data.copy()
        
#        sale_data['売上数量'] = sale_data['売上数量']-sale_data['POS値下数量']
        sale_data['売上数量'] = sale_data['売上数量']
        sale_data.rename(columns={"年週":"yearweek","地域CD":"areacd","売上数量":"sales_cnt","店CD":"store_cd","品種名":"kind_name",
                          "品目名":"class_name"},inplace=True)
    

        sale_data = sale_data.groupby(['yearweek','JAN','store_cd'])['sales_cnt'].sum().reset_index()
    
        #年週限定
        sale_data=sale_data.loc[(sale_data['yearweek']>= 201527)&(sale_data['yearweek']<= self.pre_date)].reset_index(drop=True)
        sale_data = sale_data[['yearweek','JAN','store_cd','sales_cnt']]        
        
        self.sale_data = sale_data
        self.sale_data_bk = sale_data_bk

            
        
    def create_train_frame(self):
        sale_data = self.sale_data.copy()   
        
        
        train_data = sale_data[(sale_data['yearweek']>=201527)&(sale_data['yearweek']<self.pre_date)&(sale_data['sales_cnt']>=0)][['yearweek',
                               'JAN','store_cd','sales_cnt']]

        self.train_data = train_data
        
            
    def create_future_frame(self):
        train_data0 = self.train_data.copy()
                
        test = pd.DataFrame(columns=['yearweek','JAN','store_cd','sales_cnt'])
        test_JAN = [4562342690029,4562342690036,4562342690043,4562342690050,4562342690067,4562342690074,4562342690081]
        test['JAN'] = test_JAN
        test['yearweek'] = self.pre_date
        test['store_cd'] = 99999
        
        items = ['sales_cnt']
        for item in items:
            test[item] = 0
                

        alldata = pd.concat([train_data0,test],axis=0).reset_index(drop=True)
    
        self.alldata = alldata
        self.test_JAN = test_JAN

    
    def add_baika(self):
        alldata = self.alldata.copy()
        
        baika_info = pd.read_excel(self.path + "/others/jan_ten_baika.xlsx", encoding="cp932")
        baika_info = baika_info.groupby(['JAN'])[ 'baika'].mean().reset_index()        
        
        alldata = pd.merge(alldata,baika_info,how="left",on=["JAN"])
        bins = [0,1000,6000]
        alldata['baikarng'] = pd.cut(alldata['baika'], bins, labels=[1,2])
        
        
        self.alldata = alldata
        
        
    def add_holiday_event(self):        
        alldata0 = self.alldata.copy()
        
        dateind_inf = pd.read_csv(self.path + "/others/date_ind.csv",parse_dates=[0])
        holiday_inf = pd.read_csv(self.path + "/others/" + "WEBAPI_PUBLIC_HOLIDAY.csv",parse_dates=[0])
        event_inf = pd.read_csv(self.path + "/others/" + "WEBAPI_TRIAL_EVENT.csv",parse_dates=[0])
        dateind_inf["date"] = pd.to_datetime(dateind_inf["date"])
        holiday_inf["date"] = pd.to_datetime(holiday_inf["date"])
        event_inf["date"] = pd.to_datetime(event_inf["date"])
        
        week_list = pd.merge(dateind_inf,holiday_inf,how="left",on=["date"])
        week_list = pd.merge(week_list,event_inf,how="left",on=["date"])
        
        week_list["holiday"] = week_list["holiday"].fillna(0)
        week_list.loc[week_list["holiday"] !=0, "holiday"]=1
        
        week_list["event"] = week_list["event"].fillna(0)
        week_list.loc[week_list["event"] !=0, "event"]=1
        
        week_list = week_list.groupby(['yearweek','weeknum'])['holiday','event'].sum().reset_index()
        week_list['wyear']= (week_list['yearweek']/100).astype(int)
        
        week_list.loc[week_list["weeknum"] <33, "wyear"]= week_list['wyear'] -1
        
        alldata = pd.merge(alldata0,week_list,how="left",on=["yearweek"])
        
        self.alldata = alldata
        self.dateind_inf = dateind_inf
        
        
        
    def clean_JAN(self):
        alldata = self.alldata.copy()
        #筛选JAN
        sale_JAN_2020 = pd.read_excel(self.path + "/others/2020_sale_JAN.xlsx")
        alldata2019 = alldata[alldata['wyear']==2019]
        alldata2019 = alldata2019[alldata2019['JAN'].isin(sale_JAN_2020[2020].to_list())]
        
        alldata = alldata[alldata['wyear']!=2019]
        
        alldata = pd.concat([alldata,alldata2019],axis=0).reset_index(drop=True)

        self.alldata = alldata
        
        
        
    def add_kyakusu(self,part):
        alldata0 = self.alldata.copy()
        n = part[-1]
        
        kyakusu_inf = pd.read_csv(self.path + "/others/" + f"KYAKUSU_{n}.CSV")
        kyakusu_inf.rename(columns={'店CD':'store_cd','年週':'yearweek','店舗名':'store_name','客数(POS客数)':'kyakusu'},inplace=True)
        kyakusu_inf = kyakusu_inf.groupby(['yearweek'])['kyakusu'].sum().reset_index()
        
        alldata0 = pd.merge(alldata0,kyakusu_inf,how="left",on=['yearweek'])
        alldata0['kyakusu'] = alldata0['kyakusu'].fillna(alldata0['kyakusu'].mean())
        
        self.alldata = alldata0
        
        
        
    def add_master(self):
        
        master = pd.read_csv(self.path + "/others/" + "master.csv", encoding="cp932")
        master = master[['JAN','サブカテゴリー(品種)名','セグメント(品目)名','サブセグメント名','カラーCD']]
        alldata0 = pd.merge(self.alldata,master,on=['JAN'],how='inner')
        alldata0['sales_cnt'] = alldata0['sales_cnt'].astype(int)
        
        self.alldata = alldata0
    
    
    
    def change_as_sale_sum(self):
        alldata0 = self.alldata.copy()
        
        newdata = alldata0.copy()
        newdata = newdata.groupby(['yearweek','store_cd','baikarng','サブカテゴリー(品種)名','weeknum','holiday','event',
                                       'wyear','kyakusu'])['sales_cnt'].sum().reset_index()
           
        newdata.loc[(newdata['baikarng']==1)&(newdata['サブカテゴリー(品種)名']=='紳士小物'),'JAN']='4562342690029'
        newdata.loc[(newdata['baikarng']==1)&(newdata['サブカテゴリー(品種)名']=='紳士長財布'),'JAN']='4562342690036'
        newdata.loc[(newdata['baikarng']==1)&(newdata['サブカテゴリー(品種)名']=='紳士折財布'),'JAN']='4562342690043'
        newdata.loc[(newdata['baikarng']==2)&(newdata['サブカテゴリー(品種)名']=='紳士長財布'),'JAN']='4562342690050'
        newdata.loc[(newdata['baikarng']==2)&(newdata['サブカテゴリー(品種)名']=='紳士折財布'),'JAN']='4562342690067'
        newdata.loc[(newdata['baikarng']==1)&(newdata['サブカテゴリー(品種)名']=='婦人長財布'),'JAN']='4562342690074'
        newdata.loc[(newdata['baikarng']==1)&(newdata['サブカテゴリー(品種)名']=='婦人折財布'),'JAN']='4562342690081'
        
        test_JAN_1 = ['4562342690029','4562342690036','4562342690043','4562342690050','4562342690067','4562342690074','4562342690081']
        newdata = newdata[newdata['JAN'].isin(test_JAN_1)]
        newdata['JAN'] = newdata['JAN'].astype('int64')
        
        alldata0 = newdata.copy()   
        
        self.alldata = alldata0
    
    
    
    def add_precedent_statistics(self):
        alldata0 = self.alldata.copy()                
                        
        date_list = self.dateind_inf.copy()
        
        date_list['7days_ago'] = date_list[ 'date'] - pd.Timedelta(days=7)
        date_list['14days_ago'] = date_list[ 'date'] - pd.Timedelta(days=14)
        date_list['21days_ago'] = date_list[ 'date'] - pd.Timedelta(days=21)
        date_list['28days_ago'] = date_list[ 'date'] - pd.Timedelta(days=28)
        
        weekdt1 = pd.merge(date_list[["date",'yearweek']],date_list[["7days_ago",'yearweek']],how="inner",left_on="date",right_on = '7days_ago')
        weekdt1 = weekdt1[['yearweek_x','yearweek_y']].drop_duplicates(keep='first').reset_index(drop =True)
        weekdt1.columns = ['yearweek','yearweek1']
        
        weekdt2 = pd.merge(date_list[["date",'yearweek']],date_list[["14days_ago",'yearweek']],how="inner",left_on="date",right_on = '14days_ago')
        weekdt2 = weekdt2[['yearweek_x','yearweek_y']].drop_duplicates(keep='first').reset_index(drop =True)
        weekdt2.columns = ['yearweek','yearweek2']
        
        weekdt3 = pd.merge(date_list[["date",'yearweek']],date_list[["21days_ago",'yearweek']],how="inner",left_on="date",right_on = '21days_ago')
        weekdt3 = weekdt3[['yearweek_x','yearweek_y']].drop_duplicates(keep='first').reset_index(drop =True)
        weekdt3.columns = ['yearweek','yearweek3']
        
        weekdt4 = pd.merge(date_list[["date",'yearweek']],date_list[["28days_ago",'yearweek']],how="inner",left_on="date",right_on = '28days_ago')
        weekdt4 = weekdt4[['yearweek_x','yearweek_y']].drop_duplicates(keep='first').reset_index(drop =True)
        weekdt4.columns = ['yearweek','yearweek4']
        
        weekdt = pd.merge(weekdt1,weekdt2,how="inner",on="yearweek")
        weekdt = pd.merge(weekdt,weekdt3,how="inner",on="yearweek")
        weekdt = pd.merge(weekdt,weekdt4,how="inner",on="yearweek")
        
        bf_data = alldata0[['yearweek','store_cd','JAN','sales_cnt']]
        bf_data = bf_data.merge(weekdt,how='left',on='yearweek')
        bf_data = bf_data.drop_duplicates(subset=['yearweek','JAN','store_cd'], keep='first')
        
        # 先週の売数
        bf_1w_data = bf_data[['yearweek1','store_cd','JAN','sales_cnt']]
        bf_1w_data.rename(columns={"yearweek1":"yearweek","sales_cnt":"sales_cnt_1bf"},inplace=True)
        
        # 2週前の売数
        bf_2w_data = bf_data[['yearweek2','store_cd','JAN','sales_cnt']]
        bf_2w_data.rename(columns={"yearweek2":"yearweek","sales_cnt":"sales_cnt_2bf"},inplace=True)
        
        # 3週前の売数
        bf_3w_data = bf_data[['yearweek3','store_cd','JAN','sales_cnt']]
        bf_3w_data.rename(columns={"yearweek3":"yearweek","sales_cnt":"sales_cnt_3bf"},inplace=True)
        
        # 4週前の売数
        bf_4w_data = bf_data[['yearweek4','store_cd','JAN','sales_cnt']]
        bf_4w_data.rename(columns={"yearweek4":"yearweek","sales_cnt":"sales_cnt_4bf"},inplace=True)
        
        newdata = alldata0.copy()
        for bf_data in [bf_1w_data,bf_2w_data,bf_3w_data,bf_4w_data]:
            newdata = pd.merge(newdata,bf_data,how="left",on=["JAN","store_cd",'yearweek'])
            
        for sales_cnt in ['sales_cnt_1bf','sales_cnt_2bf','sales_cnt_3bf','sales_cnt_4bf']:
            newdata[f'{sales_cnt}'] = newdata[f'{sales_cnt}'].fillna(0)
                                    
        self.newdata = newdata
        
        
    def separate_data(self):
        newdata = self.newdata.copy()        
        
        newdata = newdata[(newdata['yearweek']<=self.pre_date)].reset_index(drop=True)       
        newdata['sales_cnt'] = newdata['sales_cnt'].astype(int)
        bins = [2014,2015,2016,2017,2018,2019]
        newdata['year_team'] = pd.cut(newdata['wyear'], bins, labels=[2015,2016,2017,2018,2019])
        
                
#        newdata = newdata[newdata['wyear']!=2016]
        newdata = newdata[~newdata['weeknum'].isin(list(range(33,47)))]
              
        usecols = ['JAN','yearweek','sales_cnt_1bf','year_team','weeknum','store_cd']
        df_use = newdata[usecols].copy()
        
        newdata['original_index'] = range(len(newdata))
        df_use['original_index'] = newdata['original_index'].values
                
        
        self.newdata = newdata
        self.df_use = df_use
        self.usecols = usecols
        
        
    def pct_change(self):
        
        df_use = self.df_use.copy()
        
        for day in [1, 2]:
            for group in ['year_team','weeknum']:
                df_use = cf.diff_POS(df=df_use, group_by=[f'{group}','JAN'], on='sales_cnt_1bf', ndiff=day)
            
        self.df_use = df_use

        
        
        
    def add_STATISTICS(self):
        df_use = self.df_use.copy()
        
        for sales_cnt in ['sales_cnt_1bf']:
            for group in ['year_team','weeknum']:
                df_use = cf.ALL_STATISTICS_year_team(df_use, sales_cnt, on=group)
                df_use[f'sales_cnt_1bf_{group}_標準偏差'] = df_use[f'sales_cnt_1bf_{group}_標準偏差'].fillna(df_use[f'sales_cnt_1bf_{group}_標準偏差'].mean())

        self.df_use = df_use
        
        
    def extract_statistics(self):
        df_use = self.df_use.copy()
        
        df_use = cf.extract_precedent_statistics(
                    df=df_use,
                    on='sales_cnt_1bf',
                    group_by=['year_team','JAN'],
                    day_window=3,
                    exp_alphas=[0.25,0.75])
        df_use = cf.extract_precedent_statistics(
                    df=df_use,
                    on='sales_cnt_1bf',
                    group_by=['year_team','JAN'],
                    day_window=5,
                    exp_alphas=[0.25,0.75])
        df_use = cf.extract_precedent_statistics(
                    df=df_use,
                    on='sales_cnt_1bf',
                    group_by=['weeknum',"JAN"],
                    day_window=3,
                    exp_alphas=[0.25,0.75])

        self.df_use = df_use
        
        
    def prepare_data(self):        
        newdata = self.newdata
        
        concat_cols = [col for col in self.df_use.columns if col not in self.usecols]

        newdata = newdata.reset_index(drop=True).merge(self.df_use[concat_cols], on=['original_index'], how='left')
        newdata.drop(labels='original_index', axis=1, inplace=True)
        
        newdata = newdata.fillna(method="bfill").fillna(method="ffill")
        
        #特征拼接
        newdata['JAN_weeknum'] = newdata['JAN'].astype(str) + '_' + newdata['weeknum'].astype(str)
        newdata['サブカテゴリー(品種)名_weeknum'] = newdata['サブカテゴリー(品種)名'].astype(str) + '_' + newdata['weeknum'].astype(str)
        
        bins = [0,100,200,300,400,600,1200]
        newdata['sales_distinguish'] = pd.cut(newdata['sales_cnt_1bf'], bins, labels=['卖数小于100','卖数小于200',
               '卖数小于300','卖数小于400','卖数小于600','卖数小于1200'])
        newdata[['JAN','weeknum','sales_distinguish']] = newdata[['JAN','weeknum','sales_distinguish']].apply(lambda x:x.astype(str))
        newdata['年周_卖数'] = newdata['weeknum']+'_'+newdata['sales_distinguish']
                        
        newdata['JAN'] = newdata['JAN'].astype('int64')
        
        self.newdata = newdata


    def lgb_trian(self):
        newdata = self.newdata.copy()
        
        start_date = list(newdata.yearweek.drop_duplicates())[-3]
        end_date = list(newdata.yearweek.drop_duplicates())[-2]
        df  = newdata.drop(['sales_cnt_1bf','sales_cnt_2bf','sales_cnt_3bf','sales_cnt_4bf'],axis=1)
        df_train = df[df['yearweek']<=start_date]
        df_test = df[df['yearweek']==end_date]
        
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='サブカテゴリー(品種)名')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='JAN_weeknum')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='サブカテゴリー(品種)名_weeknum')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='sales_distinguish')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='年周_卖数')
        
        
        X_train_df = df_train.drop(labels=['sales_cnt'], axis=1)
        y_train_df = df_train['sales_cnt']
        X_test_df = df_test.drop(labels=['sales_cnt'], axis=1)
        y_test_df = df_test['sales_cnt']
        
        X_train, y_train = X_train_df.values, y_train_df.values
        X_test, y_test = X_test_df.values, y_test_df.values
        dtrain = lgb.Dataset(X_train,y_train)  
        dtest = lgb.Dataset(X_test,y_test,reference=dtrain)
        
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'regression', # 目标函数
            'metric': {'l1','l2'},  # 评估函数
            'num_leaves': 31,   # 叶子节点数(<=2^(max_depth))
            'learning_rate': 0.05,  # 学习速率
            'max_bin': 255,   #feature将存入的bin的最大数量
            'min_data_in_leaf': 20,  #一个叶子上数据的最小数量.
            'feature_fraction': 0.9, # 建树的特征选择比例
            'bagging_fraction': 0.8, # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'seed': 2100,
            'lambda_l1': 0.8,
            'lambda_l2': 5,
            'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        
        valid = [dtrain,dtest]
        gbm = lgb.train(params,dtrain,num_boost_round=20000,valid_sets=valid,early_stopping_rounds=100,verbose_eval=True) 
        preds = gbm.predict(X_test)        
        df_test['sales_pre'] = preds
        df_test_valid = df_test[['yearweek','JAN','sales_cnt','sales_pre']]
        best_iterat = gbm.best_iteration
        
        self.best_iterat = best_iterat
        self.df_test_valid = df_test_valid
        self.params = params
        
    def lgb_predict(self,part):
        newdata = self.newdata.copy()
        best_iterat = self.best_iterat
        params = self.params
        

        df  = newdata.drop(['sales_cnt_1bf','sales_cnt_2bf','sales_cnt_3bf','sales_cnt_4bf'],axis=1)
        df_train = df[df['yearweek']<self.pre_date]
        df_test = df[df['yearweek']==self.pre_date]
        
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='サブカテゴリー(品種)名')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='JAN_weeknum')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='サブカテゴリー(品種)名_weeknum')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='sales_distinguish')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='年周_卖数')
        
        
        X_train_df = df_train.drop(labels=['sales_cnt'], axis=1)
        y_train_df = df_train['sales_cnt']
        X_test_df = df_test.drop(labels=['sales_cnt'], axis=1)
#        y_test_df = df_test['sales_cnt']
        
        X_train, y_train = X_train_df.values, y_train_df.values
#        X_test, y_test = X_test_df.values, y_test_df.values
        X_test = X_test_df.values
        dtrain = lgb.Dataset(X_train,y_train)
        
        model = lgb.train(params,dtrain,best_iterat)
        preds = model.predict(X_test, num_iteration=best_iterat)
        df_test['sales_pre'] = preds
        df_test = df_test[['yearweek','JAN','store_cd','sales_pre']]

        df_test.to_csv(self.path + f'/{part}_pre/' + f'{self.pre_date}' +'.CSV', index=False)
        
        importances0 = model.feature_importance()
        importances = pd.DataFrame(importances0, columns=["importance"])
        feature_data = pd.DataFrame(list(X_train_df.columns), columns=["feature"])
        importance = pd.concat([feature_data, importances], axis=1)
        importance = importance.sort_values(["importance"], ascending=True)

        self.importance = importance
        self.test = df_test
        
                
        
    def train_predict(self,part):
        newdata = self.newdata.copy()
        
        df_train = newdata[newdata['yearweek']<self.pre_date].reset_index(drop=True)
    
        df_test = newdata[newdata['yearweek']==self.pre_date].reset_index(drop=True)
        df_test = df_test.sort_values(['JAN','store_cd'], ascending=True).reset_index(drop=True)
        df_test_original = df_test.copy()

        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='サブカテゴリー(品種)名')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='JAN_weeknum')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='サブカテゴリー(品種)名_weeknum')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='sales_distinguish')
        df_train, df_test = cf.target_encoding_to_df(df_train, df_test, label='年周_卖数')        
        
        X_train = df_train.drop(['sales_cnt','sales_cnt_1bf','sales_cnt_2bf','sales_cnt_3bf','sales_cnt_4bf'],axis=1)
        Y_train = df_train['sales_cnt']
        
        X_test = df_test.drop(['sales_cnt','sales_cnt_1bf','sales_cnt_2bf','sales_cnt_3bf','sales_cnt_4bf'],axis=1)
        
        
        params = {
        #          'random_state': 100,
                  'n_jobs': 4,
                  'max_depth': 14,
        #           'min_sample_split': 1,
        #           'max_features': 36,
                  'n_estimators': 100}
        try:
            rf = RandomForestRegressor(**params)
            rf.fit(X_train,Y_train)
        except ValueError as a:
            print('发生了异常：',a)
            X_train = X_train.fillna(method='ffill').fillna(method='bfill')
            Y_train = Y_train.fillna(method='ffill').fillna(method='bfill')
            rf = RandomForestRegressor(**params)
            rf.fit(X_train,Y_train)
            print('填充数据进行异常修改')
        finally:
            print('模型训练完毕，开始预测')
        
        Y_test = rf.predict(X_test)

        df_test_original['sales_pre'] = Y_test    
        test = df_test_original[['yearweek','JAN','store_cd','sales_pre']]
        
                            
            
        test.to_csv(self.path + f'/{part}_pre/' + f'{self.pre_date}' +'.CSV', index=False)
        
        importances0 = rf.feature_importances_
        importances = pd.DataFrame(importances0, columns=["importance"])
        feature_data = pd.DataFrame(list(X_train.columns), columns=["feature"])
        importance = pd.concat([feature_data, importances], axis=1)
        importance = importance.sort_values(["importance"], ascending=True)
        importance["importance"] = (importance["importance"] * 1000).astype(int)

        self.importance = importance
        self.test = test
        
    def importance_plt(self):
        importance = self.importance
        
        plt.figure(figsize=[10,10])
        plt.barh(range(0,29),importance['importance'], align='center')
        plt.yticks(range(0,29),importance['feature'])
        plt.show()
        
        
    def write_data(self,part):
        sale_data_bk = self.sale_data_bk

        test_bk = self.test[['JAN','store_cd','sales_pre']]
        test_bk.rename(columns={'sales_pre':'売上数量','store_cd':'店CD'},inplace=True)
        
        data = pd.merge(test_bk[['JAN','店CD']],sale_data_bk.drop(['年週','売上数量'],axis=1),how='inner',on=['JAN','店CD'])
        data = data.drop_duplicates(subset=['JAN','店CD'],keep='first').reset_index(drop=True)
        data['年週'] = self.pre_date
        data['POS値下数量'] = 0
        data = pd.merge(data,test_bk,how='inner',on=['JAN','店CD'])


        data = data[['年週', '地域CD', '地域名', 'タイプ分類CD', 'タイプ分類名', '店CD', '店舗名', '事業部CD', '事業部名',
                     'ディビジョンCD', 'ディビジョン名', 'ラインCD', 'ライン名', '部門CD', '部門名', 'ミニ部門CD',
                     'ミニ部門名', '品種CD', '品種名', '品目CD', '品目名', 'JAN', '子JAN', 'バンドル入数', '商品名',
                     '規格名称', 'ブランド名', '季節区分CD', '季節区分名', '売上数量', '売上税込金額', 'POS値下数量',
                     'POS値下税込金額', '廃棄数', '廃棄売価金額(税込)']]
        
        
        data.to_csv(self.path + f'/{part}/' + f'{self.pre_date}' +'.CSV', index=False, encoding='cp932')
       
    
    def data_accuracy(self):
        df_test_valid = self.df_test_valid.copy()
                              
        score = r2_score(df_test_valid['sales_cnt'],df_test_valid['sales_pre'])
        rmsle = cf.rmsle(df_test_valid['sales_pre'],df_test_valid['sales_cnt'])
        rmse = (mean_squared_error(df_test_valid['sales_pre'],df_test_valid['sales_cnt']))**0.5
        
        print('R2:{}'.format(score))
        print('RMSLE:{}'.format(rmsle))
        print('RMSE:{}'.format(rmse))
#        pre_act.to_csv(f'C:/Users/10167227/Desktop/test/{self.pre_date}.csv',index=False)


