# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gc
import clothes_process as cp
from datetime import datetime, timedelta

path = 'C:/Users/10167227/Desktop/wallet'
#path = '\\172.20.1.43\ServiceHeadquarters\改善改革\需要予測\16春財布消化モデル\春財布\APDATA'
today_date0 = datetime.now()
today_date = "{}-{:02}-{:02}".format(today_date0.year,today_date0.month,today_date0.day)

#pre_dates = [201950,201951,201952,202001,202002,202003,202004,202005,202006,202007,202008]
pre_dates = [202008]
parts = ['data_details_1','data_details_2','data_details_3']
#parts = ['data_details_3']
for part in parts:
    for pre_date in pre_dates:
        call_model = cp.MakeFunction(pre_date,path)
        
        call_model.get_clothes_data(part=part)  #获取相关数据
        call_model.create_train_frame()   #获取训练数据
        call_model.create_future_frame()  #创建1周测试数据并连接原数据
        call_model.add_baika()   #添加卖价数据
        call_model.add_holiday_event()  #读取节日和公司EVENT数据
        call_model.clean_JAN()     #保留卖JAN
        call_model.add_kyakusu(part=part)   #添加客数
        call_model.add_master()    #添加master
        call_model.change_as_sale_sum()    #同类商品求和
        call_model.add_precedent_statistics()   #添加n周前的卖数
        call_model.separate_data()   #分离需要计算统计量的数据
        call_model.pct_change()  #计算变化率
        call_model.add_STATISTICS()  #添加统计量
#        call_model.extract_statistics()  #添加移动窗口的统计量
        call_model.prepare_data()   #统计量合并到原数据
#        call_model.lgb_trian()
#        call_model.lgb_predict(part=part)
        call_model.train_predict(part=part)  #数据预测    
        call_model.write_data(part=part)  #写数据
#        call_model.data_accuracy()




