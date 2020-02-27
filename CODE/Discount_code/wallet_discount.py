#!/usr/bin/env python
# coding: utf-8

# In[1]:


###################################################################################################
# -------------------前期准备---------------------------------------------------------------
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
from decimal import *
import datetime
from sklearn.ensemble import RandomForestRegressor
pd.set_option('display.max_columns', 100)

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
        sale_data = sale_data.append(pd.read_csv(file_chdir + '\\' + csv,header = -1,sep=None,engine='python',encoding='cp932'))
    return sale_data

# -------------------时间预设置------------------------------------------------------------

#当前年份
year = datetime.datetime.now().isocalendar()[0]

# 执行代码周
do_time_now = datetime.datetime.now().isocalendar()[1]
if do_time_now >= 1 & do_time_now <= 8:          #如果周在47-52之间，执行周就是当前周，如果在1-8周之间，执行周就是53-60周
    do_time = do_time_now + 52
else:
    do_time = do_time_now

# 预测开始周
pred_start_time = 52 + datetime.datetime.now().isocalendar()[1]

#可以开始打折时间
if do_time > 54 :
    nbk_able_time = do_time
else :
    nbk_able_time = 54


# In[3]:


#########################数据处理 ##########################################################################
# --------------------数据读取，计算，合并-------------------------------------------------------------
#销售数据读取
origin = read_csv('C:/Users/10176726/Desktop/data/data_details_1') #商品表
origin.reset_index(inplace=True)

# 添加holiday和 event
dateind_inf = pd.read_csv(r'C:/Users/10176726/Desktop/data/others/date_ind.csv')
holiday_inf = pd.read_csv(r'C:/Users/10176726/Desktop/data/others/event.csv',encoding='cp932')
event_inf = pd.read_csv(r'C:/Users/10176726/Desktop/data/others/holiday.csv',encoding='cp932')

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

week_list.rename(columns={'yearweek':"年週"},inplace=True)

origin = pd.merge(origin,week_list,how="left",on=["年週"])

#添加客数
kyakusu_inf = pd.read_csv(r'C:/Users/10176726/Desktop/data/others/KYAKUSU_1.csv',encoding ='cp932')

kyakusu_inf = kyakusu_inf.groupby(['年週'])['客数(POS客数)'].sum().reset_index()

origin = pd.merge(origin,kyakusu_inf,how="left",on=['年週'])

origin['客数(POS客数)'] = origin['客数(POS客数)'].fillna(origin['客数(POS客数)'].mean())


# In[4]:


#销售数据时间带处理
origin['year']= (origin['年週']/100).astype(int)
origin['week']= origin['年週'] - origin['year']*100

origin["year_start_47"] = origin["year"].copy()
origin.loc[origin["week"] <= 8, "year_start_47"] = origin["year_start_47"] - 1
origin.drop(origin[(origin.year == 2016) & (origin.week == 53)].index,inplace=True)

origin["week_start_47"] = origin["week"].copy()
origin.loc[(origin['week']<=8) & (origin['year']==2017), 'week_start_47'] = origin["week_start_47"] + 52
origin.loc[(origin['week']<=8) & (origin['year']==2018), 'week_start_47'] = origin["week_start_47"] + 52
origin.loc[(origin['week']<=8) & (origin['year']==2019), 'week_start_47'] = origin["week_start_47"] + 52
origin.loc[(origin['week']<=8) & (origin['year']==2020), 'week_start_47'] = origin["week_start_47"] + 52

#origin = origin.loc[(origin.week_start_47 >= 47)&(origin.week_start_47 <= 60),]


# In[5]:


# 如果取得的数据周数最大值是60，那么time_key是61，否则就是周数的最大值
if int(origin["week_start_47"].max())==60:
    time_key = int(origin["week_start_47"].max())+1 
else:
    time_key = int(origin["week_start_47"].max())


# In[10]:


#打折信息处理
#sale_data1 = origin.iloc[:-((time_key)-pred_start_time+1)*7, :].copy()  #当前周之前的数据，实际数据
#af_data = origin.iloc[-((time_key)-pred_start_time)*7:, :].copy() #当前周之后的数据，待预测数据
sale_data1 = origin.loc[origin['year_start_47']<year-1]
sale_data2 = origin.loc[(origin['year_start_47']==year-1) & (origin['week_start_47']<pred_start_time)]
sale_data3 = origin.loc[(origin['year_start_47']==year-1) & (origin['week_start_47']>=pred_start_time)]


# In[12]:


#数据改造
sale_data1['baika'] =  (sale_data1["売上税込金額"] + sale_data1["POS値下税込金額"])/sale_data1["売上数量"]

bins = [0,1000,6000]
sale_data1['baikarng'] = pd.cut(sale_data1['baika'], bins, labels=[1,2])

sale_data1.loc[(sale_data1['baikarng']==1)&(sale_data1['品種名']=='紳士小物'),'JAN']='4562342690029' 
sale_data1.loc[(sale_data1['baikarng']==1)&(sale_data1['品種名']=='紳士長財布'),'JAN']='4562342690036'
sale_data1.loc[(sale_data1['baikarng']==1)&(sale_data1['品種名']=='紳士折財布'),'JAN']='4562342690043'
sale_data1.loc[(sale_data1['baikarng']==2)&(sale_data1['品種名']=='紳士長財布'),'JAN']='4562342690050'
sale_data1.loc[(sale_data1['baikarng']==2)&(sale_data1['品種名']=='紳士折財布'),'JAN']='4562342690067'
sale_data1.loc[(sale_data1['baikarng']==1)&(sale_data1['品種名']=='婦人長財布'),'JAN']='4562342690074'
sale_data1.loc[(sale_data1['baikarng']==1)&(sale_data1['品種名']=='婦人折財布'),'JAN']='4562342690081'

test_JAN_1 = ['4562342690029','4562342690036','4562342690043','4562342690050','4562342690067','4562342690074','4562342690081']
sale_data1 = sale_data1[sale_data1['JAN'].isin(test_JAN_1)]
sale_data1['JAN'] = sale_data1['JAN'].astype('int64')
del sale_data1['baikarng']

sale_data2 = sale_data2[sale_data2['JAN'].isin(test_JAN_1)]
sale_data2['JAN'] = sale_data2['JAN'].astype('int64')
sale_data2['baika'] =  (sale_data2["売上税込金額"] + sale_data2["POS値下税込金額"])/sale_data2["売上数量"]

sale_data3['baika'] = sale_data3['売上税込金額']

sale_data = sale_data1.append([sale_data2,sale_data3],ignore_index=True)


# In[13]:


# 打折计算
sale_data["nbk_ratio"] = round((sale_data["POS値下税込金額"]/sale_data["POS値下数量"])/sale_data["baika"]*100,0)


# In[14]:


a=sale_data["baika"].values*0.5
b=sale_data["baika"].values*0.3
c=sale_data["POS値下税込金額"].values
d=sale_data["POS値下数量"].values
y = (c - a*d)/(b - a)
x = (b * d - c)/(b - a)

f = lambda x : float(Decimal(str(x)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))

sale_data["n_50perc"] =  np.array(list(map(f, x)))
sale_data["n_30perc"] =  np.array(list(map(f, y)))


# In[15]:


sale_data['n_30perc_copy'] = sale_data['n_30perc'].copy()
sale_data['n_50perc_copy'] = sale_data['n_50perc'].copy()

#sale_data.sort_values('n_50perc_copy',ascending=True).head(2)

def n_50perc(a, b):
	if a  < 0:
		return a+b
	else:
		return b

sale_data['n_50perc'] = sale_data.apply(lambda x: n_50perc(x.n_30perc_copy, x.n_50perc_copy), axis = 1)

sale_data.loc[sale_data['n_30perc_copy']<0,'n_30perc'] = 0

def n_30perc(a, b):
	if  b < 0:
		return a+b
	else:
		return a

sale_data['n_30perc'] = sale_data.apply(lambda x: n_30perc(x.n_30perc_copy, x.n_50perc_copy), axis = 1)

sale_data.loc[sale_data['n_50perc_copy']<0,'n_50perc'] = 0

sale_data = sale_data.groupby(['年週','year','week','year_start_47','week_start_47','JAN','品種名','holiday','event','客数(POS客数)'])['売上数量','n_30perc','n_50perc'].sum().reset_index()


# In[16]:


sale_data["nbk30"] = (sale_data["n_30perc"] > 0 ).astype('int') 
sale_data["nbk50"] = (sale_data["n_50perc"] > 0 ).astype('int')


# In[17]:


df =sale_data.copy()


# In[18]:


df['baika'] = 0
df.loc[df['JAN']==4562342690029,'baika'] = 399
df.loc[df['JAN']==4562342690036,'baika'] = 998
df.loc[df['JAN']==4562342690043,'baika'] = 998
df.loc[df['JAN']==4562342690050,'baika'] = 1990
df.loc[df['JAN']==4562342690067,'baika'] = 1990
df.loc[df['JAN']==4562342690074,'baika'] = 998
df.loc[df['JAN']==4562342690081,'baika'] = 998

df.loc[df['JAN']==4562342690029,'label'] = 1
df.loc[df['JAN']==4562342690036,'label'] = 2
df.loc[df['JAN']==4562342690043,'label'] = 3
df.loc[df['JAN']==4562342690050,'label'] = 4
df.loc[df['JAN']==4562342690067,'label'] = 5
df.loc[df['JAN']==4562342690074,'label'] = 6
df.loc[df['JAN']==4562342690081,'label'] = 7


# In[20]:


#--------------------------------机器学习----------------------------------------
df = df.loc[(sale_data.week_start_47 >= 47)&(sale_data.week_start_47 <= 60),]
df.rename(columns={"年週":'yearweek','客数(POS客数)':'kyakusu'},inplace=True)

train_df = df.iloc[:-((time_key)-pred_start_time)*7, :]
test_df = df.iloc[-((time_key)-pred_start_time)*7:, :]


# In[21]:


#---------------------------------预测卖数------------------------------------------
x_train = train_df[['yearweek', 'year', 'week', 'year_start_47', 'week_start_47', 
       'holiday', 'event', 'kyakusu', 'nbk30', 'nbk50','baika','label']]
y_train = train_df[['売上数量']].reset_index(drop=True)

x_test = test_df[['yearweek', 'year', 'week', 'year_start_47', 'week_start_47', 
       'holiday', 'event', 'kyakusu','nbk30', 'nbk50','baika','label']].reset_index(drop=True)

#x_test_m = pd.concat([x_testm, nbk_df], axis=1)
#x_test_m['more'] = x_test_m['n_50perc']+x_test_m['n_30perc_copy']

rf = RandomForestRegressor(max_depth=8, n_estimators=150, random_state=2, n_jobs=4)
rf.fit(x_train, y_train)

res = []

for week_30 in range(pred_start_time, time_key+1):
    tmp=x_test.copy()
    tmp["nbk30"] = 0
    tmp.loc[tmp["week_start_47"] >= week_30, "nbk30"] = 1
    for week_50 in range(week_30, time_key+1):
        tmp["nbk50"]=0
        tmp.loc[tmp["week_start_47"] >= week_50, "nbk50"] = 1

        res.append(np.hstack([week_30, week_50, (rf.predict(tmp))]).tolist()) #时间 + 预测销售数量
        #print(tmp[["nbk30", "nbk50"]])        
#每种打折时间情况及销售情况
result = pd.DataFrame(res)


# In[22]:


#-------------------------------拆分结果-----------------------------------------------------
col_0 = ["nbk30time", "nbk50time"]
col_1 = []
col_1.append(["{}".format(i) for i in range(1,8)])
col_2 = col_1*2
col = col_0 + col_2

columns = np.hstack(col).tolist()
new_columns = pd.core.indexes.base.Index(columns)
result.columns = columns

result_nbk = result.iloc[:,:2]
result_pu1 = result["1"]
result_pu2= result["2"]
result_pu3= result["3"]
result_pu4= result["4"]
result_pu5= result["5"]
result_pu6= result["6"]
result_pu7= result["7"]

result1 = pd.concat([result_nbk,result_pu1],axis=1)
result2 = pd.concat([result_nbk,result_pu2],axis=1)
result3 = pd.concat([result_nbk,result_pu3],axis=1)
result4 = pd.concat([result_nbk,result_pu4],axis=1)
result5 = pd.concat([result_nbk,result_pu5],axis=1)
result6 = pd.concat([result_nbk,result_pu6],axis=1)
result7 = pd.concat([result_nbk,result_pu7],axis=1)

result1.columns = ['nbk30time','nbk50time','w59','w60']
result2.columns = ['nbk30time','nbk50time','w59','w60']
result3.columns = ['nbk30time','nbk50time','w59','w60']
result4.columns = ['nbk30time','nbk50time','w59','w60']
result5.columns = ['nbk30time','nbk50time','w59','w60']
result6.columns = ['nbk30time','nbk50time','w59','w60']
result7.columns = ['nbk30time','nbk50time','w59','w60']


# In[23]:


# 计算打3折销售数量，以及打5折销售数量
addlist1 = []

for i in range(result1.shape[0]): #0~20,time_key是61
    row = result1.iloc[i, :] #每一行
    if (row[1] < time_key)&(row[0] < time_key): #如果打5折的时间小于61 & 打3折的时间也小于61
        n50sum = row[(row[1]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum() #打5折能卖出去的数量 = 
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(row[1]-(pred_start_time-2)).astype('int')].sum()
    elif (row[1] == time_key)&(row[0] < time_key): #如果打5折的时间等于60 & 打3折的时间小于61
        n50sum = 0
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum()
    elif (row[1] == time_key)&(row[0] == time_key): #如果打5折的时间等于61 & 打3折的时间也小于61
        n50sum = 0
        n30sum = 0
    addlist1.append([i, n30sum, n50sum])

addlist1_df = pd.DataFrame(addlist1, columns=["ind", "n30sum", "n50sum"])

addlist2 = []

for i in range(result2.shape[0]): #0~20,time_key是61
    row = result2.iloc[i, :] #每一行
    if (row[1] < time_key)&(row[0] < time_key): #如果打5折的时间小于61 & 打3折的时间也小于61
        n50sum = row[(row[1]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum() #打5折能卖出去的数量 = 
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(row[1]-(pred_start_time-2)).astype('int')].sum()
    elif (row[1] == time_key)&(row[0] < time_key): #如果打5折的时间等于60 & 打3折的时间小于61
        n50sum = 0
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum()
    elif (row[1] == time_key)&(row[0] == time_key): #如果打5折的时间等于61 & 打3折的时间也小于61
        n50sum = 0
        n30sum = 0
    addlist2.append([i, n30sum, n50sum])

addlist2_df = pd.DataFrame(addlist2, columns=["ind", "n30sum", "n50sum"])

addlist3 = []

for i in range(result3.shape[0]): #0~20,time_key是61
    row = result3.iloc[i, :] #每一行
    if (row[1] < time_key)&(row[0] < time_key): #如果打5折的时间小于61 & 打3折的时间也小于61
        n50sum = row[(row[1]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum() #打5折能卖出去的数量 = 
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(row[1]-(pred_start_time-2)).astype('int')].sum()
    elif (row[1] == time_key)&(row[0] < time_key): #如果打5折的时间等于60 & 打3折的时间小于61
        n50sum = 0
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum()
    elif (row[1] == time_key)&(row[0] == time_key): #如果打5折的时间等于61 & 打3折的时间也小于61
        n50sum = 0
        n30sum = 0
    addlist3.append([i, n30sum, n50sum])

addlist3_df = pd.DataFrame(addlist3, columns=["ind", "n30sum", "n50sum"])

addlist4 = []

for i in range(result4.shape[0]): #0~20,time_key是61
    row = result4.iloc[i, :] #每一行
    if (row[1] < time_key)&(row[0] < time_key): #如果打5折的时间小于61 & 打3折的时间也小于61
        n50sum = row[(row[1]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum() #打5折能卖出去的数量 = 
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(row[1]-(pred_start_time-2)).astype('int')].sum()
    elif (row[1] == time_key)&(row[0] < time_key): #如果打5折的时间等于60 & 打3折的时间小于61
        n50sum = 0
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum()
    elif (row[1] == time_key)&(row[0] == time_key): #如果打5折的时间等于61 & 打3折的时间也小于61
        n50sum = 0
        n30sum = 0
    addlist4.append([i, n30sum, n50sum])

addlist4_df = pd.DataFrame(addlist4, columns=["ind", "n30sum", "n50sum"])

addlist5 = []

for i in range(result5.shape[0]): #0~20,time_key是61
    row = result5.iloc[i, :] #每一行
    if (row[1] < time_key)&(row[0] < time_key): #如果打5折的时间小于61 & 打3折的时间也小于61
        n50sum = row[(row[1]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum() #打5折能卖出去的数量 = 
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(row[1]-(pred_start_time-2)).astype('int')].sum()
    elif (row[1] == time_key)&(row[0] < time_key): #如果打5折的时间等于60 & 打3折的时间小于61
        n50sum = 0
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum()
    elif (row[1] == time_key)&(row[0] == time_key): #如果打5折的时间等于61 & 打3折的时间也小于61
        n50sum = 0
        n30sum = 0
    addlist5.append([i, n30sum, n50sum])

addlist5_df = pd.DataFrame(addlist5, columns=["ind", "n30sum", "n50sum"])

addlist6 = []

for i in range(result6.shape[0]): #0~20,time_key是61
    row = result6.iloc[i, :] #每一行
    if (row[1] < time_key)&(row[0] < time_key): #如果打5折的时间小于61 & 打3折的时间也小于61
        n50sum = row[(row[1]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum() #打5折能卖出去的数量 = 
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(row[1]-(pred_start_time-2)).astype('int')].sum()
    elif (row[1] == time_key)&(row[0] < time_key): #如果打5折的时间等于60 & 打3折的时间小于61
        n50sum = 0
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum()
    elif (row[1] == time_key)&(row[0] == time_key): #如果打5折的时间等于61 & 打3折的时间也小于61
        n50sum = 0
        n30sum = 0
    addlist6.append([i, n30sum, n50sum])

addlist6_df = pd.DataFrame(addlist6, columns=["ind", "n30sum", "n50sum"])

addlist7 = []

for i in range(result7.shape[0]): #0~20,time_key是61
    row = result7.iloc[i, :] #每一行
    if (row[1] < time_key)&(row[0] < time_key): #如果打5折的时间小于61 & 打3折的时间也小于61
        n50sum = row[(row[1]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum() #打5折能卖出去的数量 = 
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(row[1]-(pred_start_time-2)).astype('int')].sum()
    elif (row[1] == time_key)&(row[0] < time_key): #如果打5折的时间等于60 & 打3折的时间小于61
        n50sum = 0
        n30sum = row[(row[0]-(pred_start_time-2)).astype('int'):(time_key+2-pred_start_time)].sum()
    elif (row[1] == time_key)&(row[0] == time_key): #如果打5折的时间等于61 & 打3折的时间也小于61
        n50sum = 0
        n30sum = 0
    addlist7.append([i, n30sum, n50sum])

addlist7_df = pd.DataFrame(addlist7, columns=["ind", "n30sum", "n50sum"])


# In[24]:


#---------------------------------------计算库存--------------------------------------------------------
#纳品数表
buy_0 = {'buy_nu':[2610,870,2610,870,2610,2610,870],'JAN':[4562342690029,4562342690036,4562342690043,4562342690050,4562342690067,4562342690074,4562342690081]}
buy = pd.DataFrame(buy_0)

#计算库存总表
sale_left = sale_data2.groupby(['年週','year','week','year_start_47','week_start_47','JAN','品種名'])['売上数量'].sum().reset_index()
left = sale_left.merge(buy,how='left',on='JAN')
left.rename(columns = {'week_start_47':'nbk30time'},inplace=True)


# In[25]:


#每件产品的期初库存
left1 = left.loc[left['JAN'] == 4562342690029,]
left1['end_period_sales'] = left1['売上数量'].cumsum()
left1['end_period_left'] = left1['buy_nu'] - left1['end_period_sales']
a1 = left1.loc[left1['nbk30time']==pred_start_time-1,'end_period_left'].tolist()
left1['begin_period_left'] = a1[0]
left1 = left1[['nbk30time','begin_period_left']] 

left2 = left.loc[left['JAN'] == 4562342690036,]
left2['end_period_sales'] = left2['売上数量'].cumsum()
left2['end_period_left'] = left2['buy_nu'] - left2['end_period_sales']
a2 = left2.loc[left2['nbk30time']==pred_start_time-1,'end_period_left'].tolist()
left2['begin_period_left'] = a2[0]
left2 = left2[['nbk30time','begin_period_left']]

left3 = left.loc[left['JAN'] == 4562342690043,]
left3['end_period_sales'] = left3['売上数量'].cumsum()
left3['end_period_left'] = left3['buy_nu'] - left3['end_period_sales']
a3 = left3.loc[left3['nbk30time']==pred_start_time-1,'end_period_left'].tolist()
left3['begin_period_left'] = a3[0]
left3 = left3[['nbk30time','begin_period_left']]

left4 = left.loc[left['JAN'] == 4562342690050,]
left4['end_period_sales'] = left4['売上数量'].cumsum()
left4['end_period_left'] = left4['buy_nu'] - left4['end_period_sales']
a4 = left4.loc[left4['nbk30time']==pred_start_time-1,'end_period_left'].tolist()
left4['begin_period_left'] = a4[0]
left4 = left4[['nbk30time','begin_period_left']]

left5 = left.loc[left['JAN'] == 4562342690067,]
left5['end_period_sales'] = left5['売上数量'].cumsum()
left5['end_period_left'] = left5['buy_nu'] - left5['end_period_sales']
a5 = left5.loc[left5['nbk30time']==pred_start_time-1,'end_period_left'].tolist()
left5['begin_period_left'] = a5[0]
left5 = left5[['nbk30time','begin_period_left']]

left6 = left.loc[left['JAN'] == 4562342690074,]
left6['end_period_sales'] = left6['売上数量'].cumsum()
left6['end_period_left'] = left6['buy_nu'] - left6['end_period_sales']
a6 = left6.loc[left6['nbk30time']==pred_start_time-1,'end_period_left'].tolist()
left6['begin_period_left'] = a6[0]
left6 = left6[['nbk30time','begin_period_left']]

left7 = left.loc[left['JAN'] == 4562342690081,]
left7['end_period_sales'] = left7['売上数量'].cumsum()
left7['end_period_left'] = left7['buy_nu'] - left7['end_period_sales']
a7 = left7.loc[left7['nbk30time']==pred_start_time-1,'end_period_left'].tolist()
left7['begin_period_left'] = a7[0]
left7 = left7[['nbk30time','begin_period_left']]


# In[26]:


predata = origin.loc[(origin['year_start_47']==year-1)&(origin['week_start_47']>=pred_start_time),]
predata = predata[['year_start_47','week_start_47','JAN','品種名','売上数量']].reset_index(drop=True)

predata1 = predata.loc[predata['JAN']==4562342690029,['week_start_47','売上数量']]
predata2 = predata.loc[predata['JAN']==4562342690036,['week_start_47','売上数量']]
predata3 = predata.loc[predata['JAN']==4562342690043,['week_start_47','売上数量']]
predata4 = predata.loc[predata['JAN']==4562342690050,['week_start_47','売上数量']]
predata5 = predata.loc[predata['JAN']==4562342690067,['week_start_47','売上数量']]
predata6 = predata.loc[predata['JAN']==4562342690074,['week_start_47','売上数量']]
predata7 = predata.loc[predata['JAN']==4562342690081,['week_start_47','売上数量']]

predata1.columns = ['nbk30time','売上数量']
predata2.columns = ['nbk30time','売上数量']
predata3.columns = ['nbk30time','売上数量']
predata4.columns = ['nbk30time','売上数量']
predata5.columns = ['nbk30time','売上数量']
predata6.columns = ['nbk30time','売上数量']
predata7.columns = ['nbk30time','売上数量']


# In[27]:


fin_1 = result1.merge(left1,how='left',on='nbk30time').fillna(a1[0])
fin_1["pred_sum"] = fin_1.iloc[:, 2:(time_key+2-pred_start_time)].apply(axis = 1, func='sum') #预计销售总量
fin_1["sale_sum"] = predata1['売上数量'].sum()
fin_1['more'] = fin_1['pred_sum'] - fin_1['sale_sum']
fin_1["n30sum"] = addlist1_df['n30sum'].copy()                                                #打3折销售总量
fin_1["n50sum"] = addlist1_df["n50sum"].copy()                                                #打5折销售总量
fin_1["n_teika"] = round(fin_1["pred_sum"] - fin_1["n50sum"] - fin_1["n30sum"], 3)            #原价销售数量
fin_1["nbk_loss"] = 0.3*fin_1["n30sum"] + 0.5*fin_1["n50sum"]                                 #如果商品价值是1，那么如果打折销售一共损失了多少价值
fin_1['haiki_loss'] = fin_1['begin_period_left'] - fin_1['pred_sum']                          #经过打折可能没有卖掉的数量或价值
fin_1['uriage'] = fin_1['pred_sum'] - fin_1['nbk_loss']                                       #如果商品价值是1，那么通过打折销售一共获得的价值
fin_1['arari'] = fin_1['uriage'] - fin_1['haiki_loss']                                        #最终通过打折获得的价值

fin_1 = fin_1.fillna(0)

fin_2 = result2.merge(left2,how='left',on='nbk30time').fillna(a2[0])

fin_2["pred_sum"] = fin_2.iloc[:, 2:(time_key+2-pred_start_time)].apply(axis = 1, func='sum') #预计销售总量
fin_2["sale_sum"] = predata2['売上数量'].sum()
fin_2['more'] = fin_2['pred_sum'] - fin_2['sale_sum']
fin_2["n30sum"] = addlist2_df['n30sum'].copy()                                                #打3折销售总量
fin_2["n50sum"] = addlist2_df["n50sum"].copy()                                                #打5折销售总量
fin_2["n_teika"] = round(fin_2["pred_sum"] - fin_2["n50sum"] - fin_2["n30sum"], 3)            #原价销售数量
fin_2["nbk_loss"] = 0.3*fin_2["n30sum"] + 0.5*fin_2["n50sum"]                                 #如果商品价值是1，那么如果打折销售一共损失了多少价值
fin_2['haiki_loss'] = fin_2['begin_period_left'] - fin_2['pred_sum']                          #经过打折可能没有卖掉的数量或价值
fin_2['uriage'] = fin_2['pred_sum'] - fin_2['nbk_loss']                                       #如果商品价值是1，那么通过打折销售一共获得的价值
fin_2['arari'] = fin_2['uriage'] - fin_2['haiki_loss']                                        #最终通过打折获得的价值

fin_2 = fin_2.fillna(a2[0])

fin_3 = result3.merge(left3,how='left',on='nbk30time').fillna(a3[0])

fin_3["pred_sum"] = fin_3.iloc[:, 2:(time_key+2-pred_start_time)].apply(axis = 1, func='sum') #预计销售总量
fin_3["sale_sum"] = predata3['売上数量'].sum()
fin_3['more'] = fin_3['pred_sum'] - fin_3['sale_sum']
fin_3["n30sum"] = addlist3_df['n30sum'].copy()                                                #打3折销售总量
fin_3["n50sum"] = addlist3_df["n50sum"].copy()                                                #打5折销售总量
fin_3["n_teika"] = round(fin_3["pred_sum"] - fin_3["n50sum"] - fin_3["n30sum"], 3)            #原价销售数量
fin_3["nbk_loss"] = 0.3*fin_3["n30sum"] + 0.5*fin_3["n50sum"]                                 #如果商品价值是1，那么如果打折销售一共损失了多少价值
fin_3['haiki_loss'] = fin_3['begin_period_left'] - fin_3['pred_sum']                          #经过打折可能没有卖掉的数量或价值
fin_3['uriage'] = fin_3['pred_sum'] - fin_3['nbk_loss']                                       #如果商品价值是1，那么通过打折销售一共获得的价值
fin_3['arari'] = fin_3['uriage'] - fin_3['haiki_loss']                                        #最终通过打折获得的价值

fin_3= fin_3.fillna(a3[0])

fin_4 = result4.merge(left4,how='left',on='nbk30time').fillna(a4[0])

fin_4["pred_sum"] = fin_4.iloc[:, 2:(time_key+2-pred_start_time)].apply(axis = 1, func='sum') #预计销售总量
fin_4["sale_sum"] = predata4['売上数量'].sum()
fin_4['more'] = fin_4['pred_sum'] - fin_4['sale_sum']
fin_4["n30sum"] = addlist4_df['n30sum'].copy()                                                #打3折销售总量
fin_4["n50sum"] = addlist4_df["n50sum"].copy()                                                #打5折销售总量
fin_4["n_teika"] = round(fin_4["pred_sum"] - fin_4["n50sum"] - fin_4["n30sum"], 3)            #原价销售数量
fin_4["nbk_loss"] = 0.3*fin_4["n30sum"] + 0.5*fin_4["n50sum"]                                 #如果商品价值是1，那么如果打折销售一共损失了多少价值
fin_4['haiki_loss'] = fin_4['begin_period_left'] - fin_4['pred_sum']                          #经过打折可能没有卖掉的数量或价值
fin_4['uriage'] = fin_4['pred_sum'] - fin_4['nbk_loss']                                       #如果商品价值是1，那么通过打折销售一共获得的价值
fin_4['arari'] = fin_4['uriage'] - fin_4['haiki_loss']                                        #最终通过打折获得的价值

fin_4 = fin_4.fillna(a4[0])

fin_5 = result5.merge(left5,how='left',on='nbk30time').fillna(a5[0])

fin_5["pred_sum"] = fin_5.iloc[:, 2:(time_key+2-pred_start_time)].apply(axis = 1, func='sum') #预计销售总量
fin_5["sale_sum"] = predata5['売上数量'].sum()
fin_5['more'] = fin_5['pred_sum'] - fin_5['sale_sum']
fin_5["n30sum"] = addlist5_df['n30sum'].copy()                                                #打3折销售总量
fin_5["n50sum"] = addlist5_df["n50sum"].copy()                                                #打5折销售总量
fin_5["n_teika"] = round(fin_5["pred_sum"] - fin_5["n50sum"] - fin_5["n30sum"], 3)            #原价销售数量
fin_5["nbk_loss"] = 0.3*fin_5["n30sum"] + 0.5*fin_5["n50sum"]                                 #如果商品价值是1，那么如果打折销售一共损失了多少价值
fin_5['haiki_loss'] = fin_5['begin_period_left'] - fin_5['pred_sum']                          #经过打折可能没有卖掉的数量或价值
fin_5['uriage'] = fin_5['pred_sum'] - fin_5['nbk_loss']                                       #如果商品价值是1，那么通过打折销售一共获得的价值
fin_5['arari'] = fin_5['uriage'] - fin_5['haiki_loss']                                        #最终通过打折获得的价值

fin_5 = fin_5.fillna(a5[0])

fin_6 = result6.merge(left6,how='left',on='nbk30time').fillna(a6[0])

fin_6["pred_sum"] = fin_6.iloc[:, 2:(time_key+2-pred_start_time)].apply(axis = 1, func='sum') #预计销售总量
fin_6["sale_sum"] = predata6['売上数量'].sum()
fin_6['more'] = fin_6['pred_sum'] - fin_6['sale_sum']
fin_6["n30sum"] = addlist6_df['n30sum'].copy()                                                #打3折销售总量
fin_6["n50sum"] = addlist6_df["n50sum"].copy()                                                #打5折销售总量
fin_6["n_teika"] = round(fin_6["pred_sum"] - fin_6["n50sum"] - fin_6["n30sum"], 3)            #原价销售数量
fin_6["nbk_loss"] = 0.3*fin_6["n30sum"] + 0.5*fin_6["n50sum"]                                 #如果商品价值是1，那么如果打折销售一共损失了多少价值
fin_6['haiki_loss'] = fin_6['begin_period_left'] - fin_6['pred_sum']                          #经过打折可能没有卖掉的数量或价值
fin_6['uriage'] = fin_6['pred_sum'] - fin_6['nbk_loss']                                       #如果商品价值是1，那么通过打折销售一共获得的价值
fin_6['arari'] = fin_6['uriage'] - fin_6['haiki_loss']                                        #最终通过打折获得的价值

fin_6 = fin_6.fillna(a6[0])

fin_7 = result7.merge(left7,how='left',on='nbk30time').fillna(a7[0])

fin_7["pred_sum"] = fin_7.iloc[:, 2:(time_key+2-pred_start_time)].apply(axis = 1, func='sum') #预计销售总量
fin_7["sale_sum"] = predata7['売上数量'].sum()
fin_7['more'] = fin_7['pred_sum'] - fin_7['sale_sum']
fin_7["n30sum"] = addlist7_df['n30sum'].copy()                                                #打3折销售总量
fin_7["n50sum"] = addlist7_df["n50sum"].copy()                                                #打5折销售总量
fin_7["n_teika"] = round(fin_7["pred_sum"] - fin_7["n50sum"] - fin_7["n30sum"], 3)            #原价销售数量
fin_7["nbk_loss"] = 0.3*fin_7["n30sum"] + 0.5*fin_7["n50sum"]                                 #如果商品价值是1，那么如果打折销售一共损失了多少价值
fin_7['haiki_loss'] = fin_7['begin_period_left'] - fin_7['pred_sum']                          #经过打折可能没有卖掉的数量或价值
fin_7['uriage'] = fin_7['pred_sum'] - fin_7['nbk_loss']                                       #如果商品价值是1，那么通过打折销售一共获得的价值
fin_7['arari'] = fin_7['uriage'] - fin_7['haiki_loss']                                        #最终通过打折获得的价值

fin_7 = fin_7.fillna(a7[0])


# In[34]:


fin_7

