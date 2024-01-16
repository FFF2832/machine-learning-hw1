import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("./HW1_house_data.csv")
#print(data.shape) #(21613,21)
#print(data["date"].tail())

#將date 欄拆成sale_yr sale_month sale_day 三欄，然後將date 欄刪掉

# .str.slice(0,4)的method將data中原本的20141013T000000，切成2014
data['sale_yr'] = pd.to_numeric(data['date'].str.slice(0, 4)) 
 # .str.slice(4,6)的method將data中原本的20141013T000000，切成10   
data['sale_month'] = pd.to_numeric(data['date'].str.slice(4, 6)) 
 # .str.slice(6,8)的method將data中原本的20141013T000000，切成13
data['sale_day'] = pd.to_numeric(data['date'].str.slice(6, 8))   
data.drop('date', axis="columns", inplace=True) 

#inplace=True代表對data刪除date欄後直接以新資料取代data(回傳值為None)，所以就不用再對data重新賦值，也可改成 data=data.drop('date', axis="columns")
# data=data.drop('date', axis="columns") #也可改成這句
#print(data[["sale_yr","sale_month","sale_day"]].tail())

data.drop('id', axis="columns", inplace=True)

#刪除各欄(特徵)資料中有缺失值的整列資料
#print(data.shape) #,val_x.shape,test_x.shape)
data_isna=data.isna().sum()
data = data.dropna()  #刪除有缺失值的整列資料
#print(np.array(data).shape)


#print(data.tail())
#對data 的 yr_renovated欄資料重新編碼
#將原本代表哪一年整修的yr_renovated特徵值改成renovated與renovated_yrs兩個特徵值，兩者分別代表是否整修以及整修經過幾年(為何要改？因為原本的值不是0就是很大，但0代表沒有整修過，是否相較於值很大是比較不好？似乎不見得)
renovated=data['yr_renovated'].copy()
renovated[renovated>0]=1
# renovated_yrs_temp=data['sale_yr']-data['yr_renovated']
renovated_yrs=(data['sale_yr']-data['yr_renovated']).copy() #renovated_yrs_temp.copy()
renovated_yrs[data['sale_yr']==renovated_yrs]=0
df = pd.DataFrame({'renovated': renovated,'renovated_yrs': renovated_yrs,})
data=pd.concat([data.drop(['yr_renovated'], axis="columns"), df], axis=1)
#print(data.tail())


# #one-hot encoding (one-hot vector)
#方法一：
# print(np.array(data['waterfront'][:5])) #重編碼前
# train_waterfront_dummies = pd.get_dummies(data['waterfront']) 
# data.drop('waterfront', axis="columns", inplace=True)
# data=pd.concat([data, train_waterfront_dummies], axis=1)
# print(np.array(train_waterfront_dummies[:5])) #重編碼後
# print(data.tail())

#方法二：
# condition = data.pop('condition')
# condition_unique=np.unique(np.array(condition))
# for i in range(len(condition_unique)):
#     data[f'condition{condition_unique[i]}'] = (condition == condition_unique[i])*1.0
# print(data.tail())


# # print(data.columns)
#print(data.info())
data_stats = data.describe()
# data_stats = data_stats.transpose() #data_stats.T
#print(data_stats)
# # print(data['sqft_living'].describe()) #data.loc[:,'sqft_living'].describe()
# # print(data[['bedrooms','sqft_living','sqft_lot']].describe())
# # print(data.loc[:, 'bedrooms':'sqft_lot'].describe())
# # print(data.loc[:, ['bedrooms','sqft_living','sqft_lot']].describe())

#正規化(normalization)
mean = data.mean()
std = data.std()
normed_data = (data - mean) / std  # (data - data_stats['mean']) / data_stats['std']
# print(normed_data.head())


colums=list(normed_data.columns)
#print(len(colums)) #23
#對 yr_renovated 重新編碼的方式
# 畫出各feature分別與price的散佈關係圖(分析是否有關係)
plt.figure(figsize=(12, 7))
plt.subplots_adjust(hspace=0.5, wspace=1)
for i, feature in enumerate(normed_data):
    plt.subplot(5, 5, i + 1)
    sns.scatterplot(x=feature, y='price', data=normed_data, alpha=0.5)
    plt.xticks(fontsize=7) 
    plt.yticks(fontsize=7) 
    plt.xlabel(feature, fontsize=9.5)
    plt.ylabel('price', fontsize=9.5, labelpad=0.25)
plt.savefig('scatter_train_all.png')

# plt.figure(figsize=(20, 10))
# colums=list(data.columns)
# print(len(colums)) #23
# sns.pairplot(data[['price','sqft_living','grade','sqft_above']], diag_kind="kde")

#畫熱力圖，透過相關係數去觀察有哪些特徵變數和目標變數有較高的相關性等等
plt.figure(figsize=(20, 10))
correlation_matrix = data.corr().round(2)
sns.heatmap(data=correlation_matrix, annot = True) # annot = True 讓我們可以把數字標進每個格子裡
plt.savefig('hot_figure.png')
# plt.show()


#畫盒鬚圖
# plt.figure()
# sns.boxplot(x="value", y="variable", orient='h', data=pd.melt(normed_data[['sqft_living','sqft_above','sqft_living15','grade']])); # pd.melt()把多個欄位合併成一個
# plt.savefig('box_pre.png')
#plt.show()

# print(normed_data.shape)
# indices=[]
# features = ['sqft_living','sqft_above','sqft_living15','grade']
# for i, feature in enumerate(features):
#     q1, q3 = np.percentile(normed_data[feature], [25, 75])
#     above = q3 + 1.5 * (q3 - q1)
#     below = q1 - 1.5 * (q3 - q1)
#     drop_index=normed_data[ (normed_data[feature]<below) | (normed_data[feature]>above) ].index
#     indices.append(np.array(drop_index))
#     print(len(indices[-1]),indices[-1][-5:])
# print(np.unique(np.concatenate(indices)).shape)
# normed_data=normed_data.drop(np.concatenate(indices))
# print(normed_data.shape)
# plt.figure()
# sns.boxplot(x="value", y="variable", orient='h', data=pd.melt(normed_data[['sqft_living','sqft_above','sqft_living15','grade']])); # pd.melt()把多個欄位合併成一個
# plt.savefig('box_post.png')
# plt.show()

normed_data.drop(['sale_yr','sale_month','sale_day','zipcode'], axis="columns", inplace=True)


# 6: 1: 3 = training data num: validation data num: test data num
# 7: 1 :2
# 0.8*num*0.8: 0.8*num*0.2: 0.2*num = 0.64: 0.16: 0.2 = 16: 4: 5
#資料分割，將讀出來的資料切成訓練集、驗證集與測試集
data_num = normed_data.shape[0]
# 取得一筆與data數量相同的亂數索引，主要目的是用於打散資料
indices = np.random.permutation(data_num)
# print(indices)
# 並將亂數索引值分為train_indices、val_indices與test_indices，劃分比例為6: 1: 3
train_indices = indices[:round(data_num*(1-0.4))] #0:12 (以data_num=20為例)
val_indices = indices[round(data_num*(1-0.4)):round(0.7*data_num)] #12:14 (以data_num=20為例)
test_indices = indices[round(data_num*0.7):] #14:20 (以data_num=20為例)
# print(train_indices)
# print(val_indices)
# print(test_indices)
test_data=pd.DataFrame(np.array(normed_data)[test_indices], columns=normed_data.columns)
test_data.to_csv('test_data.csv',index=False)
val_data=pd.DataFrame(np.array(normed_data)[val_indices], columns=normed_data.columns)
val_data.to_csv('val_data.csv',index=False)
train_data=pd.DataFrame(np.array(normed_data)[train_indices], columns=normed_data.columns)
train_data.to_csv('train_data.csv',index=False)


train_y = np.array(train_data['price']).copy() #np.array(train_data['price'].apply(np.log)).copy()
val_y = np.array(val_data['price']).copy() #np.array(val_data['price'].apply(np.log)).copy()
test_y = np.array(test_data['price']).copy() #np.array(val_data['price'].apply(np.log)).copy()
train_x=train_data.drop('price', axis='columns')
val_x=val_data.drop('price', axis='columns')
test_x=test_data.drop('price', axis='columns')
#線性回歸

x, y = train_data.loc[:, ['sqft_living']], train_data.loc[:, ['price']]
lr = LinearRegression()
lr.fit(x, y)
print('w_1 =', lr.coef_[0])
print('w_0 =', lr.intercept_)

# 創建新的圖形窗口
fig = plt.figure()

# 繪製散點圖
plt.scatter(x, y, facecolor='xkcd:azure', edgecolor='black', s=20)
plt.xlabel('sqft_living', fontsize=14)
plt.ylabel("price", fontsize=14)

# 繪製迴歸線
n_x = np.linspace(x.min(), x.max(), 100)
n_y = lr.intercept_ + lr.coef_[0] * n_x
plt.plot(n_x, n_y, color='r', lw=3)

# 調整圖形窗口的間距
plt.subplots_adjust(hspace=0.5, wspace=1)

# 保存圖片
plt.savefig('LinearRegression1.png')




x, y = train_data.loc[:, ['grade']], train_data.loc[:, ['price']]
lr = LinearRegression()
lr.fit(x, y)
print('w_1 =', lr.coef_[0])
print('w_0 =', lr.intercept_)

# 創建新的圖形窗口
fig = plt.figure()

# 繪製散點圖
plt.scatter(x, y, facecolor='xkcd:azure', edgecolor='black', s=20)
plt.xlabel('grade', fontsize=14)
plt.ylabel("price", fontsize=14)

# 繪製迴歸線
n_x = np.linspace(x.min(), x.max(), 100)
n_y = lr.intercept_ + lr.coef_[0] * n_x
plt.plot(n_x, n_y, color='r', lw=3)

# 調整圖形窗口的間距
plt.subplots_adjust(hspace=0.5, wspace=1)

# 保存圖片
plt.savefig('LinearRegression2.png')

# 顯示圖形
#plt.show()

#MSE
lr = LinearRegression()
lr.fit(train_x, train_y)
train_y_pred = lr.predict(train_x)
test_y_pred = lr.predict(test_x)
print('MSE(training): %.3f, MSE(testing): %.3f' %( 
    mean_squared_error(train_y, train_y_pred), 
    mean_squared_error(test_y, test_y_pred)))


def adj_R2(r2, n, k):
    return 1 - (n-1)*(1-r2)/(n-k-1)


def linReg_adj_R2(X, y):
    train_x, test_x, train_y, test_y = train_test_split(X, y, 
                                                        test_size=0.2,
                                                        random_state=0)
    lr = LinearRegression().fit(train_x, train_y)
    
    # 更新预测值
    train_y_pred = lr.predict(train_x)
    test_y_pred = lr.predict(test_x)

    r2_train = r2_score(train_y, train_y_pred)
    r2_test = r2_score(test_y, test_y_pred)

    print('Adj. R^2(training): %.3f, Adj. R^2(testing): %.3f' %( 
        adj_R2(r2_train, train_x.shape[0], train_x.shape[1]), 
        adj_R2(r2_test, test_x.shape[0], test_x.shape[1])))

# 在需要计算R^2得分的地方调用 linReg_adj_R2 函数
X, y = data.loc[:, ['sqft_living','sqft_living15','grade']], data.loc[:, ['price']]
train_X, test_x, train_y, test_y = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)

lr = LinearRegression().fit(train_X, train_y)

train_y_pred = lr.predict(train_X)
test_y_pred = lr.predict(test_x)
test_y_pred = lr.predict(test_x)

print('R^2(training): %.3f, R^2(testing): %.3f' %( 
    r2_score(train_y, train_y_pred), 
    r2_score(test_y, test_y_pred)))

print(X.shape)
print('=== 原始數據 ===')
linReg_adj_R2(X, y)

np.random.seed(0)
rand_n = np.random.rand(X.shape[0], 4)
df_rand = pd.DataFrame(data=rand_n, 
                       columns=list('ABCD'))
X = pd.concat([X, df_rand], axis=1)

#預測結果
print(train_y_pred)
print(test_y_pred)
# print(X.shape)
# print('=== 隨機增加無關的 4 個特徵 ===')
# linReg_adj_R2(X, y)
# print(test_y_pred)


# #把price轉回沒有正規化
# train_y = np.array(train_data['price']).copy()*std['price']+mean['price']
# val_y = np.array(val_data['price']).copy()*std['price']+mean['price']
# test_y = np.array(test_data['price']).copy()*std['price']+mean['price']
# train_x=train_data.drop('price', axis='columns')
# val_x=val_data.drop('price', axis='columns')
# test_x=test_data.drop('price', axis='columns')


#殘差分析
import statsmodels.api as sm
from scipy.stats import shapiro

X, y = data.loc[:, ['sqft_living','sqft_living15']], data.loc[:, ['price']]
X = sm.add_constant(X)  # 增加常數行作為截距項
model = sm.OLS(y, X).fit()

# Shapiro-Wilk 常態性檢定
stat, p = shapiro(model.resid)
print('Statistics: %.3f, p-value: %.3f' % (stat, p))
alpha = 0.05

if p > alpha:
    print('看起來是常態分布（無法拒絕H0）')
else:
    print('看起來不是常態分布（拒絕H0）')


from scipy.stats import jarque_bera

# Jarque-Bera 常態性檢定
stat, p = jarque_bera(model.resid)
print('Statistics: %.3f, p-value: %.3f' % (stat, p))
alpha = 0.05

if p > alpha:
    print('看起來是常態分布（無法拒絕H0）')
else:
    print('看起來不是常態分布（拒絕H0）')

from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(model.resid)
print('dw: %.3f' % dw)

if 2 <= dw <= 4:
    print('誤差項獨立')
elif 0 <= dw < 2:
    print('誤差項不獨立')
else:
    print('計算錯誤')




# 創建模型儲存目錄：
dir_='./lab1-logs'
model_dir = f'{dir_}/model'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir) # ,mode=0o777)

#建立與訓練網路(有使用dropout、加入L1或L2 正則化，以及測試不同dropout rate、神經元數、batch size以及activation function，找出最佳組合)
neurons_num=128
dropout_rate=0.15
epochs_num=1500
activation_='elu' #'leaky_relu' #
batchSize=128
earlyStop_pati=300
model = keras.Sequential(name='model')
model.add(layers.Dense(neurons_num, 
                         kernel_regularizer=keras.regularizers.l2(0.001), 
                         activation=activation_, input_shape=(train_x.shape[1],)))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(neurons_num, kernel_regularizer=keras.regularizers.l2(0.001), activation=activation_))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(neurons_num, kernel_regularizer=keras.regularizers.l2(0.001), activation=activation_))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(neurons_num, kernel_regularizer=keras.regularizers.l2(0.001), activation=activation_))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(1))
model.compile(keras.optimizers.Adam(0.001),
                loss=keras.losses.MeanSquaredError(), #keras.losses.MeanAbsoluteError(), #
                metrics=[keras.metrics.MeanAbsoluteError()]) #[keras.metrics.MeanSquaredError()]) #
log_dir = f'{dir_}/log' #os.path.join(dir_, 'log')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(f'{model_dir}/Best-model.h5',monitor='val_loss',save_best_only=True,mode='min')
# model_mckp = keras.callbacks.ModelCheckpoint(f'{model_dir}/Best-model.h5',monitor='mean_absolute_error',save_best_only=True,mode='min')
earlyStop=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', #'val_mean_squared_error', #'mean_absolute_error',
    patience=earlyStop_pati,
    mode='min', #'auto',
    restore_best_weights=True
    # min_delta=0, verbose=0, baseline=None,
)
history = model.fit(train_x, train_y, batch_size=batchSize, epochs=epochs_num, validation_data=(val_x, val_y), callbacks=[model_cbk, model_mckp, earlyStop])

#畫出訓練的 epoch curve
hist = pd.DataFrame(history.history)
# print(hist.columns)
hist['epoch'] = history.epoch
def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss') #目前的loss 是MAE
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Epoch_curve_loss')
    plt.savefig('epoch_curve.png')

    plt.figure()
    plt.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
    plt.plot(history.history['val_mean_absolute_error'], label='val_mean_absolute_error')
    # plt.plot(history.history['val_mean_squared_error'], label='val_mean_squared_error')
    # plt.plot(history.history['val_mean_squared_error'], label='val_mean_squared_error')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Epoch_curve_MAE')
    plt.savefig('epoch_curve_MAE.png')
    # plt.ylabel('MSE') 
    # plt.title('Epoch_curve_MSE') 
    # plt.savefig('epoch_curve_MSE.png') 
plot_loss(history)


# model = keras.models.load_model(f'{model_dir}/Best-model.h5')
y_pred = model.predict(test_x) #*std['price']+mean['price']
# print(len(y_pred))
loss, mae = model.evaluate(test_x, test_y, verbose=2)
print("Testing set Mean squared Error: {:.3f}".format(loss))
print("Testing set Mean absolute Error: {:.3f}".format(mae))

# print(test_y.shape,np.squeeze(y_pred).shape)
from sklearn.metrics import mean_squared_error, mean_absolute_error
print('MSE(testing): %.3f' %(mean_squared_error(test_y, np.squeeze(y_pred))))
print('MAE(testing): %.3f' %(mean_absolute_error(test_y, np.squeeze(y_pred))))
#預測



#     from sklearn.preprocessing import StandardScaler
# import seaborn as sns
# # 創建模型對象
# model = sm.OLS(y, X)

# # 擬合模型
# result = model.fit()

# # 創建殘差數據框
# df_resid = pd.DataFrame()
# df_resid['y_pred'] = result.fittedvalues
# df_resid['resid'] = result.resid
# kws = {'color':'red', 'lw':3}
# df_resid = StandardScaler().fit_transform(df_resid)
# plt.subplots_adjust(hspace=0.5, wspace=1)
# sns.residplot(x=df_resid[:, 0], y=df_resid[:, 1], 
#               lowess=True, line_kws=kws)
# plt.xlabel('Predicted Values (standardization)', fontsize=14)
# plt.ylabel('Residual (standardization)', fontsize=14)
# plt.savefig('predict.png')

# # 創建模型儲存目錄：
# model_dir = 'lab1-logs/models'
# if not os.path.isdir(model_dir):
#     os.makedirs(model_dir) # ,mode=0o777)

# #建立與訓練網路(有使用dropout、加入L1或L2 正則化，以及測試不同dropout rate、神經元數、batch size以及activation function，找出最佳組合)
# neurons_num=128
# dropout_rate=0.15
# epochs_num=10
# activation_='elu' #'leaky_relu' #
# batchSize=128
# earlyStop_pati=300
# model = keras.Sequential(name='model-4')
# model.add(layers.Dense(neurons_num, 
#                          kernel_regularizer=keras.regularizers.l2(0.001), 
#                          activation=activation_, input_shape=(train_x.shape[1],)))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Dense(neurons_num, kernel_regularizer=keras.regularizers.l2(0.001), activation=activation_))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Dense(neurons_num, kernel_regularizer=keras.regularizers.l2(0.001), activation=activation_))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Dense(neurons_num, kernel_regularizer=keras.regularizers.l2(0.001), activation=activation_))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Dense(1))
# model.compile(keras.optimizers.Adam(0.001),
#                 loss=keras.losses.MeanAbsoluteError(), #keras.losses.MeanSquaredError(), #
#                 metrics=[keras.metrics.MeanSquaredError()]) #[keras.metrics.MeanAbsoluteError()]) #
# log_dir = os.path.join('lab1-logs', 'model-4')
# model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
# model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model.h5',monitor='val_mean_squared_error',save_best_only=True,mode='min')
# # model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model.h5',monitor='mean_absolute_error',save_best_only=True,mode='min')
# earlyStop=tf.keras.callbacks.EarlyStopping(
#     monitor='val_mean_squared_error', #'val_loss', #'mean_absolute_error',
#     patience=earlyStop_pati,
#     mode='min', #'auto',
#     restore_best_weights=True
#     # min_delta=0, verbose=0, baseline=None,
# )
# history = model.fit(train_x, train_y, batch_size=64,epochs=epochs_num, validation_data=(val_x, val_y), callbacks=[model_cbk, model_mckp, earlyStop])

# #畫出訓練的 epoch curve
# hist = pd.DataFrame(history.history)
# # print(hist.columns)
# hist['epoch'] = history.epoch
# def plot_loss(history):
#     plt.figure()
#     plt.plot(history.history['loss'], label='loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.title('Epoch_curve_loss')
#     plt.savefig('epoch_curve.png')

#     plt.figure()
#     # plt.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
#     # plt.plot(history.history['val_mean_absolute_error'], label='val_mean_absolute_error')
#     plt.plot(history.history['val_mean_squared_error'], label='val_mean_squared_error')
#     plt.plot(history.history['val_mean_squared_error'], label='val_mean_squared_error')
#     plt.legend()
#     plt.grid(True)
#     plt.xlabel('Epoch')
#     # plt.ylabel('MAE')
#     # plt.title('Epoch_curve_MAE')
#     # plt.savefig('epoch_curve_MAE.png')
#     plt.ylabel('MSE') 
#     plt.title('Epoch_curve_MSE') 
#     plt.savefig('epoch_curve_MSE.png') 
# plot_loss(history)


# # model = keras.models.load_model('lab1-logs/models/Best-model.h5')
# y_pred = model.predict(test_x) #*std['price']+mean['price']
# print(len(y_pred))
# loss, mse = model.evaluate(test_x, test_y, verbose=2)
# print("Testing set Mean squared Error: {:.3f}".format(mse))

# # print(test_y.shape,np.squeeze(y_pred).shape)
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# print('MSE(testing): %.3f' %(mean_squared_error(test_y, np.squeeze(y_pred))))