#勇敢牛牛，不怕困难！！！
#Sun Jul  4 14:08:16 2021
import evaluate
import pandas as pd
import numpy as np
import matplotlib.pylab as plt  
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs as nd
from sklearn.metrics import r2_score
#%% acf pacf
dta = pd.read_csv('EC-W(ND).csv', names=['DATA'], header=0)
x = dta.DATA
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(x); axes[0].set_title('0')
axes[1].plot(x.diff()); axes[1].set_title('1')
axes[2].plot(x.diff().diff()); axes[2].set_title('2')
plt.show()
#%%
plot_pacf(dta)
plot_acf(dta.DATA.squeeze(), lags = 54)
plt.show()
#%% test
print(nd(x, test='adf'))
print(nd(x, test='kpss'))
print(nd(x, test='pp'))  
#%% model
dta2 = pd.read_csv('EC-W(D).csv', index_col=0, parse_dates=[0])
y = dta2['1965':'2019']
modela = ARIMA(y, order = (2,1,2))
ftmodela = modela.fit(disp= 0)
print(ftmodela.summary())
ftmodela.plot_predict(dynamic=False)
plt.show()
#%% train
train = dta2['1965':'2000']
test = dta2['2000':'2019']

modelb = ARIMA(train, order = (2,1,3))
ftmodelb = modelb.fit(disp= -1)
fc, se, conf = ftmodelb.forecast(20, alpha=0.05)

fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

plt.grid(alpha =.6, linestyle = "-") 
plt.figure(figsize=(17,11), dpi=100)
plt.plot(train, label='train')
plt.plot(test, label='test')
plt.plot(fc_series, label='fc')
plt.show()
#%% accuracy
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) 
    mae = np.mean(np.abs(forecast - actual))    
    rmse = np.mean((forecast - actual)**2)**.5                   
    return({'mape':mape,
            'mae': mae, 
            'rmse':rmse})

print(forecast_accuracy(fc, test.values))
#%%
r2 = r2_score(fc, test.values)
print("r^2:", r2)
