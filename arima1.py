#勇敢牛牛，不怕困难！！！
#Thu Jul  1 16:07:47 2021

import pandas as pd
import numpy as np
import matplotlib.pylab as plt  
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
import sklearn.metrics as sms

data = pd.read_csv('df(K1).csv', index_col=0, parse_dates=[0])
data.columns = ['P']
plt.plot(data, 'o', color='black')
#%%
data_train = data['1965':'2019']

model = ARIMA(data_train, order=(0,2,1))
result = model.fit()

fc = result.predict('2000', '2030', dynamic = True, typ = 'levels')
#%%
print(fc)
plt.figure(figsize=(17,12))
plt.plot(data_train)
plt.plot(fc)
plt.grid(alpha =.6, linestyle = "-") 
#%%

