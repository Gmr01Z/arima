#勇敢牛牛，不怕困难！！！
#Sun Oct 31 20:01:58 2021
import pandas as pd
import matplotlib.pylab as plt 
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs as nd
from sklearn.metrics import r2_score
#%% read in data
df = pd.read_csv("EC_ARIMA3.csv",  index_col="Yr", parse_dates=[0])
#%% order stationary
dta = df.EC
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(dta); axes[0].set_title('0')
axes[1].plot(dta.diff()); axes[1].set_title('1')
axes[2].plot(dta.diff().diff()); axes[2].set_title('2')
plt.show()
# check
dtadiff1 = dta.diff().dropna()
dtadiff2 = dtadiff1.diff().dropna()
dtadiff3 = dtadiff2.diff().dropna()
def adf_test(dataset):
     dftest = adfuller(dataset, autolag = 'AIC')
     print("1. ADF : ",dftest[0])
     print("2. P-Value : ", dftest[1])
     print("3. Num Of Lags : ", dftest[2])
     print("4. Num Of Observations Used For ADF Regression:", dftest[3])
     print("5. Critical Values :")
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)
         
adf_test(dta)
adf_test(dtadiff1)
adf_test(dtadiff2)
adf_test(dtadiff3)
#%% acf&pacf
plot_acf(dta)
plot_pacf(dta)
stepwise_fit = auto_arima(dta, trace=True, suppress_warnings=True)
#%%
train = df['1965':'2000']
test = df['2000':'2019']
modela = ARIMA(df, order = (2,2,2))
ftmodela = modela.fit(disp= 0)
print(ftmodela.summary())
ftmodela.plot_predict(2,80)
plt.show()
