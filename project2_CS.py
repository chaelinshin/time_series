
# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import functions as func # import file containing functions created in previous labs

# read data
df = pd.read_csv('data/Metro_Traffic_clean.csv', parse_dates=[0], index_col=0)
# df.head()
# df.info()
traffic = df['traffic_volume']

# plot traffic
plt.figure()
plt.plot(traffic, label='Traffic Volume', color='green')
plt.xlabel("Time")
plt.xticks(fontsize=8, rotation=13)
plt.ylabel("Volume")
plt.title("Traffic Volume Over Time")
plt.show()

# too many data --> plot a slice of data too see clearly
plt.figure()
plt.plot(traffic[:200], label='Traffic Volume', color='green')
plt.xlabel("Time")
plt.xticks(fontsize=8, rotation=13)
plt.ylabel("Volume")
plt.title("Traffic Volume Over Time (200 samples)")
plt.show()

# ACF/PACF
lags = 150
plt.figure()
plt.subplot(211)
plot_acf(traffic, ax=plt.gca(), lags=lags)
plt.subplot(212)
plot_pacf(traffic, ax=plt.gca(), lags=lags)
plt.tight_layout(pad=1)
plt.show()

# import functions as func
traffic_acf = func.get_autocorrfunc(traffic, 40)

x = np.linspace(-40, 40, 81)
m = 1.96 / (np.sqrt(len(traffic)))
plt.stem(x, traffic_acf, use_line_collection=True)
plt.xlabel("Lags")
plt.axhspan(-m, m, alpha=0.25, color='blue')
plt.ylabel("Magnitude")
plt.title("Traffic ACF")
plt.show()

# ADF Test
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
       print('\t%s: %.3f' % (key, value))

print("ADF test result for traffic volume: ")
ADF_Cal(traffic)

# mean/variance over time
traffic_avg = []
traffic_var = []

for i in range(1, 5001):
    traffic_avg.append(np.mean(traffic.head(i)))
    traffic_var.append(np.var(traffic.head(i)))

plt.figure()
plt.plot(traffic_avg, label='Mean', color='green')
plt.xlabel("Samples")
plt.ylabel("Mean")
plt.title("Mean Versus Time")
plt.show()

plt.figure()
plt.plot(traffic_var, label='Variance', color='orange')
plt.xlabel("Samples")
plt.ylabel("Variance")
plt.title("Variance Versus Time")
plt.show()

# STL decomposition method
STL = STL(traffic, period=24)
res = STL.fit()

T = res.trend # trend
S = res.seasonal # seasonal
R = res.resid # remainder

plt.figure()
plt.plot(T[:500], label='Trend')
plt.plot(S[:500], label='Seasonal')
plt.plot(R[:500], label='Remainder')
plt.xticks(fontsize=10, rotation=11)
plt.xlabel("Date")
plt.ylabel("Values")
plt.title("STL")
plt.legend()
plt.show()

# seasonally adjusted vs. original
adj_season = traffic - S

plt.figure()
plt.plot(traffic[:500], label='Original', color='green')
plt.plot(adj_season[:500], label="Adjusted Seasonal", color='orange')
plt.xticks(fontsize=10, rotation=11)
plt.xlabel("Date")
plt.ylabel("Values")
plt.title("Seasonally Adjusted vs. Original")
plt.legend()
plt.show()

# detrended
detrended = traffic - T

plt.figure()
plt.plot(traffic[:500], label='Original', color='green')
plt.plot(detrended[:500], label="Detrended", color='orange')
plt.xticks(fontsize=10, rotation=11)
plt.xlabel("Date")
plt.ylabel("Values")
plt.title("Detrended vs. Original")
plt.legend()
plt.show()

# strength of trend
Ft = max(0, (1 - (np.var(R) / np.var(adj_season))))
print(f'The strength of trend for this dataset is {Ft:.2f}.')

# strength of seasonality
Fs = max(0, (1 - (np.var(R) / np.var(detrended))))
print(f'The strength of seasonality for this dataset is {Fs:.2f}.')

# try differencing
def difference(dataset, interval=1):
   diff = []
   for i in range(interval, len(dataset)):
      value = dataset[i] - dataset[i - interval]
      diff.append(value)
   return diff

diff1 = difference(traffic, 1)

lags = 100
plt.figure()
plt.subplot(211)
plot_acf(diff1, ax=plt.gca(), lags=lags)
plt.subplot(212)
plot_pacf(diff1, ax=plt.gca(), lags=lags)
plt.tight_layout(pad=1)
plt.show()

diff2 = difference(diff1,24)

lags = 200
plt.figure()
plt.subplot(211)
plot_acf(diff2, ax=plt.gca(), lags=lags)
plt.subplot(212)
plot_pacf(diff2, ax=plt.gca(), lags=lags)
plt.tight_layout(pad=1)
plt.show()

plt.figure()
plt.plot(traffic.values[:500], label='original', color='green')
plt.plot(diff2[:500], label='after differencing', color='orange')
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Values")
plt.title("Original vs. Differencing")
plt.show()

# ADF test
print("ADF test result for traffic volume after differencing: ")
ADF_Cal(diff2)
