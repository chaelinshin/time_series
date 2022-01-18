import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
import functions as func # import file containing functions created in previous labs

# read data
df = pd.read_csv('data/Metro_Traffic_clean.csv', parse_dates=[0], index_col=0)
# df.head()
# df.info()
traffic = df['traffic_volume']

def difference(dataset, interval=1):
   diff = []
   for i in range(interval, len(dataset)):
      value = dataset[i] - dataset[i - interval]
      diff.append(value)
   return diff

diff1 = difference(traffic, 1)
diff2 = difference(diff1, 24)

# split data (training - 80% ; test - 20%)
X = df[['temp', 'rain', 'snow', 'clouds', 'weather_short', 'weather_long', 'holiday', 'weather_short2', 'weather_long2']]
X = X[25:] # to match the number of samples with diff2
X_train, X_test, Y_train, Y_test = train_test_split(X, diff2, shuffle=False, test_size=0.2)

# order determination
# ACF/PACF plot
lags = 240
plt.figure()
plt.subplot(211)
plot_acf(diff2, ax=plt.gca(), lags=lags)
plt.subplot(212)
plot_pacf(diff2, ax=plt.gca(), lags=lags)
plt.tight_layout(pad=1)
plt.show()

# GPAC table
diff2_acf = func.get_autocorrfunc(diff2, 50)
table = func.cal_GPAC(diff2_acf, 13, 13)
print(table)

# first SARIMA model
ar_order = (6, 1, 0)
ma_order = (0, 1, 1, 24)
model1 = sm.tsa.statespace.SARIMAX(Y_train, order=ar_order, seasonal_order=ma_order).fit(max_iter=50, method='powell')
model1.summary()
model1.conf_int() # confidence interval
model1.cov_params_approx # covariance

# make prediction
model_train1 = model1.predict(start=0, end=len(Y_train)-1)
model_hat1 = model1.predict(start=0, end=len(Y_test)-1)

plt.figure()
plt.plot(Y_test[:200], label='test', color='orange')
plt.plot(model_hat1[:200], label='predicted', color='red')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.title('First SARIMA Model Prediction (first 200 samples)')
plt.show()

plt.figure()
plt.plot(Y_train[:500], label='train', color='orange')
plt.plot(model_train1[:500], label='predicted', color='red')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.title('First SARIMA Model Prediction (first 500 samples)')
plt.show()

# errors
fore = Y_test - model_hat1
re = func.get_autocorrfunc(fore, 50)
res = Y_train - model_train1
re2 = func.get_autocorrfunc(res, 50)

# Q value
Q = len(Y_test) * np.sum(np.square(re[50:]))
print(f"The Q value is {Q}.")

# mean of errors
print(f"Mean of residuals: {np.mean(res):.2f}")
print(f"Mean of forecast errors: {np.mean(fore):.2f}")

# variance and standard deviation of errors
print(f"Variance of residuals: {np.var(res):.2f}")
print(f"Standard deviation of residuals: {np.sqrt(np.var(res)):.2f}")
print(f"Variance of forecast errors: {np.var(fore):.2f}")
print(f"Standard deviation of forecast errors: {np.sqrt(np.var(fore)):.2f}")

# ACF of errors
x = np.linspace(-50, 50, 101)
m = 1.96 / (np.sqrt(len(Y_test)))
plt.stem(x, re, use_line_collection=True)
plt.xlabel("Lags")
plt.axhspan(-m, m, alpha=0.25, color='blue')
plt.ylabel("Magnitude")
plt.title("Errors ACF (First SARIMA Model)")
plt.show()

# plot of errors
plt.figure()
plt.plot(res, color="green")
plt.title("Plot of Errors")
plt.xlabel("Samples")
plt.ylabel("Errors")
plt.show()

# histogram of errors
plt.hist(res, bins=80, color='green')
plt.title("Histogram of Errors")
plt.xlabel("Errors")
plt.ylabel("Frequency")
plt.show()

# second SARIMA model
ar_order = (6, 1, 0)
ma_order = (0, 1, 2, 24)
model2 = sm.tsa.statespace.SARIMAX(Y_train, order=ar_order, seasonal_order=ma_order).fit(max_iter=50, method='powell')
model2.summary()
model2.conf_int() # confidence interval
model2.cov_params_approx # covariance
print(f"AR roots: {model2.arroots}")
print(f"MA roots: {model2.maroots}")

# make prediction (one-step)
model_train2 = model2.predict(start=0, end=len(Y_train)-1)
model_hat2 = model2.predict(start=0, end=len(Y_test)-1)

plt.figure()
plt.plot(Y_test[:200], label='test', color='orange')
plt.plot(model_hat2[:200], label='predicted', color='red')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.title('Second SARIMA Model Prediction (first 200 samples)')
plt.show()

plt.figure()
plt.plot(Y_train[:500], label='train', color='orange')
plt.plot(model_train2[:500], label='predicted', color='red')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.title('Second SARIMA Model Prediction (first 500 samples)')
plt.show()

# errors
fore = Y_test - model_hat2
re = func.get_autocorrfunc(fore, 50)
res = Y_train - model_train2
re2 = func.get_autocorrfunc(res, 50)

# Q value
Q = len(Y_test) * np.sum(np.square(re[50:]))
print(f"The Q value is {Q}.")

# mean of errors
print(f"Mean of residuals: {np.mean(res):.2f}")
print(f"Mean of forecast errors: {np.mean(fore):.2f}")

# variance and standard deviation of errors
print(f"Variance of residuals: {np.var(res):.2f}")
print(f"Standard deviation of residuals: {np.sqrt(np.var(res)):.2f}")
print(f"Variance of forecast errors: {np.var(fore):.2f}")
print(f"Standard deviation of forecast errors: {np.sqrt(np.var(fore)):.2f}")

# ACF of errors
x = np.linspace(-50, 50, 101)
m = 1.96 / (np.sqrt(len(Y_test)))
plt.stem(x, re, use_line_collection=True)
plt.xlabel("Lags")
plt.axhspan(-m, m, alpha=0.25, color='blue')
plt.ylabel("Magnitude")
plt.title("Errors ACF (Second SARIMA Model)")
plt.show()

# plot of errors
plt.figure()
plt.plot(res, color="green")
plt.title("Plot of Errors")
plt.xlabel("Samples")
plt.ylabel("Errors")
plt.show()

# histogram of errors
plt.hist(res, bins=80, color='green')
plt.title("Histogram of Errors")
plt.xlabel("Errors")
plt.ylabel("Frequency")
plt.show()

# forecast
steps = len(Y_test) # number of steps we would like to forecast
forecast = model2.forecast(steps)

plt.figure()
plt.plot(Y_test, label='test', color='orange')
plt.plot(forecast, label='forecast', color='red')
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Values")
plt.title("Test vs. Forecast")
plt.show()

steps = 30 # number of steps we would like to forecast
forecast = model2.forecast(steps)

plt.figure()
plt.plot(Y_test[:30], label='test', color='orange')
plt.plot(forecast, label='forecast', color='red')
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Values")
plt.title("Test vs. Forecast")
plt.show()
