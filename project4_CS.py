# other models/ feature selection/ multiple linear regression

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import linalg as LA
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import functions as func # import file containing functions created in previous labs
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# read data
df = pd.read_csv('data/Metro_Traffic_clean.csv', parse_dates=[0], index_col=0)
# df.head()
# df.info()
traffic = df['traffic_volume']

# split data (training - 80% ; test - 20%)
X = df[['temp', 'rain', 'snow', 'clouds', 'weather_short', 'weather_long', 'holiday', 'weather_short2', 'weather_long2']]
X_train, X_test, Y_train, Y_test = train_test_split(X, traffic, shuffle=False, test_size=0.2)

# Holt-Winter Seasonal Method
traffic_hw = ets.ExponentialSmoothing(Y_train, trend=None, damped=False, seasonal='additive', seasonal_periods=24).fit()
traffic_hw2 = traffic_hw.forecast(steps = len(Y_test))
traffic_hw2 = pd.DataFrame(traffic_hw2).set_index(Y_test.index)

plt.figure()
plt.plot(Y_train, label='training', color='green')
plt.plot(Y_test, label='test', color='orange')
plt.plot(traffic_hw2, label='forecast', color='red')
plt.title("Traffic Volume Forecast (Holt-Winter Seasonal Method)")
plt.legend()
plt.xlabel('time (t)')
plt.xticks(rotation=13)
plt.ylabel('Volume')
plt.show()

plt.figure()
plt.plot(Y_test[-120:], label='test', color='orange')
plt.plot(traffic_hw2[-120:], label='forecast', color='red')
plt.title('Traffic Volume Forecast (Holt Winter Method, last 120 samples)')
plt.legend()
plt.xlabel('time (t)')
plt.ylabel('Volume')
plt.show()

traffic_hw2 = traffic_hw2.squeeze() # change dataframe to series
traffic_hw_mse = func.get_MSE(Y_test, traffic_hw2)
print(f"MSE of Holt-Winter method: {traffic_hw_mse:.2f}")

# forecast errors
traffic_hw_err_list = func.forecast_errors(Y_test, traffic_hw2)
traffic_hw_acf = func.get_autocorrfunc(traffic_hw_err_list, 50)

x = np.linspace(-50, 50, 101)
m = 1.96 / (np.sqrt(len(Y_test)))
plt.stem(x, traffic_hw_acf, use_line_collection=True)
plt.xlabel("Lags")
plt.axhspan(-m, m, alpha=0.25, color='blue')
plt.ylabel("Magnitude")
plt.title("Traffic Forecast Errors ACF (Holt-Winter Seasonal Method)")
plt.show()

Q_hw = func.get_Q(len(Y_test), traffic_hw_acf, 50)
print(f"Q value of Holt_Winter Seasonal method: {Q_hw}")

# feature selection
X = df[['temp', 'rain', 'snow', 'clouds', 'holiday', 'weather_short2', 'weather_long2']]
X = sm.add_constant(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, traffic, shuffle=False, test_size=0.2)

# SVD analysis
X = X_train.values
H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print(f"Singular Values: {d}") # last number sort of close to 0 (around 2)

# Condition number
print(f"Condition Number: {LA.cond(X)}") # high number (around 30672)

# find coefficients using OSL
model = sm.OLS(Y_train, X_train).fit()
print(model.summary())

# remove 'snow'
X_train.drop(['snow'], axis=1, inplace=True)
model = sm.OLS(Y_train, X_train).fit()
print(model.summary())

# remove 'rain'
X_train.drop(['rain'], axis=1, inplace=True)
model = sm.OLS(Y_train, X_train).fit()
print(model.summary())

# remove 'weather_long2'
X_train.drop(['weather_long2'], axis=1, inplace=True)
model = sm.OLS(Y_train, X_train).fit()
print(model.summary())

# prediction
X_test = X_test[['const', 'temp', 'clouds', 'holiday', 'weather_short2']]
ypred = model.predict(X_test)

# plot
plt.figure()
plt.plot(Y_train, label='Train', color='green')
plt.plot(Y_test, label='Test', color='orange')
plt.plot(ypred, label='Prediction', color='red')
plt.legend()
plt.xlabel('Sample')
plt.xticks(rotation=10)
plt.ylabel('Volume')
plt.title('Traffic Volume Prediction')
plt.show()

plt.figure()
plt.plot(Y_test[-200:], label='Test', color='orange')
plt.plot(ypred[-200:], label='Prediction', color='red')
plt.legend()
plt.xlabel('Sample')
plt.xticks(rotation=10)
plt.ylabel('Volume')
plt.title('Traffic Volume Prediction (last 200 samples)')
plt.show()

# prediction error
ypred2 = model.predict(X_train)
ypred2 = ypred2.tolist()
y_train = Y_train.values.tolist()
pred_error = []

for i in range(len(y_train)):
      error = y_train[i] - ypred2[i]
      pred_error.append(error)

pred_acf = func.get_autocorrfunc(pred_error, 50)

N = len(X_train)
m = 1.96 / (np.sqrt(N))
x = np.linspace(-50, 50, 101)
plt.stem(x, pred_acf)
plt.axhspan(-m, m, alpha=0.25, color='blue')
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.title("Prediction Error ACF")
plt.show()

# forecast error
ypred = ypred.tolist()
y_test = Y_test.values.tolist()
fore_error = []

for i in range(len(y_test)):
      error = y_test[i] - ypred[i]
      fore_error.append(error)

fore_acf = func.get_autocorrfunc(fore_error, 20) # say # of lags = 20

x = np.linspace(-20, 20, 41)
plt.stem(x, fore_acf)
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.axhspan(-m, m, alpha=0.25, color='blue')
plt.title("Forecast Error ACF")
plt.show()

# variance of prediction error
T = len(Y_train)
k = 4

errorsq = 0
for item in pred_error:
      errorsq += item**2

pred_var = errorsq / (T - k - 1)
print(f"Estimated variance of prediction error: {pred_var:.2f}")
print(f"Estimated standard deviation of prediction error: {np.sqrt(pred_var):.2f}")

# variance of forecast error
T = len(Y_test)
k = 4

errorsq = 0
for item in fore_error:
      errorsq += item**2

fore_var = errorsq / (T - k - 1)
print(f"Estimated variance of forecast error: {fore_var:.2f}")
print(f"Estimated standard deviation of forecast error: {np.sqrt(fore_var):.2f}")

# mean
print(f"Estimated mean of prediction error: {np.mean(pred_error):.2f}")
print(f"Estimated mean of forecast error: {np.mean(fore_error):.2f}")

# t-test
print(model.t_test(np.eye(len(model.params))))

# F-test
print(f"The F-value from the F-test is {(model.fvalue):.2f}, and the p-value is {(model.f_pvalue):.2f}.")

# Q value
y_test = np.asarray(y_test)
ypred = np.asarray(ypred)

res = y_test - ypred
re = func.get_autocorrfunc(res, 50)
Q_reg = len(y_test) * np.sum(np.square(re[50:]))
print(f"The Q value is {Q_reg}.")

# MSE
ols_mse = func.get_MSE(Y_test, ypred)
print(f"MSE of multiple linear regression: {ols_mse:.2f}")