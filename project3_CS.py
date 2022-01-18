# base models

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

# Average method
def average_predict(list):

    forecast = 0
    for item in list:
        forecast += item
    forecast = forecast / len(list)

    return forecast

l1 = len(Y_train)
l2 = len(Y_test)

traffic_ave = [average_predict(Y_train)] * l2
traffic_ave2 = [None]*l1 + traffic_ave
Y_train2 = Y_train + [None]*l2
Y_test2 = [None]*l1 + Y_test

plt.figure()
plt.plot(Y_train2, label='training', color='green')
plt.plot(Y_test2, label='test', color='orange')
plt.plot(traffic_ave2, label='forecast', color='red')
plt.title('Traffic Volume Forecast (Average Method)')
plt.legend()
plt.xlabel('time (t)')
plt.ylabel('Volume')
plt.show()

# only look at test set and forecast
plt.figure()
plt.plot(Y_test2[-500:], label='test', color='orange')
plt.plot(traffic_ave2[-500:], label='forecast', color='red')
plt.title('Traffic Volume Forecast (Average Method, last 500 samples)')
plt.legend()
plt.xlabel('time (t)')
plt.ylabel('Volume')
plt.show()

# MSE
traffic_ave_mse = func.get_MSE(Y_test, traffic_ave)
print(f"MSE of average method: {traffic_ave_mse:.2f}")

# forecast errors and ACF
traffic_ave_err_list = func.forecast_errors(Y_test, traffic_ave)
traffic_ave_acf = func.get_autocorrfunc(traffic_ave_err_list, 50)

x = np.linspace(-50, 50, 101)
m = 1.96 / (np.sqrt(len(Y_test)))
plt.stem(x, traffic_ave_acf, use_line_collection=True)
plt.xlabel("Lags")
plt.axhspan(-m, m, alpha=0.25, color='blue')
plt.ylabel("Magnitude")
plt.title("Traffic Forecast Errors ACF (Average Method)")
plt.show()

# Q value
Q_ave = func.get_Q(len(Y_test), traffic_ave_acf, 50)
print(f"Q value of average method: {Q_ave}")

# Naive method
traffic_naive = [Y_train[len(Y_train) - 1]] * l2
traffic_naive2 = [None]*l1 + traffic_naive

plt.figure()
plt.plot(Y_train2, label='training', color='green')
plt.plot(Y_test2, label='test', color='orange')
plt.plot(traffic_naive2, label='forecast', color='red')
plt.title('Traffic Volume Forecast (Naive Method)')
plt.legend()
plt.xlabel('time (t)')
plt.ylabel('Volume')
plt.show()

# only look at test set and forecast
plt.figure()
plt.plot(Y_test2[-500:], label='test', color='orange')
plt.plot(traffic_naive2[-500:], label='forecast', color='red')
plt.title('Traffic Volume Forecast (Naive Method, last 500 samples)')
plt.legend()
plt.xlabel('time (t)')
plt.ylabel('Volume')
plt.show()

# MSE
traffic_naive_mse = func.get_MSE(Y_test, traffic_naive)
print(f"MSE of naive method: {traffic_naive_mse:.2f}")

# forecast errors
traffic_naive_err_list = func.forecast_errors(Y_test, traffic_naive)
traffic_naive_acf = func.get_autocorrfunc(traffic_naive_err_list, 50)

plt.stem(x, traffic_naive_acf, use_line_collection=True)
plt.xlabel("Lags")
plt.axhspan(-m, m, alpha=0.25, color='blue')
plt.ylabel("Magnitude")
plt.title("Traffic Forecast Errors ACF (Naive Method)")
plt.show()

# Q value
Q_naive = func.get_Q(len(Y_test), traffic_naive_acf, 50)
print(f"Q value of naive method: {Q_naive}")

# Drift method
def drift_predict(list, length):
    # list --> train data
    # length --> length of test data

    predicts = []
    slope = (list[len(list) - 1] - list[0]) / (len(list) - 1)

    for i in range(length):
        predict = (slope * (len(list) + i)) + (list[0] - slope)
        predicts.append(predict)

    return predicts

traffic_drift = drift_predict(Y_train, l2)
traffic_drift2 = [None]*l1 + traffic_drift

plt.figure()
plt.plot(Y_train2, label='training', color='green')
plt.plot(Y_test2, label='test', color='orange')
plt.plot(traffic_drift2, label='forecast', color='red')
plt.title('Traffic Volume Forecast (Drift Method)')
plt.legend()
plt.xlabel('time (t)')
plt.ylabel('Volume')
plt.show()

# only look at test set and forecast
plt.figure()
plt.plot(Y_test2[-2000:], label='test', color='orange')
plt.plot(traffic_drift2[-2000:], label='forecast', color='red')
plt.title('Traffic Volume Forecast (Drift Method, last 2000 samples)')
plt.legend()
plt.xlabel('time (t)')
plt.ylabel('Volume')
plt.show()

# MSE
traffic_drift_mse = func.get_MSE(Y_test, traffic_drift)
print(f"MSE of drift method: {traffic_drift_mse:.2f}")

# forecast errors
traffic_drift_err_list = func.forecast_errors(Y_test, traffic_drift)
traffic_drift_acf = func.get_autocorrfunc(traffic_drift_err_list, 50)

plt.stem(x, traffic_drift_acf, use_line_collection=True)
plt.xlabel("Lags")
plt.axhspan(-m, m, alpha=0.25, color='blue')
plt.ylabel("Magnitude")
plt.title("Traffic Forecast Errors ACF (Drift Method)")
plt.show()

# Q value
Q_drift = func.get_Q(len(Y_test), traffic_drift_acf, 50)
print(f"Q value of drift method: {Q_drift}")

# SES Method (alpha=0.5)
def ses_predict(list, a=0.5):
    # list --> train data
    # alpha --> alpha value
    predict = list[0]

    for i in range(len(list)):
        predict = (a * list[i]) + ((1 - a) * predict)

    return predict

traffic_ses = [ses_predict(Y_train)] * l2
traffic_ses2 = [None]*l1 + traffic_ses

plt.figure()
plt.plot(Y_train2, label='training', color='green')
plt.plot(Y_test2, label='test', color='orange')
plt.plot(traffic_ses2, label='forecast', color='red')
plt.title('Traffic Volume Forecast (SES Method, alpha=0.5)')
plt.legend()
plt.xlabel('time (t)')
plt.ylabel('Volume')
plt.show()

# only look at test set and forecast
plt.figure()
plt.plot(Y_test2[-500:], label='test', color='orange')
plt.plot(traffic_ses2[-500:], label='forecast', color='red')
plt.title('Traffic Volume Forecast (SES Method, alpha=0.5, last 500 samples)')
plt.legend()
plt.xlabel('time (t)')
plt.ylabel('Volume')
plt.show()

# MSE
traffic_ses_mse = func.get_MSE(Y_test, traffic_ses)
print(f"MSE of SES method (alpha=0.5): {traffic_ses_mse:.2f}")

# forecast errors
traffic_ses_err_list = func.forecast_errors(Y_test, traffic_ses)
traffic_ses_acf = func.get_autocorrfunc(traffic_ses_err_list, 50)

plt.stem(x, traffic_ses_acf, use_line_collection=True)
plt.xlabel("Lags")
plt.axhspan(-m, m, alpha=0.25, color='blue')
plt.ylabel("Magnitude")
plt.title("Traffic Forecast Errors ACF (SES Method, alpha=0.5)")
plt.show()

Q_ses = func.get_Q(len(Y_test), traffic_ses_acf, 50)
print(f"Q value of SES method: {Q_ses}")

# table
fig, ax = plt.subplots()
table_data = [
    ['Method', 'MSE', 'Q value'],
    ['Average', round(traffic_ave_mse, 2), round(Q_ave, 2)],
    ['Naive', round(traffic_naive_mse, 2), round(Q_naive, 2)],
    ['Drift', round(traffic_drift_mse, 2), round(Q_drift, 2)],
    ['SES', round(traffic_ses_mse, 2), round(Q_ses, 2)]
]

table = ax.table(cellText = table_data, loc='center')

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2,2)
ax.axis('off')

plt.show()







