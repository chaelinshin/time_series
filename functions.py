import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import signal
import statistics
import math

def correlation_coefficient_cal(x, y):
    # x and y should be lists

    x_mean, y_mean = statistics.mean(x), statistics.mean(y)

    numerator, x_denominator, y_denominator = 0, 0, 0

    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        x_denominator += (x[i] - x_mean)**2
        y_denominator += (y[i] - y_mean)**2

    r = numerator / (math.sqrt(x_denominator) * math.sqrt(y_denominator))

    return r

def get_autocorrfunc(x, lag):
    # x must be a list
    # parameter lag is the number of lags
    import numpy as np

    r_list = []
    mean_ = np.mean(x)
    denominator, numerator = 0, 0
    for item in x:
        denominator += (item - mean_)**2

    # i = number of lags
    # length = T
    length = len(x)

    for i in range(lag+1):
        for j in range(i, length):
            numerator += ((x[j] - mean_) * (x[j-i] - mean_))
        r_list.append(numerator / denominator)
        numerator = 0

    r_list2 = r_list[::-1]
    r_final_list = r_list2[:-1] + r_list

    return r_final_list

def getARMA():
    # construct ARMA process and generate dataset
    import numpy as np

    T = int(input("Enter number of data samples: "))
    mean = int(input("Enter mean of white noise: "))
    var = int(input("Enter variance of white noise: "))
    na = int(input("Enter AR order: "))
    nb = int(input("Enter MA order: "))

    # examples for input an or bn --> 0.5 1 (make sure to put a space between each value)
    an = [float(x) for x in
          input("Enter the coefficients of AR (enter multiple values and put a space between each value): ").split()]
    bn = [float(x) for x in
          input("Enter the coefficients of MA (enter multiple values and put a space between each value): ").split()]

    arparams = np.array(an)
    maparams = np.array(bn)

    ar = np.r_[1, arparams]
    ma = np.r_[1, maparams]

    arma_process = sm.tsa.ArmaProcess(ar, ma)

    # generate ARMA process dataset
    y_mean = mean * ((1 + np.sum(bn)) / (1 + np.sum(an)))
    y = arma_process.generate_sample(T, scale=np.sqrt(var)) + y_mean

    return arma_process, y

# Question 2 - code for GPAC table

def cal_GPAC(ry, k, j):
    # parameter ry is the list of estimated autocorrelation (ACF) using ACF function from Lab 4
    # parameters k and j should be integers
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    list1 = [] # will be list of lists

    for num_k in range(1, k): # from 1 to k-1
        list2 = []
        for num_j in range(0, j): # from 0 to j-1
            # denominator - kxk matrix
            d = []
            first = (len(ry) // 2) - num_j

            for idx in range(first, -1, -1):
                slice = ry[idx: (idx + num_k)]
                d.append(slice)
                if len(d) >= num_k:
                    break

            # numerator - kxk matrix
            n = np.copy(d)
            for i in range(len(n)):
                idx = first - i
                n[i][len(n) - 1] = ry[idx - 1]

            # get determinants
            det1 = np.linalg.det(d) # denominator
            det2 = np.linalg.det(n) # numerator

            value = det2 / det1
            list2.append(round(value, 2))

        list1.append(list2)

    dict = {}
    for i in range(1, len(list1) + 1):
        dict[i] = list1[i - 1]

    table = pd.DataFrame(dict)

    ax = sns.heatmap(table, annot=True, cmap='Greens', square=True)
    bottom, top = ax.get_ylim()
    ax.set_title("Generalized Partial Autocorrelation Function (GPAC)")
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
    plt.show()

    return table

def MovingAverage(data):
    # data should be in a form of a list
    # this function will implement moving average of order m (user input)
    # and will return a list of estimates

    m = int(input("What is the order of moving average (that is larger than 0)? "))

    if m % 2 == 1: # if m is an odd number
        final_list = []
        k = int((m - 1) / 2)
        for i in range(k, (len(data) - k)):
            sum_ = 0
            for j in range((i - k), (i + k + 1)):
                sum_ += data[j]
            average = sum_ / m
            final_list.append(average)

    elif m % 2 == 0: # if m is an even number
        m2 = int(input("What is the second order of moving average (that is even)? "))
        if (m2 == 1) or (m2 == 2):
            print("Invalid order of moving average.")
            return None

        else:
            list1, final_list = [], []
            for i in range(0, (len(data) - m + 1)):
                sum_ = 0
                for j in range(i, i + m):
                    sum_ += data[j]
                average = sum_ / m
                list1.append(average)

            for i in range(0, (len(list1) - m2 + 1)):
                sum_ = 0
                for j in range(i, i + m2):
                    sum_ += list1[j]
                average = sum_ / m2
                final_list.append(average)

    num_ = (len(data) - len(final_list)) // 2
    final_list = ([None] * num_) + final_list + ([None] * num_)

    return final_list


def detrend(original, estimate):

    final_list = []
    for i in range(len(original)):
        if estimate[i] == None:
            final_list.append(None)
        else:
            value = original[i] / estimate[i]
            final_list.append(value)

    return final_list

def getSARIMA():

    T = int(input("Enter the number of samples: "))
    mean = float(input("Enter the mean of white noise: "))
    var = float(input("Enter the variance of white noise: "))
    na = int(input("Enter AR order: "))
    nb = int(input("Enter MA order: "))

    an = [float(x) for x in
          input("Enter the coefficients of AR (enter multiple values and put a space between each value): ").split()]
    # make sure number of coefficients is the same as number of coefficients of MA

    bn = [float(x) for x in
          input("Enter the coefficients of MA (enter multiple values and put a space between each value): ").split()]

    arparams = np.array(an)
    maparams = np.array(bn)

    den = np.r_[1, arparams] # denominator
    num = np.r_[1, maparams] # numerator
    e = np.random.normal(mean, np.sqrt(var), size=T)

    # simulate SARIMA model
    sys = (num, den, 1)
    _, y = signal.dlsim(sys, e)

    return y

def difference(dataset, interval=1):
   diff = []
   for i in range(interval, len(dataset)):
      value = dataset[i] - dataset[i - interval]
      diff.append(value)
   return diff

def lm_step1(theta, y):

    if nb == 0:
        maparams = [0] * na
        arparams = theta[:,0].tolist()

    elif na == 0:
        maparams = theta[:,0].tolist()
        arparams = [0] * nb

    else:
        maparams = theta[-nb:,0].tolist() + [0.] * (na - nb)
        arparams = theta[:na,0].tolist()

    num = np.r_[1, maparams]
    den = np.r_[1, arparams]
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)

    # initial SSE
    SSE_theta = np.matmul(e.T, e)

    # xi and X
    X = np.zeros((N, 0))
    delta = 10 ** (-6)

    theta_ = theta.copy()
    for i in range(n):

        theta_[i,0] += delta

        if nb == 0:
            maparams = [0] * na
            arparams = theta_[:,0].tolist()

        elif na == 0:
            maparams = theta_[:,0].tolist()
            arparams = [0] * nb

        else:
            maparams = theta_[-nb:,0].tolist() + [0.] * (na - nb)
            arparams = theta_[:na,0].tolist()

        num = np.r_[1, maparams]
        den = np.r_[1, arparams]
        sys = (den, num, 1)
        _, ei = signal.dlsim(sys, y)

        xi = (e - ei) / delta
        xi = xi.tolist()
        X = np.hstack([X, xi])
        theta_ = theta.copy()

    # A and g
    A = np.matmul(X.T, X)
    g = np.matmul(X.T, e)

    return SSE_theta, A, g

def lm_step2(theta, y, A, g, mu=0.01):

    # delta_theta and theta_new
    I = np.identity(n)
    delta_theta = np.matmul(np.linalg.inv(np.add(A, (mu * I))), g)
    theta_new = theta + delta_theta

    # e_new
    if nb == 0:
        maparams = [0] * na
        arparams = theta_new[:,0].tolist()

    elif na == 0:
        maparams = theta_new[:,0].tolist()
        arparams = [0] * nb

    else:
        maparams = theta_new[-nb:,0].tolist() + [0.] * (na - nb)
        arparams = theta_new[:na,0].tolist()

    num = np.r_[1, maparams]
    den = np.r_[1, arparams]
    sys = (den, num, 1)
    _, e_new = signal.dlsim(sys, y)

    # SSE_new
    SSE_new = np.matmul(e_new.T, e_new)

    return delta_theta, theta_new, SSE_new

# step 3
def lm_step3(theta, y, theta_new, SSE_theta, SSE_new, A, g, delta_theta):

    epsilon = 10 ** (-4)
    num_iteration = 1
    max_ = 50
    mu = 0.01
    mu_max = 10 ** 10
    theta_hat, var_e, cov_theta = 0, 0, 0
    SSE = []
    SSE.extend([SSE_theta, SSE_new])

    while num_iteration < max_:
        if SSE_new < SSE_theta:
            if np.linalg.norm(delta_theta) < epsilon:
                theta_hat = theta_new
                var_e = SSE_new / (N - n)
                cov_theta = var_e * (np.linalg.inv(A))

                return theta_hat, var_e, cov_theta, SSE
                break

            else:
                theta = theta_new
                mu = mu / 10

        while SSE_new >= SSE_theta:
            mu *= 10
            if mu > mu_max:
                theta_hat = theta_new
                var_e = SSE_new / (N - n)
                cov_theta = var_e * (np.linalg.inv(A))
                print("Error! mu is too large!")
                return theta_hat, var_e, cov_theta, SSE
                break
            else:
                delta_theta, theta_new, SSE_new = lm_step2(theta, y, A, g, mu)
                SSE.append(SSE_new)

        num_iteration += 1
        if num_iteration > max_:
            theta_hat = theta_new
            var_e = SSE_new / (N - n)
            cov_theta = var_e * (np.linalg.inv(A))
            print("Error! Number of iterations reached max!")
            return theta_hat, var_e, cov_theta, SSE
            break
        else:
            theta = theta_new
            SSE_theta, A, g = lm_step1(theta, y)
            delta_theta, theta_new, SSE_new = lm_step2(theta, y, A, g)
            SSE.append(SSE_new)

def get_MSE(list1, list2):
    # list1 --> test
    # list2 --> forecast
    errorsq = 0
    for i in range(len(list1)):
        errorsq += (list1[i] - list2[i]) ** 2
    mse = errorsq / len(list1)

    return mse

def forecast_errors(list1, list2):
    # list1 --> test
    # list2 --> forecast
    errors = []
    for i in range(len(list1)):
        err = list1[i] - list2[i]
        errors.append(err)

    return errors

def get_Q(l, ry, lags):
    # l is size of samples
    # ry is the list of acf values
    Q = l * np.sum(np.square(ry[lags:]))

    return Q


