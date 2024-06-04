#!/usr/bin/env python3
from numpy import array, asarray
from features_extraction_lib import extract
import numpy as np
from pandas import DataFrame, concat
from ipid_prediction_lib import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import datasets, preprocessing
#from neupy import algorithms
import time
import decimal
import threading
from ctypes import *
import concurrent.futures
import pandas as pd
import math
import statistics

#from grnn02 import one_time_forecast02
from scipy.stats import norm
import random
import csv
import logging
#from matplotlib.animation import FuncAnimation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
import warnings
import pmdarima as pm
import statsmodels

warnings.filterwarnings("ignore")

lib = cdll.LoadLibrary("./ipid_pred_lib.so")
#logging.basicConfig(level=logging.INFO, filename='grnn.test.log')


class go_string(Structure):
    _fields_ = [
        ("p", c_char_p),
        ("n", c_int)]


def modify(times):
    start = times[0]
    i = 0
    for time in times:
        times[i] = int(round(float(time - start)/1000000.0))
        i += 1
    return times


def computeIpidVelocity(ids, times, MAX):

    spd = float(0)

    for i in range(0, len(ids)-1):

        gap = float(rela_diff(ids[i], ids[i+1], MAX))
        # dur = float(times[i+1]-times[i])/1000000000.0 #unit: ID/s
        dur = float(times[i+1]-times[i])
        spd += gap/dur

    spd /= float(len(ids)-1)

    return round(spd, 3)


def computeIpidVelocity02(ids, times, MAX):
    id_seg = list()
    time_seg = list()
    vels = list()
    for i in range(len(ids)):
        id_seg.append(ids[i])
        time_seg.append(times[i])
        if len(id_seg) == 3:
            vel = computeIpidVelocity(id_seg, time_seg, MAX)
            vels.append(vel)
            id_seg = []
            time_seg = []
    return np.median(vels)


def computeIpidVelocitySeg(ids, times, MAX):
    id_segment = []
    time_segment = []
    vels = []
    for i in range(len(ids)):
        if math.isnan(ids[i]):
            if len(id_segment) >= 3:
                vel = computeIpidVelocity(id_segment, time_segment, MAX)
                vels.append(vel)
            id_segment = []
            time_segment = []
            continue
        id_segment.append(ids[i])
        time_segment.append(times[i])
    if len(id_segment) >= 3:  # without NAN
        vel = computeIpidVelocity(id_segment, time_segment, MAX)
        vels.append(vel)
    if len(vels) == 2 and len(id_segment) > len(ids)/2:
        return vels[1]
    return np.median(vels)


def computeIpidVelocityNan(ids, times, MAX):
    id_segment = []
    time_segment = []
    for i in range(len(ids)):
        if math.isnan(ids[i]):
            continue
        id_segment.append(ids[i])
        time_segment.append(times[i])
    vel = computeIpidVelocity(id_segment, time_segment, MAX)
    return vel


def count_ipid_wraps(data):
    count = 0
    for i in range(0, len(data)-1):
        if data[i+1]-data[i] < 0:
            count = count + 1
    return count


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))

    # put it all together
    agg = concat(cols, axis=1)
    # print(agg)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def obtain_restore_data(sequence, diff_data):
    base_data = list()
    restore_data = list()
    for i in range(3, len(diff_data)):
        if math.isnan(diff_data[i-3]+diff_data[i-2]+diff_data[i-1]+diff_data[i]):
            continue
        base_data.append(sequence[i])
        restore_data.append(sequence[i+1])
    return base_data, restore_data

# split a univariate dataset into train/test sets


def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

# split a univariate sequence into samples


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)-n_steps):
        # find the end of this pattern
        end_ix = i + n_steps
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def sMAPE02(chps_ind, actual, predictions):
    res = list()
    for i in range(len(actual)):
        if i in chps_ind and abs(predictions[i]-actual[i]) > 30000:
            if predictions[i] < actual[i]:
                pre = predictions[i] + 65536
                res.append(2 * abs(pre-actual[i]) / (actual[i] + pre))
            else:
                ac = actual[i] + 65536
                res.append(2 * abs(predictions[i]-ac) / (ac + predictions[i]))
            continue
        if (actual[i] + predictions[i]) != 0:
            res.append(
                2 * abs(predictions[i]-actual[i]) / (actual[i] + predictions[i]))
        else:
            res.append(0)
    after_res = list()
    for v in res:
        if math.isnan(v):
            continue
        after_res.append(v)
    return np.mean(after_res)


def forecast(train, X_test, kernel):
    train = asarray(train)
    # split into input and output columns
    X_train, y_train = train[:, :-1], train[:, -1]

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
    gp.fit(X_train, y_train)

    y_pred, sigma = gp.predict([X_test], return_std=True)
    return y_pred[0]


# Define the kernels
rbf_kernel = 1.0 * RBF(length_scale=1.0)
dot_product_kernel = DotProduct(sigma_0=1.0)
# Adjust 'nu' for desired smoothness
matern_kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
rational_quadratic_kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
white_kernel = WhiteKernel(noise_level=0.1)  # White noise kernel


def walk_forward_validation(history, n_hold_out, measure):

    train, test = train_test_split(history, n_hold_out)

    '''
    X_train, y_train = train[:, :-1], train[:, -1]  # y_train shape: 1D array
    X_test, y_test = test[:, :-1], test[:, -1]
    y_train = y_train.reshape(-1, 1)
    #y_test = y_test.reshape(-1, 1)
    '''

    errs = []
    paras = []

    #dot_product_kernel, matern_kernel, rational_quadratic_kernel

    for kernel in [white_kernel]:

        y_pred = list()

        h_d = [x for x in train]
        for i in range(len(test)):
            # split test row into input and output columns
            X_test, y_test = test[i, :-1], test[i, -1]
            yhat = forecast(h_d, X_test, kernel)
            # store forecast in list of predictions
            y_pred.append(yhat)
            # add actual observation to history for the next loop
            h_d.append(test[i])
            h_d.pop(0)
        y_pred = array(y_pred)
        y_pred = y_pred.reshape(-1)
        y_test = test[:, -1]
        y_test = y_test.reshape(-1)
        diff = y_pred - y_test

        if measure == "MAE":
            err = np.mean(np.abs(diff))  # MAE
        elif measure == "RMSE":
            err = np.mean(array(diff)**2)**.5  # RSME
        elif measure == "SMAPE":
            err = sMAPE(y_test, y_pred)
        if math.isnan(err):

            continue
        errs.append(err)
        paras.append(kernel)

    if len(errs) == 0:
        return rbf_kernel

    min_index = min((v, i) for i, v in enumerate(errs))[1]
    kernel = paras[min_index]

    return kernel


def real_time_forecasting02(diff_data, n_step, n_hold_out, measure):

    history = reshape_inputs(diff_data, n_step)
    if len(history) == 0:
        return math.nan
    #kernel = walk_forward_validation(history, n_hold_out, measure)
    kernel = white_kernel
    pred = [history[-1, 1:]]
    y_pred = train_and_predict(history, pred, kernel)
    return y_pred


def sMAPE(actual, prediction):
    actual = array(actual).reshape(-1)
    prediction = array(prediction).reshape(-1)
    res = 2 * np.abs(prediction-actual) / (np.abs(actual) + np.abs(prediction))
    after_res = list()
    for v in res:
        if math.isnan(v):
            v = 0
        after_res.append(v)
    return np.mean(after_res)


def one_time_forecast(sequence, predictions, n_step, n_hold_out, MAX, measure):
    
    diff_data, maximum, minimum = pre_processing(sequence, MAX)
    
    #y_pred = real_time_forecasting02(sequence, diff_data, n_step, C, maximum, minimum, MAX)
    y_pred = real_time_forecasting02(diff_data, n_step, n_hold_out, measure)
    
    y_pred = denormalize(y_pred, maximum, minimum)
    
    prediction = (y_pred[0] + sequence[-1]) % MAX
    
    predictions.append(prediction)
    
    
def is_outliers(u, s, diff):
    v = (diff-u)/s
    if norm.cdf(v) > 0.98 or norm.cdf(v) < 0.02:  # p = 0.05
        return True
    return False
    
def alarm_turning_point(thr, a1, a2, MAX):
    alarm = False
    delta = a2 - a1
    # a2-a1+MAX approximates to a2 (close to 1 in ideal)
    if delta < 0 and rela_diff(a1, a2, MAX) < thr:
        alarm = True
    return alarm


def eliminate_trans_error(chps_ind, actual, predictions):
    diff = list()
    for i in range(len(actual)):
        # if the turning point is predicted with a prior second, then the main prediction error is on the upper turining point, otherwise, th error is on the lower turning point.
        if i in chps_ind and abs(predictions[i]-actual[i]) > 30000:
            if predictions[i] < actual[i]:
                diff.append(predictions[i]-actual[i] + 65536)
            else:
                diff.append(predictions[i]-actual[i] - 65536)
            continue
        diff.append(predictions[i]-actual[i])
    return diff


def containNAN(data):
    for i in range(len(data)):
        if math.isnan(data[i]):
            return True
    return False


def countNans(data):
    num = 0
    for i in range(len(data)-2):
        if math.isnan(data[i]):
            if math.isnan(data[i+1]) and math.isnan(data[i+2]):
                num = 3
                return num
    return num


def filter_outliers(outliers, thr, history, MAX):
    data = filter_outliers01(outliers, history, thr, MAX)
    return data


def data_preprocess(thr, data, MAX):
    wraps = list()
    for i in range(len(data)-1):
        if data[i+1] - data[i] < 0 and rela_diff(data[i], data[i+1], MAX) < thr:
            wraps.append(i+1)
    for _, i in enumerate(wraps):
        for t in range(i, len(data)):
            data[t] = data[t] + MAX
    return wraps


def pre_processing(sequence, MAX):
    diff_data = difference(sequence, 1, MAX)
    diff_data = array(diff_data).reshape(-1, 1)
    scaler = preprocessing.MinMaxScaler()
    # scaling the input and output data to the range of (0,1)
    diff_data = scaler.fit_transform(diff_data)
    minimum = scaler.data_min_
    maximum = scaler.data_max_
    return diff_data, maximum, minimum


def train_and_predict(data, pred, kernel):
    
    X_train, y_train = data[:, :-1], data[:, -1]

    y_train = y_train.reshape(-1, 1)

    # n_restarts_optimizer=2 or 3
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
    gp.fit(X_train, y_train)
    y_pred, sigma = gp.predict(pred, return_std=True)
  
    return y_pred


def reshape_inputs(diff_data, n_steps):
    num = len(diff_data)
    values = asarray(diff_data).reshape((num, 1))
    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=n_steps)
    return data


def one_time_forecast_with_ARMA(sequence, predictions, MAX):

    train = DataFrame(sequence)
    ar = pm.auto_arima(train, information_criterion='aic', start_p=0, start_q=0,
                       max_p=7, max_q=7,  # maximum p and q
                       d=0,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    order = ar.order

    warnings.filterwarnings("ignore")
    model = statsmodels.tsa.arima.model.ARIMA(
        sequence, order=order)  # order = (p, d, q)
    model_fit = model.fit()
    y_pred = model_fit.forecast()

    prediction = (y_pred[0]) % MAX
    predictions.append(prediction)
    
    
def calculate_pred_error(ip, sequence, l,  MAX):
    try:
        for i, v in enumerate(sequence):
            if v == -1:
                sequence[i] = math.nan

        n_step = None
        n_hold_out = None

        elps = list()

        history, actual = sequence[0:l], sequence[l:]
        chps_ind = list()
        predictions = list()
        tem_actual = sequence[0:l]
        for i in range(len(actual)+1):
            start = time.monotonic()

            if containNAN(history):
                vel = computeIpidVelocitySeg(
                    history, list(range(len(history))), MAX)
            else:
                # eliminate the outliers' impact
                vel = computeIpidVelocity02(
                    history, list(range(len(history))), MAX)
            '''
		if vel < 1000:
		    thr = 15000  # experimentially specify the threshold
		else:
		    thr = 30000
		'''

            thr = MAX/2

            if i > 1 and alarm_turning_point(thr, tem_actual[-2], tem_actual[-1], MAX):
                chps_ind.append(i-2)
                chps_ind.append(i-1)

            if i == len(actual):
                break

            # history = fill_miss_values(history) # base.res, try linear_interpolate_miss_values
            history = fill_predicted_values(history, predictions)

            one_time_forecast_with_ARMA(history, predictions, MAX)

            if predictions[-1] < 0:
                predictions[-1] = 0

            end = time.monotonic()
            elps.append(end-start)
            tem_actual.append(actual[i])
            history.append(actual[i])
            history.pop(0)
        # identify change points and then eliminate the error from the transformation at the restore.
        t = statistics.mean(elps)
        after_predictions = list()
        for v in predictions:
            if math.isnan(v):
                after_predictions.append(v)
            else:
                after_predictions.append(round(v))
        predictions = after_predictions

        diff = eliminate_trans_error(chps_ind, actual, predictions)
        errs = list()
        for v in diff:
            if math.isnan(v):
                continue
            errs.append(v)
        if errs == []:
            print(actual, predictions, sequence)

        mae = np.mean(np.abs(errs))

        rmse = np.mean((np.array(errs)/MAX)**2)**.5 * MAX

        smape = sMAPE02(chps_ind, actual, predictions)

        return ip, mae, rmse, smape, t
    except:
        return ip, None, None, None, None


def ARMA():
    sequences = []

    MAX = 65536

    for l in [5]:

        f = open('./ARMA_test_on_benchmark.res', 'w')
        with open('./benchmark.data', 'r') as filehandle:
            filecontents = filehandle.readlines()
            random.shuffle(filecontents)
            for line in filecontents:
                fields = line.split(",")
                if len(fields) < 2:
                    continue
                #ns = fields[1]
                ip = fields[0]
                dataStr = fields[1]
                s = extract(dataStr)
                sequences.append((ip, s))
        print(len(sequences))
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for ip, s in sequences:
                futures.append(executor.submit(
                    calculate_pred_error, ip, s, l,  MAX))

            for future in concurrent.futures.as_completed(futures):
                ip, mae, rmse, smape, t = future.result()

                f.write(ip+','+str(mae)+','+str(rmse) +
                        ','+str(smape)+','+str(t)+'\n')

        f.close()
        
def fill_predicted_values(data, predictions):
    if len(predictions) == 0:
        data = fill_miss_values(data)
        return data
    elif math.isnan(data[-1]):
        data[-1] = int(predictions[-1])
    return data


dataset = {
    'ip': [],
    'rmse': [],
    'smape': []
}


def main():
    # gp()
    ARMA()
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
