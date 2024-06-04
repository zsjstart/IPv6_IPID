#!/usr/bin/env python3
from numpy import array, asarray
import numpy as np
from ipid_prediction_lib import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import datasets, preprocessing
import time
import decimal
import threading
from ctypes import *
import concurrent.futures
import pandas as pd
import math
import statistics
from scipy.stats import norm, linregress
import random
import csv
import logging
from sklearn import linear_model
#import features_extraction_lib
from features_extraction_lib import extract
import re
import warnings
from matplotlib import pyplot as plt
import GP

warnings.filterwarnings("ignore")


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


'''
def extract(string_arr):
    arr = []
    matches = re.findall(regex, string_arr)
    for match in matches:
        arr.append(int(match))
    return arr
'''


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


def split(ids):
    ids1 = []
    ids2 = []
    for i in range(0, len(ids)):
        if i % 2 == 0:
            ids1.append(ids[i])

        else:
            ids2.append(ids[i])

    return ids1, ids2

# This method is appliable mainly when testing global counters


def sMAPE02(chps_ind, actual, predictions, MAX):
    res = list()
    for i in range(len(actual)):
        if i in chps_ind and abs(predictions[i]-actual[i]) > MAX/2:
            if predictions[i] < actual[i]:
                predictions[i] = predictions[i] + MAX
                #res.append(2 * abs(pre-actual[i]) / (actual[i] + pre))
            else:
                actual[i] = actual[i] + MAX
                #res.append(2 * abs(predictions[i]-ac) / (ac + predictions[i]))

        if (actual[i] + predictions[i]) != 0:
            if (actual[i] + predictions[i]) < 0:

                continue
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



def MAPE(chps_ind, actual, predictions):
    res = list()
    for i in range(len(actual)):
        if i in chps_ind and abs(predictions[i]-actual[i]) > 30000:
            if predictions[i] < actual[i]:
                pre = predictions[i] + 65536
                res.append(abs(pre-actual[i]) / actual[i])
            else:
                ac = actual[i] + 65536
                res.append(abs(predictions[i]-ac) / ac)
            continue
        if (actual[i] + predictions[i]) != 0:
            if (actual[i] + predictions[i]) < 0:
                continue
            res.append(abs(predictions[i]-actual[i]) / actual[i])
        else:
            res.append(0)
    after_res = list()
    for v in res:
        if math.isnan(v):
            continue
        after_res.append(v)
    return np.mean(after_res)
    
    
def alarm_turning_point(thr, a1, a2, MAX):
    alarm = False
    delta = a2 - a1
    # a2-a1+MAX approximates to a2 (close to 1 in ideal)
    if delta < 0 and rela_diff(a1, a2, MAX) < thr:
        alarm = True
    return alarm


def eliminate_trans_error(chps_ind, actual, predictions, MAX):
    diff = list()
    for i in range(len(actual)):
        # if the turning point is predicted with a prior second, then the main prediction error is on the upper turining point, otherwise, th error is on the lower turning point.
        if i in chps_ind and abs(predictions[i]-actual[i]) > MAX/2:
            if predictions[i] < actual[i]:
                diff.append(predictions[i]-actual[i] + MAX)
            else:
                diff.append(predictions[i]-actual[i] - MAX)
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


def data_preprocess(thr, history, MAX):
    data = [i for i in history]
    wraps = list()
    for i in range(len(data)-1):
        if data[i+1] - data[i] < 0 and rela_diff(data[i], data[i+1], MAX) < thr:
            wraps.append(i+1)
    for _, i in enumerate(wraps):
        for t in range(i, len(data)):
            data[t] = data[t] + MAX
    return wraps, data


def pre_processing(sequence, MAX):
    #history = filter_outliers(history, MAX)

    diff_data = difference(sequence, 1, MAX)

    diff_data = array(diff_data).reshape(-1, 1)
    scaler = preprocessing.MinMaxScaler()
    # scaling the input and output data to the range of (0,1)
    diff_data = scaler.fit_transform(diff_data)
    minimum = scaler.data_min_
    maximum = scaler.data_max_
    return diff_data, maximum, minimum


def one_time_forecast(sequence, predictions, MAX):
    #X = np.array(times).reshape(-1, 1)
    #y = np.array(data)
    diff_data, maximum, minimum = pre_processing(sequence, MAX)
    X = np.array(range(len(diff_data))).reshape(-1, 1)
    y = np.array(diff_data)
    model = linear_model.LinearRegression().fit(X, y)
    nt = np.array(len(diff_data)).reshape(-1, 1)

    y_pred = model.predict(nt)[0]
    y_pred = denormalize(y_pred, maximum, minimum)
    prediction = (y_pred[0] + sequence[-1]) % MAX

    predictions.append(prediction)



def fill_miss_values(data, MAX):
    s = pd.Series(data)
    s = s.interpolate(method='pad')
    return (s.interpolate(method='linear', limit_direction='both').values % MAX).tolist()


def fill_predicted_values(data, predictions, MAX):
    if len(predictions) == 0:
        data = fill_miss_values(data, MAX)
        return data
    elif math.isnan(data[-1]):
        data[-1] = int(predictions[-1])
    return data


def calculate_pred_error(sequence, l, valid_length, MAX, container):
    
    for i, v in enumerate(sequence):
        if v == -1:
            sequence[i] = math.nan
            # preprocessing dataset

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
       

        thr = MAX/2

        if i > 1 and alarm_turning_point(thr, tem_actual[-2], tem_actual[-1], MAX):
            chps_ind.append(i-2)
            chps_ind.append(i-1)
        
        if i == len(actual):
            break
         
        history = fill_predicted_values(history, predictions, MAX)
        n_step = 2
        n_hold_out = 1
        measure = 'RMSE'
        GP.one_time_forecast(history, predictions, n_step, n_hold_out, MAX, measure)
        
        
        if predictions[-1] < 0:
            predictions[-1] = 0

        end = time.monotonic()
        elps.append(end-start)
        tem_actual.append(actual[i])
        history.append(actual[i])
        history.pop(0)
        
    # identify change points and then eliminate the error from the transformation at the restore.
    t = np.mean(elps)
    
    after_predictions = list()
    for v in predictions:
        if math.isnan(v):
            after_predictions.append(v)
        else:
            after_predictions.append(round(v))
    predictions = after_predictions

    diff = eliminate_trans_error(chps_ind, actual, predictions, MAX)
    errs = list()
    for v in diff:
        if math.isnan(v):
            continue
        errs.append(v)
    
    if valid_length is None: valid_length = len(errs)
    errs = errs[:valid_length]
    
    mae = np.mean(np.abs(errs))

    rmse = np.mean((np.array(errs)/MAX)**2)**.5 * MAX

    smape = sMAPE02(chps_ind[:valid_length], actual[:valid_length], predictions[:valid_length], MAX)
    
    if container is not None:
        container.append(smape)
    
    
    x = np.max(errs, axis=-1)
    
    u = np.mean(errs)
    s = np.std(errs)
    a = 0.05
    
    ns = 1+int(-norm.ppf(a)*s+x-u)   # 2.06,
    upper = np.array(predictions) - norm.ppf(a)*s - u
    lower = np.array(predictions) - norm.ppf(1-a)*s - u
    
    return mae, rmse, smape, t, u, s, ns, predictions, actual


def ipidtsf():
    ips = []
    maes, rmses, smapes, nss = [], [], [], []
    MAX = 65536
    
    for l in [5]:

       
        with open('./Datasets/app_val_global_4573.dat', 'r') as filehandle:
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

                x, y = split(s)
                ids = s[0:35]
                valid_length = len(ids) - 5
                mae, rmse, smape, _, _, _, ns, _, _ = calculate_pred_error(
                    ids, l, valid_length, MAX, None)
                    
                
    
def main():
    ipidtsf()
    


if __name__ == "__main__":
    # execute only if run as a script
    main()
