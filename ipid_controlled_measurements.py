#!/usr/bin/env python3
from numpy import array, asarray
import numpy as np
from ipid_prediction_lib import *
from math import sqrt
#from sklearn import datasets, preprocessing
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
from subprocess import STDOUT, check_output
import re
from collections import defaultdict
#from ipid_online_analysis_lr import predict_ipids
import GP


class go_string(Structure):
    _fields_ = [
        ("p", c_char_p),
        ("n", c_int)]


def extract(string_arr):
    arr = []
    matches = re.findall(regex, string_arr)
    for match in matches:
        arr.append(int(match))
    return arr


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
                predictions[i] = predictions[i] + 65536
                #res.append(2 * abs(pre-actual[i]) / (actual[i] + pre))
            else:
                actual[i] = actual[i] + 65536
                #res.append(2 * abs(predictions[i]-ac) / (ac + predictions[i]))
            continue
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



def probe(sip, ipv4, protocol, flag, port, ns):
    port = str(port)
    sip = bytes(sip, 'utf-8')
    ipv4 = bytes(ipv4, 'utf-8')
    protocol = bytes(protocol, 'utf-8')
    flag = bytes(flag, 'utf-8')
    port = bytes(port, 'utf-8')
    ns = bytes(ns, 'utf-8')
    sip = go_string(c_char_p(sip), len(sip))
    ip = go_string(c_char_p(ipv4), len(ipv4))
    proto = go_string(c_char_p(protocol), len(protocol))
    flag = go_string(c_char_p(flag), len(flag))
    port = go_string(c_char_p(port), len(port))
    ns = go_string(c_char_p(ns), len(ns))

    a = lib.probe(sip, ip, proto, flag, port, ns)
    return a


def spoofing_probe(ipv4, protocol, port, ns, dst_ip, dst_port, n, flag):
    ipv4 = bytes(ipv4, 'utf-8')
    protocol = bytes(protocol, 'utf-8')
    ns = bytes(ns, 'utf-8')
    dst_ip = bytes(dst_ip, 'utf-8')
    n = bytes(n, 'utf-8')
    flag = bytes(flag, 'utf-8')
    port = bytes(port, 'utf-8')
    dst_port = bytes(dst_port, 'utf-8')

    ip = go_string(c_char_p(ipv4), len(ipv4))
    proto = go_string(c_char_p(protocol), len(protocol))
    ns = go_string(c_char_p(ns), len(ns))
    dst_ip = go_string(c_char_p(dst_ip), len(dst_ip))
    n = go_string(c_char_p(n), len(n))
    flag = go_string(c_char_p(flag), len(flag))
    port = go_string(c_char_p(port), len(port))
    dst_port = go_string(c_char_p(dst_port), len(dst_port))
    lib.spoofing_probe(ip, dst_ip, proto, port, dst_port,
                       ns, n, flag)  # port: reflector port


def spoofing_samples(diff_data):
    # when the estimated error is the maximum of previous errors, maybe an abnormal value when there is ana outlier
    
    x = np.max(diff_data, axis=-1)
    
    u = np.mean(diff_data)
    s = np.std(diff_data)
    n = 0
    
    #print(norm.ppf(0.05), x, u, s)
    
    n = 1+int(-norm.ppf(0.05)*s+x-u)  # 2.06
    
    
    #n = int(1 - norm.ppf(0.05) * s) + 3 # this formular is only applicable for well-predictable IPID counters

    return n, u, s


def is_open_port(u, s, e, n):
    if s == 0:
        if abs(e) >= n:
            return True
        else:
            return False

    v = (e-u)/s
    if norm.cdf(v) < 0.05:  # p = 0.02
        return True
    return False


def test_dst_port(sip, ip, protocol, flag, port, ns):
    count = 0
    status = 'open'
    for i in range(3):
        ipid = probe(sip, ip, protocol, flag, port, ns)
        if ipid == -1:
            count = count+1
    if count == 3:
        status = 'closed'
    return status

def single_port_scan(sip, ip, protocol, port, ns, dst_ip, dst_port, plth, spoof, dataset):
    code = 0
    count = 0
    for i in range(3):
        ipid = probe(sip, ip, protocol, 'SA', port, ns)
        if ipid <= 0:
            count = count+1
    if count == 3:
        logging.info(
            'Test Failed due to unreachable or inapplicable: {a}'.format(a=ip))
        code = 1
        return code, dst_ip
    src_ip = sip
    #astatus = test_dst_port(src_ip, dst_ip, protocol, 'S', dst_port, ns)
    astatus = 'closed'
    
    sliding_window = list()
    wlth = 5 # 10
    flag = 'control'

    ipids = list()
    actual = list()
    predictions = list()
    chps_ind = list()
    outlier_ind = list()
    tem_actual = list()

    mae, smape, n, u, s = 0.0, 0.0, int(0), 0.0, 0.0
    while True:
        if i % 2 == 0:
            sip = "45.125.236.166"
        else:
            sip = "45.125.236.167"
        ipid = probe(sip, ip, protocol, 'SA', port, ns)
        start = time.monotonic()
        ipids.append(ipid)
        if ipid == -1:
            ipid = math.nan
        sliding_window.append(ipid)
        tem_actual.append(ipid)
        if len(sliding_window) == wlth+1:
            actual.append(sliding_window[-1])
            sliding_window.pop(0)
        if len(predictions) == plth-1:
            diff = eliminate_trans_error(chps_ind, actual, predictions)
            after_diff = list()
            for v in diff:
                if math.isnan(v):
                    continue
                after_diff.append(v)

            if len(after_diff) < (plth-1) * 0.7:
                logging.info('Invalid: {a}'.format(a=ip))
                code = 1
                return code, dst_ip
            
            mae = np.mean(abs(array(after_diff)))
            rmse = np.mean(array(after_diff)**2)**.5
            smape = sMAPE02(chps_ind, actual, predictions)
           
            n, u, s = spoofing_samples(after_diff)
            # f.write(ip+','+str(smape)+','+str(n)+'\n')
            if n > 10:
                #only for validating good counters
                logging.info('n>10, require retest: {a}'.format(a=ip))  # 10
                code = 1
                return code, dst_ip
                
                #n = 10

            if spoof:

                # spoofing_probe(ip, protocol, port, ns, dst_ip, dst_port, n, flag)  # port should be random

                # test_pred_n, port should be random
                spoofing_probe(dst_ip, protocol, str(dst_port),
                               ns, ip, str(port), str(n), flag)

        if len(sliding_window) == wlth:
            count = 0
            for x in sliding_window:
                if math.isnan(x):
                    count = count + 1
            if count/wlth > 0.5:
                predictions.append(math.nan)
                end = time.monotonic()
                elapsed = end-start
                #lambda elapsed:  time.sleep(1-elapsed) if elapsed < 1 else time.sleep(0)
                time.sleep(1)
                continue
            times = list()
            for i in range(len(sliding_window)):
                times.append(i)
            tHistory = times
            MAX = 65536


            if containNAN(sliding_window):
                vel = computeIpidVelocityNan(
                    sliding_window, list(range(len(sliding_window))), MAX)
            else:
                vel = computeIpidVelocity02(sliding_window, list(
                    range(len(sliding_window))), MAX)  # eliminate the outliers' impact

            thr = MAX / 2
            if len(predictions) > 1 and alarm_turning_point(thr, tem_actual[-2], tem_actual[-1], MAX):
                chps_ind.append(i-2)
                chps_ind.append(i-1)

            if len(predictions) == plth:
                break  # Update!!!

            

            
            #one_time_forecast(sliding_window, predictions, MAX)
            n_step = 2
            n_hold_out = 1
            measure = 'RMSE'
            sliding_window = fill_predicted_values(sliding_window, predictions)
            GP.one_time_forecast(sliding_window, predictions, n_step, n_hold_out, MAX, measure)
            
            
            
            if predictions[-1] < 0:
                predictions[-1] = 0

        end = time.monotonic()
        elapsed = end-start
        #lambda elapsed:  time.sleep(1-elapsed) if elapsed < 1 else time.sleep(0)
        time.sleep(1)
    diff = eliminate_trans_error(chps_ind, actual, predictions)
    if math.isnan(diff[-1]):
        logging.info('Packet loss: {a}'.format(a=ip))
        code = 1
        return code, dst_ip
    err = diff[-1]  # err is always negative.
    status = None

    if is_open_port(u, s, err, n):
        status = 'open port'
    else:
        status = 'closed or filtered port!'

    dataset['ip'].append(ip)
    dataset['mae'].append(mae)
    dataset['rmse'].append(rmse)
    dataset['smape'].append(smape)
    dataset['n'].append(n)
    dataset['status'].append(status)
    dataset['dst_ip'].append(dst_ip)
    dataset['astatus'].append(astatus)
    #print(ip, dst_ip, status, astatus)
    logging.info('{a} | {b} | {c} | {d}'.format(
        a=ip, b=dst_ip, c=actual, d=predictions))
    return code, dst_ip



def detect_new(err1, p, err2, u, s, n):
    status1 = 'inbound censor'
    status2 = 'no censor'
    status3 = 'outbound censor'
    status4 = 'unkonwn'
    st1 = detect01(err1, p, u, s, n)
    st2 = detect01(err2, p, u, s, n)
    if st1 == 'normal':
        return status1
    elif st1 == 'abnormal' and st2 == 'normal':
        return status2
    elif st1 == 'abnormal' and st2 == 'abnormal':
        return status3
    else:
        return status4


def single_censor_rov_measure(sip, ip, protocol, port, ns, dst_ip, dst_port, plth, spoof1, spoof2, dataset):
    
    code = 0
    count = 0
    for i in range(3):
        ipid = probe(sip, ip, protocol, 'SA', port, ns)
        if ipid <= 0:
            count = count+1
    if count == 3:
        logging.info(
            'Test Failed due to unreachable or inapplicable: {a}'.format(a=ip))
        code = 1
        return code, dst_ip
    src_ip = sip
    #astatus = test_dst_port(src_ip, dst_ip, protocol, 'S', dst_port, ns)
    astatus = 'no censor'
    

    '''
	if astatus == 'open': ##need to be updated when no open
		logging.info('Open: {a}'.format(a= dst_ip))
		code = 1
		return code, dst_ip
	'''
    
    sliding_window = list()
    wlth = 5 # 10
    flag = 'control'
    RTO = 3 

    ipids = list()
    actual = list()
    predictions = list()
    chps_ind = list()
    outlier_ind = list()
    tem_actual = list()

    mae, smape, n, u, s = 0.0, 0.0, int(0), 0.0, 0.0
    while True:
        if i % 2 == 0:
            sip = "45.125.236.166"
        else:
            sip = "45.125.236.167"
        ipid = probe(sip, ip, protocol, 'SA', port, ns)
        start = time.monotonic()
        ipids.append(ipid)
        if ipid == -1:
            ipid = math.nan
        sliding_window.append(ipid)
        tem_actual.append(ipid)
        if len(sliding_window) == wlth+1:
            actual.append(sliding_window[-1])
            sliding_window.pop(0)
        if len(predictions) == plth-1:
            diff = eliminate_trans_error(chps_ind, actual, predictions)
            
            after_diff = list()
            for v in diff:
                if math.isnan(v):
                    continue
                after_diff.append(v)

            if len(after_diff) < (plth-1) * 0.7:
                logging.info('Invalid: {a}'.format(a=ip))
                code = 1
                return code, dst_ip
                
            
            mae = np.mean(abs(array(after_diff)))
            rmse = np.mean(array(after_diff)**2)**.5
            smape = sMAPE02(chps_ind, actual, predictions)
            
            
            n, u, s = spoofing_samples(after_diff)
            # f.write(ip+','+str(smape)+','+str(n)+'\n')
            if n > 10:
                #only for validating good counters
                logging.info('n>10, require retest: {a}'.format(a=ip))  # 10
                code = 1
                return code, dst_ip
                
                #n = 10

            if spoof1:

                # spoofing_probe(ip, protocol, port, ns, dst_ip, dst_port, n, flag)  # port should be random

                # test_pred_n, port should be random
                spoofing_probe(dst_ip, protocol, str(dst_port),
                               ns, ip, str(port), str(n), flag)
        
        if len(predictions) == plth+RTO-1:
            if spoof2:
            	spoofing_probe(dst_ip, protocol, str(dst_port),
                               ns, ip, str(port), str(n), flag)
        		
        	
        if len(sliding_window) == wlth:
            count = 0
            for x in sliding_window:
                if math.isnan(x):
                    count = count + 1
            if count/wlth > 0.5:
                predictions.append(math.nan)
                end = time.monotonic()
                elapsed = end-start
                #lambda elapsed:  time.sleep(1-elapsed) if elapsed < 1 else time.sleep(0)
                time.sleep(1)
                continue
            times = list()
            for i in range(len(sliding_window)):
                times.append(i)
            tHistory = times
            MAX = 65536

            

            if containNAN(sliding_window):
                vel = computeIpidVelocityNan(
                    sliding_window, list(range(len(sliding_window))), MAX)
            else:
                vel = computeIpidVelocity02(sliding_window, list(
                    range(len(sliding_window))), MAX)
                    
            thr = MAX / 2
            
            if len(predictions) > 1 and alarm_turning_point(thr, tem_actual[-2], tem_actual[-1], MAX):
                chps_ind.append(i-2)
                chps_ind.append(i-1)

            if len(predictions) == plth+RTO:
                break  # Update!!!
                
            n_step = 2
            n_hold_out = 1
            measure = 'RMSE'
            sliding_window = fill_predicted_values(sliding_window, predictions)
            
            GP.one_time_forecast(sliding_window, predictions, n_step, n_hold_out, MAX, measure)
            
            if predictions[-1] < 0:
                predictions[-1] = 0

        end = time.monotonic()
        elapsed = end-start
        #lambda elapsed:  time.sleep(1-elapsed) if elapsed < 1 else time.sleep(0)
        time.sleep(1)
    diff = eliminate_trans_error(chps_ind, actual, predictions)
    if math.isnan(diff[-1]):
        logging.info('Packet loss: {a}'.format(a=ip))
        code = 1
        return code, dst_ip
        
    err1 = diff[-(RTO+1)]
    p = 0.05
    err2 = diff[-1]
    status = None
    status = detect_new(err1, p, err2, u, s, n)
    #print(status, n, predictions, actual)
    dataset['ip'].append(ip)
    dataset['mae'].append(mae)
    dataset['rmse'].append(rmse)
    dataset['smape'].append(smape)
    dataset['n'].append(n)
    dataset['status'].append(status)
    dataset['dst_ip'].append(dst_ip)
    dataset['astatus'].append(astatus)
    #print(ip, dst_ip, status, astatus)
    logging.info('{a} | {b} | {c} | {d}'.format(
        a=ip, b=dst_ip, c=actual, d=predictions))
    return code, dst_ip
    
def fill_miss_values(data):
    s = pd.Series(data)
    s = s.interpolate(method='pad')
    return (s.interpolate(method='linear', limit_direction='both').values % 65536).tolist()


def fill_predicted_values(data, predictions):
    if len(predictions) == 0:
        data = fill_miss_values(data)
        
    elif math.isnan(data[-1]) and not math.isnan(predictions[-1]):
        
        data[-1] = int(predictions[-1])
        
    else:
    
    	data = fill_miss_values(data)
    	
    return data



def test():

    # with open('../ipid_prediction/evaluate/online_analysis/lr.reflectors.(low).res', 'r') as filehandle:
    for i in [1, 2, 3]:
        dataset = {
            'ip': [],
            'mae': [],
            'rmse': [],
            'smape': [],
            'n': [],
            'dst_ip': [],
            'status': [],
            'astatus': [],
        }
        
        reflectors = []
        
        protocol = 'tcp'
        # port: zombie's port
        #port = 33435 #zombie's port when udp
        #port = 80 # zombie's port when tcp
        port = random.randrange(10000, 65535, 1) # when TCP random 
        ns = ''
        dst_ip = '45.125.236.72'  # an IP we control
        dst_port = 80
        src_ip = '45.125.236.166'
        
        with open('./tcp_random_global_high_Pr_ips.dat', 'r') as filehandle: #udp_random_zombies.dat
            filecontents = filehandle.readlines()
            for line in filecontents:
                fields = line.split(",")
                if len(fields) < 1:
                    continue
                ip = fields[0].strip('\n')
                reflectors.append(ip)
                
                
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for ip in reflectors:
                #futures.append(executor.submit(single_censor_measure, src_ip, ip, protocol, port, ns, dst_ip, dst_port, 30, True, False, dataset))
                futures.append(executor.submit(single_port_scan, src_ip, ip, protocol, port, ns, dst_ip, dst_port, 30, False, dataset))

            for future in concurrent.futures.as_completed(futures):
                future.result()

        df = pd.DataFrame(dataset)
        df.to_csv('./app_val_port_scan.nonspoof.0'+str(i)+'.v2.res', index=False)
        
        
def zombies_analysis_TN():

    ips = set()
    code = 0
    res = defaultdict(list)
    
    #f = open('../SoK/Datasets/results/ipid_tcp_scan.random.nonspoof.TN.res', 'w')
    f = open('./Datasets/results/app_val_port_scan.nonspoof.TN.res', 'w')
    for i in ["01", "02", "03"]:
        #with open('../SoK/Datasets/results/ipid_tcp_scan.random.nonspoof.'+i+'.res') as filehandle:
        with open('./Datasets/results/app_val_port_scan.nonspoof.'+i+'.v2.res') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                fields = line.split(",")
                if len(fields) < 1:
                    continue
                ip = fields[0]
                
                
                status = fields[-2]
                
                
                if "closed" in status:
                    code = 1
                elif "open" in status:
                    code = 0
                else:
                    continue
                
                
                '''
                if "no censor" in status:
                    code = 1
                elif "inbound" in status or "outbound" in status:
                    code = 0
                else:
                    continue
                '''

                res[ip].append(code)
                
                ips.add(fields[0])

        # f0.close()
    print(len(ips))
    for ip in res:

        if sum(res[ip]) >= 2:

            f.write(ip+','+'TN'+'\n')
        elif len(res[ip]) == 2 and sum(res[ip]) == 1:
            f.write(ip+','+'Failed'+'\n')
        elif len(res[ip]) == 1:
            f.write(ip+','+'Failed'+'\n')
        elif len(res[ip]) > 2 and sum(res[ip]) < 2:
            f.write(ip+','+'FP'+'\n')

    f.close()


def zombies_analysis_TP():

    ips = set()
    code = 0
    res = defaultdict(list)
    
    #f = open('../SoK/Datasets/results/ipid_tcp_scan.random.spoof.TP.res', 'w')
    f = open('./Datasets/results/app_val_port_scan.spoof.TP.res', 'w')
    for i in ["01", "02", "03"]:
        #with open('../SoK/Datasets/results/ipid_tcp_scan.random.spoof.'+i+'.res') as filehandle:
        with open('./Datasets/results/app_val_port_scan.spoof.'+i+'.v2.res') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                fields = line.split(",")
                if len(fields) < 1:
                    continue
                ip = fields[0]

                status = fields[-2]
                
                
                if "closed" in status:
                    code = 0
                elif "open" in status:
                    code = 1
                else:
                    continue
                
                
                '''
                if "no censor" in status:
                    code = 0
                elif "inbound" in status or "outbound" in status:
                    code = 1
                else:
                    continue
                '''
                    
                res[ip].append(code)
                
                ips.add(fields[0])

        # f0.close()
    print(len(ips))
    for ip in res:

        if sum(res[ip]) >= 2:

            f.write(ip+','+'TP'+'\n')
        elif len(res[ip]) == 2 and sum(res[ip]) == 1:
            f.write(ip+','+'Failed'+'\n')
        elif len(res[ip]) == 1:
            f.write(ip+','+'Failed'+'\n')
        elif len(res[ip]) > 2 and sum(res[ip]) < 2:
            f.write(ip+','+'FN'+'\n')

    f.close()



lib = cdll.LoadLibrary("./ipid_pred_lib.so")
logging.basicConfig(level=logging.INFO, filename='./ipid_controlled_measurements.log')  # idle_scan


def main():
    test()
   


if __name__ == "__main__":
    # execute only if run as a script
    main()
