import numpy as np
import matplotlib.pyplot as plt
import re
import tsfel
import pandas as pd
import math
from random import randrange
import random
import os
import glob
from scipy.stats import entropy
import time
import numpy
import pickle
import joblib
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import StrMethodFormatter
import ipidtsf
import concurrent.futures


def diff_arr(data):
    diff_arr = []
    for i in range(0, len(data)-1):
        diff_arr.append(data[i+1]-data[i])
    return diff_arr
    
def sd(data):
    return round(np.std(data), 3)


def exp(data):
    n = len(data)
    prb = 1 / n
    sum = 0
    for i in range(0, n):
        sum += (data[i] * prb)
    return round(float(sum), 3)


def average(data):
    return round(sum(data)/len(data), 3)


def weighted_average(data, seg_len):
    #a = np.array(data)
    return round(np.dot(data, seg_len / np.sum(seg_len)), 3)


def rela_diff(a, b, MAX):
    return (b + MAX - a) % MAX


def split_s(ids):
    ids1 = []
    ids2 = []
    for i in range(0, len(ids)):
        if i % 2 == 0:
            ids1.append(ids[i])
        else:
            ids2.append(ids[i])
    return ids1, ids2


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


def find_weird(ids):
    for i in range(len(ids)-1):
        if rela_diff(ids[i], ids[i+1]) > 10000:
            f2.write('Find!'+str(ids[i])+','+str(ids[i+1])+'||')
            f2.write(" ".join(str(x) for x in ids)+'\n')
            return True
    return False


def split(ids, times):
    ids1 = []
    ids2 = []
    times1 = []
    times2 = []
    for i in range(0, len(ids)):
        if i % 2 == 0:
            ids1.append(ids[i])
            # times1.append(times[i])
        else:
            ids2.append(ids[i])
            # times2.append(times[i])
    return ids1, times1, ids2, times2


def count_ipid_wraps(data):
    count = 0
    for i in range(0, len(data)-1):
        # if data[i] == -1 or data[i+1] == -1: continue
        if data[i+1]-data[i] < 0:
            count = count + 1
    return count


def power_bandwidth(data, fs):
    return round(tsfel.feature_extraction.features.power_bandwidth(data, fs), 3)


def computeIpidVelocity(ids, times, MAX):
    # the mean of relative differences between two consecutive IP ID values
    spd = float(0)

    for i in range(0, len(ids)-1):

        gap = float(rela_diff(ids[i], ids[i+1], MAX))
        # dur = float(times[i+1]-times[i])/1000000000.0 #unit: ID/s
        dur = 1
        spd += gap/dur

    spd /= float(len(ids)-1)

    return round(spd, 3)


def max_increment(ids, times, MAX):
    r = []
    for i in range(0, len(ids)-1):
        gap = float(rela_diff(ids[i], ids[i+1], MAX))
        # dur = float(times[i+1]-times[i])/1000000000.0 #unit: ID/s
        dur = 1
        r.append(gap/dur)

    return max(r)


def autocorr(data):
    s = pd.Series(data)
    auto_corr = s.autocorr(lag=1)
    if math.isnan(auto_corr):
        return 0
    else:
        return round(auto_corr, 3)


def crosscorr(x, y):
    s1 = pd.Series(x)
    s2 = pd.Series(y)
    cross_corr = s1.corr(s2)
    if math.isnan(cross_corr):
        return 0
    else:
        return round(cross_corr, 3)


def plot_data(ids):
    y = ids
    x = []
    for i in range(0, len(ids)):
        x.append(i)
    plt.plot(x, y, "-")
    plt.show()


def spectral_centroid(fmg, f):
    if not np.sum(fmg):
        return 0
    else:
        return round(np.dot(f, fmg / np.sum(fmg)), 3)


def spectral_roll_off(fmg, f):
    cum_ff = np.cumsum(fmg)
    value = 0.85 * (np.sum(fmg))
    return round(f[np.where(cum_ff >= value)[0][0]], 3)
    
switcher = {
    'global': 1,
    
    'perConn': 2,
    'random': 3,
    'constant': 4,
    'anomalous': 5,
    
    'random_or_others': 3,
}


def write_to_dic(dataset, c, ip, num_wrap, auto_corr, fd, b,  fr, pr_s, pr_x, pr_y):
    ipid_class = switcher[c]

    dataset['ip'].append(ip)
   
    dataset['num_wrap'].append(num_wrap)
    dataset['autocorr'].append(auto_corr)
   
    dataset['b'].append(b)
    dataset['fd'].append(fd)
    dataset['fr'].append(fr)
    
    dataset['pr_s'].append(pr_s)
    dataset['pr_x'].append(pr_x)
    dataset['pr_y'].append(pr_y)
    
    dataset['class'].append(ipid_class)


def segment_features(ids, times, features, seg_len, sampling_rate, MAX):
    y1, y2 = split_s(ids)
    # distinguish between perConn and random or load-balancing
    # modified it for the entire sequence
    num_wrap = count_ipid_wraps(ids)/len(ids)
    v = computeIpidVelocity(ids, times, MAX)  # / sampling_rate
    max_inc = max_increment(ids, times, MAX)
    ids1, times1, ids2, times2 = split(ids, times)
    max_inc1 = max_increment(ids1, times1, MAX)
    max_inc2 = max_increment(ids2, times2, MAX)
    auto_corr = autocorr(ids)
    cross_corr = crosscorr(ids1, ids2)

    data = np.array(ids, dtype=float)
    data[:] = data - np.mean(data)  # remove zero freqeuncy
    fourier_transform = np.fft.fft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    N = len(data)

    frequency = np.linspace(0, sampling_rate, N)
    fqs = frequency[:N//2+1]
    fqs = fqs / sampling_rate
    ps = power_spectrum[:N//2+1]
    max_index = np.argmax(ps)
    fd = round(fqs[max_index], 3)
    b = power_bandwidth(data, 1)
    fmg = abs_fourier_transform[:N//2+1]
    fc = spectral_centroid(fmg, fqs)
    fr = spectral_roll_off(fmg, fqs)

    features['num_wrap'].append(num_wrap)
    features['autocorr'].append(auto_corr)
   
    features['b'].append(b)
    features['fd'].append(fd)
    features['fr'].append(fr)
    
    seg_len.append(len(ids))


def compute_segment_feature(ids, times, sampling_rate, MAX):
    id_segment = []
    time_segment = []
    features = {
        'num_wrap': [],
        'autocorr': [],
        
        'b': [],
        'fd': [],
        'fr': []
    }
    seg_len = []
    for i in range(len(ids)):
        if ids[i] == -1:
            if len(id_segment) >= 20:

                segment_features(id_segment, time_segment,
                                 features, seg_len, sampling_rate, MAX)
            id_segment = []
            time_segment = []
            continue
        id_segment.append(ids[i])
        # time_segment.append(times[i])

    if len(id_segment) >= 20:  # for the sequence without packet loss
        segment_features(id_segment, time_segment, features,
                         seg_len, sampling_rate, MAX)

    return features, seg_len


def spectrum(ids, sampling_rate):
    data = np.array(ids, dtype=float)
    data[:] = data - np.mean(data)  # remove zero freqeuncy
    fourier_transform = np.fft.fft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    N = len(data)

    frequency = np.linspace(0, sampling_rate, N)
    fm = frequency[-1]
    # print(fm)
    fqs = frequency[:N//2+1]/sampling_rate
    ps = power_spectrum[:N//2+1]
    max_index = np.argmax(ps)
    fd = round(fqs[max_index], 3)
    # /sampling_rate # normalization of fft
    print(fd)
    return fqs, ps


def frequency_features(ids, sampling_rate):
    data = np.array(ids, dtype=float)
    data[:] = data - np.mean(data)  # remove zero freqeuncy
    fourier_transform = np.fft.fft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    N = len(data)

    frequency = np.linspace(0, sampling_rate, N)
    fm = frequency[-1]
    # print(fm)
    #fqs = frequency[:N]
    #ps = power_spectrum[:N]
    fqs = frequency[:N//2+1]/sampling_rate  # normalization of fft
    # print(fqs)
    ps = power_spectrum[:N//2+1]
    max_index = np.argmax(ps)
    fd = round(fqs[max_index], 3)
    b = power_bandwidth(data, 1)
    fmg = abs_fourier_transform[:N//2+1]
    fc = spectral_centroid(fmg, fqs)
    fr = spectral_roll_off(fmg, fqs)
    return fqs, ps
    
    
def adjust(arr):
    narr = []
    for d in arr:
        if d != -1:
            narr.append(d)
    return narr


def enp(data):
    data = adjust(data)
    value, counts = np.unique(data, return_counts=True)
    return entropy(counts, base=2)


def feature_extraction():
    sampling_rates = [1]

    
    classes = ['global', 'perConn', 'random_or_others']
    
    
    MAX = 4294967296
    elps = []
    count = 0
    for sampling_rate in sampling_rates:
    
            dataset = {
                'ip': [],
                
                'num_wrap': [],
                'autocorr': [],
               
                'b': [],
                'fd': [],
                'fr': [],
                'pr_s': [],
                'pr_x': [],
                'pr_y': [],
                'class': []
            }
            
            
            for c in classes: 
                
                with open('../SoK/Datasets/IPv6/'+c+'.data', 'r') as filehandle:
                    filecontents = filehandle.readlines()
                    for line in filecontents:
                        fields = line.split(",")
                        if len(fields) < 2:
                            continue
                        ip = fields[0]
                        s = extract(fields[1])
                        if len(s) < 100: continue
                        
                       # NOTE: for constant IPID values
                        if s[0] == s[1] and s[1] == s[-1]:
                            count = count + 1
                            continue
                       
                        start = time.monotonic()
                        
                        x, y = split_s(s)

                        times = list(range(0, l))

                        s_entropy = enp(s)

                        features, seg_len = compute_segment_feature(
                            s, times, sampling_rate, MAX)
                        if len(features.get('num_wrap')) == 0:
                            continue

                        num_wrap = np.median(features.get('num_wrap'))

                        auto_corr = np.median(features.get('autocorr'))
                        #cross_corr = np.median(features.get('crosscorr'))

                        b = np.median(features.get('b'))
                        fd = np.median(features.get('fd'))
                        fr = np.median(features.get('fr'))

                        results = []
                        smapes, xsmapes, ysmapes = [], [], []
                        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                            future1 = executor.submit(
                                ipidtsf.calculate_pred_error, s[:35], 5, None, MAX, smapes)
                            future2 = executor.submit(
                                ipidtsf.calculate_pred_error, x[:35], 5, None, MAX, xsmapes)
                            future3 = executor.submit(
                                ipidtsf.calculate_pred_error, y[:35], 5, None, MAX, ysmapes)

                        for future in concurrent.futures.as_completed([future1, future2, future3]):
                            results.append(future.result())

                        pr_s = np.median(smapes)
                        pr_x = np.median(xsmapes)
                        pr_y = np.median(ysmapes)
                        
                        end = time.monotonic()
                        elps.append(end-start)
                        
                        write_to_dic(dataset, c, ip=ip,  s_entropy=s_entropy, num_wrap=num_wrap,
                                     auto_corr=auto_corr, fd=fd, b=b, fr=fr, pr_s=pr_s, pr_x=pr_x, pr_y=pr_y)
                        
        df = pd.DataFrame(dataset)
       
        df.to_csv('./Datasets/topov6_ipid_data_smape_8f.csv', index=False)
       
    print("The number of constant IPIDs: ", count)
    
    


def main():
   
    feature_extraction()

if __name__ == "__main__":
    # execute only if run as a script
    main()
