import pandas as pd
import statsmodels.api as sm
import scipy
import pmdarima as pm
import numpy as np
import time
import statistics

file_paht = ''
result_path = ''
p, d, q = 1, 0, 0
threshold_single = 1.645
threshold_double = 1.645*2


def check_spick(data):

    pre_spike = data[:10]
    post_spike = data[10:]

    mod = sm.tsa.arima.ARIMA(pre_spike, order=(p, d, q))
    res = mod.fit()
    steps = len(post_spike)
    predicts = res.forecast(steps=steps)
    print(post_spike, predicts)
    a = []
    for i in range(0, len(post_spike)):
        a.append(post_spike[i] - predicts[i])

    zscores = scipy.stats.zscore(a, axis=0, ddof=0, nan_policy='propagate')
    print(a, zscores)
    spike = []
    spike_count = 0
    for i in range(0, len(zscores)):
        if zscores[i] > threshold_double:
            spike.append([i, 2])
            spike_count = spike_count + 2
        elif zscores[i] > threshold_single:
            spike.append([i, 1])
            spike_count = spike_count + 1

    return [spike, spike_count]


def grid_search(pre_spike):
    p_range = range(0, 2)
    d_range = range(0, 2)
    q_range = range(0, 2)

    # Perform grid search
    best_model = None
    best_aic = float("inf")
    best_p, best_d, best_q = 1, 0, 0

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = pm.ARIMA(order=(p, d, q))
                    model.fit(pre_spike)
                    aic = model.aic()

                    if aic < best_aic:
                        best_aic = aic
                        best_model = model
                        best_p = p
                        best_d = d
                        best_q = q

                except:
                    continue

    return best_p, best_d, best_q


def our_revision_check_spick(data):

    pre_spike = data[:10]
    post_spike = data[10:]

    ar = pm.auto_arima(pre_spike, start_p=0, start_q=0,

                       max_p=3, max_q=3,  # maximum p and q

                       d=1,           # let model determine 'd'

                       suppress_warnings=True,
                       stepwise=True)
    order = ar.order
    mod = sm.tsa.arima.ARIMA(pre_spike, order=order)
    res = mod.fit()
    steps = len(post_spike)
    predicts = res.forecast(steps=steps)
    print(post_spike, predicts)

    a = []
    for i in range(0, len(post_spike)):
        a.append(post_spike[i] - predicts[i])

    zscores = scipy.stats.zscore(a, axis=0, ddof=0, nan_policy='propagate')
    print(a, zscores)
    spike = []
    spike_count = 0
    for i in range(0, len(zscores)):
        if zscores[i] > threshold_double:
            spike.append([i, 2])
            spike_count = spike_count + 2
        elif zscores[i] > threshold_single:
            spike.append([i, 1])
            spike_count = spike_count + 1

    return [spike, spike_count]


def calculate_pred_error(ip, sequence, valid_length, l):
    history, actual = sequence[0:l], sequence[l:]
    chps_ind = list()
    predictions = list()
    errs = list()
    elps = list()
    for i in range(len(actual)+1):

        if i == len(actual):
            break
        start = time.monotonic()
        ar = pm.auto_arima(history, start_p=0, start_q=0,

                           max_p=3, max_q=3,  # maximum p and q

                           d=1,           # let model determine 'd'

                           suppress_warnings=True,
                           stepwise=True)
        order = ar.order
        mod = sm.tsa.arima.ARIMA(history, order=order)
        res = mod.fit()
        predicts = res.forecast(step=1)
        predictions.append(predicts[0])
        end = time.monotonic()
        elps.append(end-start)

        history.append(actual[i])
        history.pop(0)
    # identify change points and then eliminate the error from the transformation at the restore.

    for i in range(0, len(actual)):
        errs.append(predictions[i]-actual[i])

    errs = errs[:valid_length]
    u = np.mean(errs)
    s = np.std(errs)
    t = statistics.mean(elps)
    
    return ip, u, s, predictions, actual, t


def check_type(result):
    spike, spike_count = result
    if spike_count > 3:
        spike_type = 'misclass'
    elif spike_count == 1:
        spike_type = 'inbound'
    elif spike_count > 1:
        spike_type = 'outbound'
    else:
        spike_type = 'no'
    return spike_type


def spike2string(result):
    spike, spike_count = result
    results = ''
    for i in range(0, len(spike)):
        results = results + str(spike[i][0]) + ':' + \
            str(spike[i][1]) + ':' + str(spike_count)
    return results


def parse_ipid(data):
    # time, asn, ip, port,
    data['ids'] = data['ids'].apply(lambda x: x.split(' '))
    data['spike'] = data['ids'].apply(check_spick)
    data['type'] = data['spike'].apply(check_type)
    data['spike'] = data['spike'].apply(spike2string)
    data = data.to_csv(result_path, index=False)
    return data
