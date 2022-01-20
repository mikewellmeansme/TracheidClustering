import numpy as np
import pandas as pd


def dropna(x,y):
    x, y = np.array(x), np.array(y)
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    x, y = x[~nas], y[~nas]
    return x, y


def get_normalized_list(x: list, norm : int):
    """
    Функция получения нормированного списка
    :param x: список для нормирования
    :param norm: норма
    :return: l_norm - нормированный к e список l
    """
    l_raw = []  # промежуточный список
    n = len(x)
    for i in range(n):
        for j in range(norm):
            l_raw += [x[i]]
    l_norm = []
    for i in range(norm):
        l_norm += [1 / n * sum([l_raw[j] for j in range(n * i, n * (i + 1))])]
    return l_norm


def running_mean(x, N=21):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def get_moving_avg(df, window=7, nanmean=False):
    """
    Возвращает скользящее средние для температуры
    window : окно
    nanmean : используем nanmean для сглаживания? (тогда потеряются данные по краям)
    """
    month = df['Month']
    if 'Week' in df.columns:
        week = df['Week']
    day = df['Day']
    if nanmean:
        result = df.rolling(window=window, center=True).apply(np.nanmean)
    else:
        result = df.rolling(window=window, center=True, min_periods=1).mean()
    result['Month'] = month
    if 'Week' in df.columns:
        result['Week'] = week
    result['Day'] = day
    return result

def get_moving_sum(df, window=7):
    """
    Возвращает скользящую сумму для осадков
    window : окно
    """
    month = df['Month']
    day = df['Day']
    result = df.rolling(window=window, center=True, min_periods=1).sum()
    result['Month'] = month
    result['Day'] = day
    return result

def list_to_cumulative(lst):
    return [np.nansum(lst[:i+1]) for i in range(len(lst))]
