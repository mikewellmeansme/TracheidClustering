import numpy as np
from scipy.stats import pearsonr, spearmanr

def dropna(x,y):
    x, y = np.array(x), np.array(y)
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    x, y = x[~nas], y[~nas]
    return x, y

def dropna_pearsonr(x, y):
    x, y = dropna(x,y)
    r, p = pearsonr(x, y)
    return r, p

def dropna_spearmanr(x, y):
    x, y = dropna(x,y)
    r, p = spearmanr(x, y)
    return r, p

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
