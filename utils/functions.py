import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, bootstrap


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


def running_mean(x, N=21):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def df_calculate_pvalues(df, method='pearson'):
    """
    Функция вычисления p-value для корреляций DataFrame'а df.
    Поддерживает method pearson и spearman
    """
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            if method == 'pearson':
                pvalues[r][c] = dropna_pearsonr(df[r], df[c])[1]
            elif method == 'spearman':
                pvalues[r][c] = dropna_spearmanr(df[r], df[c])[1]
            else:
                raise Exception('Wrong correlation method!')
    return pvalues


def df_calculate_bootstrap_corr(df, method='spearman'):
    """
    Функция вычисления bootstrap корреляцию для DataFrame'а df.
    Поддерживает method pearson и spearman
    """
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            if method == 'spearman':
                def get_corr(x, y):
                    return dropna_spearmanr(x, y)[0]
            elif method == 'pearson':
                def get_corr(x, y):
                    return dropna_spearmanr(x, y)[0]
            else:
                raise Exception('Wrong correlation method!')
            
            if r != c:
                res = bootstrap((df[r], df[c]), get_corr, vectorized=False, paired=True, random_state=1, n_resamples=1000)
                low, high = res.confidence_interval
                se = res.standard_error
                #pvalues[r][c] = f'[{low:.2f}; {high:.2f}]\n(se={se:.2f})'
                pvalues[r][c] = f'{low+(high-low)/2:.2f}'
            else:
                pvalues[r][c] = np.nan
    return pvalues
