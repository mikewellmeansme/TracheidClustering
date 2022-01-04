import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    bootstrap,
    t
)
from .functions import dropna


def dropna_pearsonr(x, y):
    x, y = dropna(x,y)
    r, p = pearsonr(x, y)
    return r, p


def dropna_spearmanr(x, y):
    x, y = dropna(x,y)
    r, p = spearmanr(x, y)
    return r, p


def get_t_stat(r, n):
    return (r * np.sqrt(n - 2)) / (np.sqrt(1 - r ** 2))


def get_p_value(r, n):
    r = abs(r)
    t_stat = get_t_stat(r, n)
    return t.sf(t_stat, n-2)*2


def get_df_corr(df, method='pearson'):
    """
    Функция вычисления корреляций и p-value для DataFrame'а df.
    Поддерживает method pearson и spearman
    """
    dfcols = pd.DataFrame(columns=df.columns)
    result = dfcols.transpose().join(dfcols, how='outer')

    if method == 'pearson':
        get_corr = dropna_spearmanr
    elif method == 'spearman':
        get_corr = dropna_spearmanr
    else:
        raise Exception('Wrong correlation method!')
    
    for c1 in df.columns:
        for c2 in df.columns:
            r, p = get_corr(df[c1], df[c2])
            result[c1][c2] = f'{r:.2f}\n(p={p:.3f})'
    
    return result


def df_calculate_bootstrap_corr(df, method='spearman'):
    """
    Функция вычисления bootstrap корреляцию для DataFrame'а df.
    Поддерживает method pearson и spearman
    """
    dfcols = pd.DataFrame(columns=df.columns)
    result = dfcols.transpose().join(dfcols, how='outer')

    if method == 'spearman':
        def get_corr(x, y):
            return dropna_spearmanr(x, y)[0]
    elif method == 'pearson':
        def get_corr(x, y):
            return dropna_spearmanr(x, y)[0]
    else:
        raise Exception('Wrong correlation method!')

    for c1 in df.columns:
        for c2 in df.columns:
            if c1 != c2:
                res = bootstrap((df[c1], df[c2]), get_corr, vectorized=False, paired=True, random_state=1, n_resamples=1000)
                low, high = res.confidence_interval
                se = res.standard_error
                #result[c1][c2] = f'[{low:.2f}; {high:.2f}]\n(se={se:.2f})'
                r = low+(high-low)/2
                n = len(dropna(df[c1], df[c2])[0])
                p = get_p_value(r, n)
                result[c1][c2] = f'{r:.2f}\n(p={p:.3f})'
            else:
                result[c1][c2] = np.nan
    return result
