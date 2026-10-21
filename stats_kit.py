"""stats-kit: the test rule every number on the indicators had to pass before it was shown.

An effect counts (✓) only when all three hold:
  1. pooled p < 0.05, with standard errors clustered by date (many rows on one day are not independent evidence);
  2. the same sign in each period (for example 2018–24 and 2025–26), so it is not one era;
  3. the same sign on each symbol (for example NQ, ES and RTY), so it is not one market.

  from stats_kit import check, check_fe, table, wilson
  rows = [check(df, 'hit ~ signal + C(bucket)', 'signal', test='signal, same bucket')]
  print(table(rows))

`df` needs a date (index or column), a period column and a symbol column (names are arguments).
Requires numpy, pandas, statsmodels, patsy.
"""
import math
import numpy as np, pandas as pd
import patsy
import statsmodels.api as sm

__all__ = ['cfit', 'check', 'check_fe', 'table', 'wilson']


def _dates(df, date_col):
    return df.index.to_numpy() if date_col is None else df[date_col].to_numpy()


def cfit(df, formula, kind='ols', date_col=None):
    """Fit `formula` (patsy syntax) with errors clustered by date. kind: 'ols' (linear probability model / OLS)
    or 'logit'. Rows with NaN in the used columns are dropped first."""
    d = df.copy()
    d['_date'] = _dates(df, date_col)
    d = d.reset_index(drop=True)
    y, X = patsy.dmatrices(formula, d, return_type='dataframe', NA_action='drop')
    g = pd.factorize(d.loc[y.index, '_date'])[0]
    if kind == 'ols':
        return sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': g})
    return sm.Logit(y, X).fit(cov_type='cluster', cov_kwds={'groups': g}, disp=0)


def check(df, formula, term, kind='ols', period_col='half', group_col='sym', date_col=None, test=None, period_p=None):
    """Pooled coefficient and p for `term`, then the coefficient in each period and each group.
    Returns a dict; 'ok' is '✓' when the rule passes. period_p (e.g. 0.10) also requires each period's own p
    below it: same sign alone lets a tiny, insignificant coefficient in one period through."""
    r = cfit(df, formula, kind, date_col)
    out = {'test': test or term, 'coef': float(r.params[term]), 'p': float(r.pvalues[term]), 'n': int(r.nobs)}
    signs = [np.sign(out['coef'])]
    strict = True
    for col in (period_col, group_col):
        if col is None:
            continue
        for v, g in df.groupby(col):
            rr = cfit(g, formula, kind, date_col)
            out[str(v)] = float(rr.params[term]); out['p_' + str(v)] = float(rr.pvalues[term])
            signs.append(np.sign(rr.params[term]))
            if col == period_col and period_p is not None and rr.pvalues[term] >= period_p:
                strict = False
    out['ok'] = '✓' if out['p'] < 0.05 and len(set(signs)) == 1 and strict else ''
    return out


def _fe(d, y, x, cells, dates):
    g = d.groupby(cells, observed=True)
    yd = d[y] - g[y].transform('mean')
    xd = d[x] - g[x].transform('mean')
    m = sm.OLS(yd.to_numpy(), xd.to_numpy()[:, None]).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(dates)[0]})
    return float(m.params[0]), float(m.pvalues[0])


def check_fe(df, y, x, cells, period_col='half', group_col='sym', date_col=None, test=None):
    """Within-cell slope of y on x (fixed effects for every combination of `cells`), errors clustered by date.
    Use it when a full dummy matrix would be too big for memory, e.g. 'same bars left × distance decile'."""
    b, p = _fe(df, y, x, cells, _dates(df, date_col))
    out = {'test': test or f'{x} | {"×".join(cells)}', 'coef': b, 'p': p, 'n': len(df)}
    signs = [np.sign(b)]
    for col in (period_col, group_col):
        if col is None:
            continue
        for v, g in df.groupby(col):
            bb, pp = _fe(g, y, x, cells, _dates(g, date_col))
            out[str(v)] = bb; out['p_' + str(v)] = pp; signs.append(np.sign(bb))
    out['ok'] = '✓' if p < 0.05 and len(set(signs)) == 1 else ''
    return out


def table(rows, digits=4):
    """Rows from check / check_fe as one DataFrame, test names as the index."""
    T = pd.DataFrame(rows).set_index('test')
    first = [c for c in ('coef', 'p', 'n') if c in T.columns]
    rest = [c for c in T.columns if c not in first + ['ok']]
    return T[first + rest + ['ok']].round(digits)


def wilson(k, n, z=1.96):
    """Wilson score interval for k hits in n tries (95% by default)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)
