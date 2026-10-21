# stats-kit

**The test rule every number on the Hueco Mundo indicators had to pass before it was shown**, as a small Python helper.

An effect counts (**✓**) only when all three hold:
1. **pooled p < 0.05 with standard errors clustered by date.** Several setups on one day are not independent evidence, and plain standard errors would count them as if they were.
2. **the same sign in each period** (for example 2018–24 and 2025–26), so it is not one era.
3. **the same sign on each symbol** (for example NQ, ES and RTY), so it is not one market.

Most claims tested this way did not survive. That was the point.

## Use

```python
from stats_kit import check, check_fe, table, wilson

rows = [
    check(df, 'hit ~ signal + C(bucket)', 'signal', test='signal, same bucket'),          # linear probability model
    check(df, 'hit ~ signal + C(bucket)', 'signal', kind='logit', test='signal (logit)'),
    check_fe(df, 'held', 'protected', ['bars_left', 'dist_decile'], test='protected vs other swings'),
]
print(table(rows))
lo, hi = wilson(k=412, n=600)          # 95% Wilson interval for a hit rate
```

- `df` needs the date as its index (or pass `date_col=`), a period column (`period_col='half'`) and a symbol column (`group_col='sym'`).
- `check` fits any [patsy](https://patsy.readthedocs.io) formula.
- `check_fe` is the same rule for a within-cell (fixed-effects) slope. Use it when a full dummy matrix would not fit in memory, e.g. "same bars left × same distance decile" on 650k rows.
- `table` lines the results up, with each period's and each symbol's coefficient next to the pooled one.

## The rule's weak spot

"Same sign" does not mean "significant in each period". In `example.py`, an effect that exists only in 2018–24 passes, because its 2025–26 coefficient is +0.001 (p 0.93). Read the per-period p in the table, or pass `period_p=0.10` to require each period to stand on its own:

```text
test                       coef       p   18-24  p_18-24   25-26  p_25-26  ok
real effect (+5 points)  0.0417  0.0000  0.0403   0.0000  0.0495   0.0004  ✓
only in 2018-24          0.0603  0.0000  0.0729   0.0000  0.0013   0.9281  ✓
only in 2018-24, strict  0.0603  0.0000  0.0729   0.0000  0.0013   0.9281
noise                    0.0050  0.3971  0.0071   0.2757 -0.0035   0.7984
```

`python example.py` reproduces it on synthetic data: one real effect, one that only exists in one period, and one that is noise.

Requires Python 3.9+, numpy, pandas, statsmodels and patsy.

## Licence

Mozilla Public License 2.0, see [LICENSE](LICENSE). © Unclehk2000.
