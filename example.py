"""stats-kit example on synthetic data: one real effect, one that only exists in one period, one that is noise.
python example.py"""
import numpy as np, pandas as pd
from stats_kit import check, check_fe, table, wilson

rng = np.random.default_rng(1)
rows = []
for sym in ('NQ', 'ES', 'RTY'):
    for day in pd.bdate_range('2018-01-01', '2026-06-30'):
        for _ in range(4):                                   # several setups a day: clustering matters
            signal = rng.integers(0, 2)
            era = rng.integers(0, 2)
            bucket = rng.integers(0, 4)                      # e.g. where the candle closed (a control)
            late = day.year >= 2025
            p = 0.30 + 0.10 * bucket + 0.05 * signal + (0.08 * era if not late else 0.0)
            rows.append(dict(date=day, sym=sym, half='18-24' if not late else '25-26',
                             signal=signal, era=era, noise=rng.integers(0, 2), bucket=bucket,
                             hit=float(rng.random() < p)))
df = pd.DataFrame(rows).set_index('date')

res = [check(df, 'hit ~ signal + C(bucket)', 'signal', test='real effect (+5 points)'),
       check(df, 'hit ~ era + C(bucket)', 'era', test='only in 2018-24'),
       check(df, 'hit ~ era + C(bucket)', 'era', test='only in 2018-24, strict', period_p=0.10),
       check(df, 'hit ~ noise + C(bucket)', 'noise', test='noise'),
       check_fe(df, 'hit', 'signal', ['bucket'], test='real effect, within bucket')]
print(table(res).to_string())
k, n = int(df.hit.sum()), len(df)
lo, hi = wilson(k, n)
print(f'\noverall hit rate {k / n:.3f}  (95% Wilson {lo:.3f}–{hi:.3f})')
