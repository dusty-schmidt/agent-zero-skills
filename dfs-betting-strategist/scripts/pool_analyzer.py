
#!/usr/bin/env python3
"""
Analyze a single lineup pool: ownership distribution, win pool, driver exposure, quality metrics.
Usage: python pool_analyzer.py --pool lineups.csv --proj proj.csv
"""
import argparse
import pandas as pd
from collections import Counter

DRIVER_COLS = ["D","D.1","D.2","D.3","D.4","D.5"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--proj", required=True)
    args = parser.parse_args()

    proj = pd.read_csv(args.proj)
    id_name = dict(zip(proj["DFS ID"].astype(str), proj["Name"]))
    id_own = dict(zip(proj["Name"], proj["My Own"]))

    pool = pd.read_csv(args.pool)
    pool["drivers"] = pool.apply(lambda r: [id_name.get(str(int(r[c])), str(r[c])) for c in DRIVER_COLS], axis=1)
    own = pool["Ownership"]
    p25 = own.quantile(0.25)
    wp = pool[own <= p25]

    print(f"=== POOL STATS ===")
    print(f"Total lineups:  {len(pool)}")
    print(f"Own mean:       {own.mean():.1f}%")
    print(f"Own median:     {own.median():.1f}%")
    print(f"Own std:        {own.std():.1f}%")
    print(f"Own range:      {own.min():.0f}-{own.max():.0f}%")
    print(f"Win cutoff p25: {p25:.1f}%")
    print(f"Win pool size:  {len(wp)}")
    print(f"99th mean:      {pool["99th"].mean():.1f}")
    print(f"SimOpt mean:    {pool["Sim Optimals"].mean():.2f}")
    print(f"Win pool 99th:  {wp["99th"].mean():.1f}")

    counter = Counter([d for lst in pool["drivers"] for d in lst])
    print(f"
=== DRIVER EXPOSURE ===")
    print(f"{'Driver':<25} {'MyExp':>7} {'ProjOwn':>8} {'Leverage':>9}")
    for d, cnt in counter.most_common():
        me = cnt/len(pool)*100
        po = id_own.get(d, 0)
        print(f"  {d:<23} {me:>6.1f}% {po:>7.1f}% {me-po:>+8.1f}%")

if __name__ == "__main__":
    main()
