
#!/usr/bin/env python3
"""
Multi-run pool combiner + best-N extractor.
Usage: python multi_run_combiner.py --pools file1.csv file2.csv ... --proj projections.csv --n 150 --out best150.csv
"""
import argparse
import pandas as pd
from collections import Counter

DRIVER_COLS = ["D","D.1","D.2","D.3","D.4","D.5"]

def load_pool(path, id_name):
    lu = pd.read_csv(path)
    lu["key"] = lu.apply(lambda r: frozenset(str(int(r[c])) for c in DRIVER_COLS), axis=1)
    lu["drivers"] = lu.apply(lambda r: [id_name.get(str(int(r[c])), str(r[c])) for c in DRIVER_COLS], axis=1)
    return lu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", nargs="+", required=True)
    parser.add_argument("--proj", required=True, help="DK projections CSV with DFS ID, Name, My Own")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--out", default="best_lineups.csv")
    parser.add_argument("--t1", type=float, default=0.25, help="Tier1 fraction (win shots)")
    parser.add_argument("--t2", type=float, default=0.45, help="Tier2 fraction (core)")
    args = parser.parse_args()

    proj = pd.read_csv(args.proj)
    id_name = dict(zip(proj["DFS ID"].astype(str), proj["Name"]))

    all_pools = []
    seen_keys = set()
    for i, path in enumerate(args.pools):
        pool = load_pool(path, id_name)
        new = pool[~pool["key"].isin(seen_keys)]
        new_rate = len(new)/len(pool)*100
        print(f"Pool {i+1} ({path.split('//'[-1]}): {len(pool)} total | {len(new)} new ({new_rate:.0f}%)")
        seen_keys |= set(new["key"])
        all_pools.append(new)

    combined = pd.concat(all_pools).reset_index(drop=True)
    combined["score"] = combined["Sim Optimals"] * 10 + combined["99th"] / 30
    print(f"
Combined unique pool: {len(combined)} lineups")
    print(f"Own mean: {combined["Ownership"].mean():.1f}% | 99th mean: {combined["99th"].mean():.1f} | SimOpt: {combined["Sim Optimals"].mean():.2f}")

    # Tiered extraction
    p25 = combined["Ownership"].quantile(0.25)
    p50 = combined["Ownership"].quantile(0.50)
    n1 = int(args.n * args.t1)
    n2 = int(args.n * args.t2)
    n3 = args.n - n1 - n2

    t1 = combined[combined["Ownership"] <= p25].sort_values("score", ascending=False).head(n1)
    t2 = combined[(combined["Ownership"] > p25) & (combined["Ownership"] <= p50)].sort_values("score", ascending=False).head(n2)
    t3 = combined[combined["Ownership"] > p50].sort_values("score", ascending=False).head(n3)

    best = pd.concat([t1, t2, t3]).reset_index(drop=True)
    print(f"
Extracted {len(best)} lineups:")
    print(f"  Tier1 (<=p25={p25:.0f}%): {len(t1)} | SimOpt={t1["Sim Optimals"].mean():.2f} | 99th={t1["99th"].mean():.1f}")
    print(f"  Tier2 (p25-p50={p50:.0f}%): {len(t2)} | SimOpt={t2["Sim Optimals"].mean():.2f} | 99th={t2["99th"].mean():.1f}")
    print(f"  Tier3 (>p50): {len(t3)} | SimOpt={t3["Sim Optimals"].mean():.2f} | 99th={t3["99th"].mean():.1f}")

    best[DRIVER_COLS + ["Ownership","99th","Sim Optimals","Proj Score"]].to_csv(args.out, index=False)
    print(f"
Saved: {args.out}")

if __name__ == "__main__":
    main()
