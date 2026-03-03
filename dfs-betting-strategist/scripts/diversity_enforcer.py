
#!/usr/bin/env python3
"""
Enforce minimum unique player constraints between all lineup pairs.
Usage: python diversity_enforcer.py --pool best150.csv --all-pools pool1.csv pool2.csv ... --proj proj.csv --min-unique 2 --out best150_diverse.csv
"""
import argparse
import pandas as pd
from collections import Counter
from itertools import combinations

DRIVER_COLS = ["D","D.1","D.2","D.3","D.4","D.5"]

def load_pool(path, id_name):
    lu = pd.read_csv(path)
    lu["key"] = lu.apply(lambda r: frozenset(str(int(r[c])) for c in DRIVER_COLS), axis=1)
    lu["drivers"] = lu.apply(lambda r: [id_name.get(str(int(r[c])), str(r[c])) for c in DRIVER_COLS], axis=1)
    lu["dset"] = lu["drivers"].apply(set)
    return lu

def get_violations(df, min_unique):
    max_overlap = 6 - min_unique
    v = []
    rows = list(df.iterrows())
    for (i, r1), (j, r2) in combinations(rows, 2):
        if len(r1["dset"] & r2["dset"]) > max_overlap:
            v.append((i, j))
    return v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, help="Input pool CSV (best N lineups)")
    parser.add_argument("--all-pools", nargs="+", help="All source pools for reserve candidates")
    parser.add_argument("--proj", required=True)
    parser.add_argument("--min-unique", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--out", default="pool_diverse.csv")
    args = parser.parse_args()

    proj = pd.read_csv(args.proj)
    id_name = dict(zip(proj["DFS ID"].astype(str), proj["Name"]))

    working = load_pool(args.pool, id_name)
    if "Sim Optimals" in working.columns:
        working["score"] = working["Sim Optimals"] * 10 + working["99th"] / 30
    else:
        working["score"] = working["99th"] / 30

    # Build reserve from all pools
    reserve = None
    if args.all_pools:
        all_pools = [load_pool(p, id_name) for p in args.all_pools]
        reserve = pd.concat(all_pools).drop_duplicates(subset=["key"])
        reserve = reserve[~reserve["key"].isin(set(working["key"]))]
        if "Sim Optimals" in reserve.columns:
            reserve["score"] = reserve["Sim Optimals"] * 10 + reserve["99th"] / 30
        reserve = reserve.sort_values("score", ascending=False).reset_index(drop=True)

    v_before = get_violations(working, args.min_unique)
    print(f"Violations before: {len(v_before)}")

    for iteration in range(args.max_iter):
        violations = get_violations(working, args.min_unique)
        if not violations:
            print(f"All violations resolved after {iteration} iterations")
            break
        vc = Counter()
        for i, j in violations:
            vc[i] += 1
            vc[j] += 1
        worst_idx = vc.most_common(1)[0][0]

        replaced = False
        if reserve is not None:
            test_pool = working.drop(index=worst_idx).reset_index(drop=True)
            for _, candidate in reserve.iterrows():
                if candidate["key"] in set(working["key"]):
                    continue
                new_v = sum(1 for _, r in test_pool.iterrows()
                           if len(r["dset"] & candidate["dset"]) > (6 - args.min_unique))
                if new_v == 0:
                    working = pd.concat([test_pool, candidate.to_frame().T]).reset_index(drop=True)
                    replaced = True
                    break
        if not replaced:
            working = working.drop(index=worst_idx).reset_index(drop=True)

    v_after = get_violations(working, args.min_unique)
    v3 = get_violations(working, 3)
    print(f"Violations after ({args.min_unique}-unique): {len(v_after)}")
    print(f"<3-unique pairs remaining: {len(v3)}")
    print(f"Final pool: {len(working)} lineups")
    print(f"Own mean: {working["Ownership"].mean():.1f}% | SimOpt: {working["Sim Optimals"].mean():.2f}")

    working[DRIVER_COLS].to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
