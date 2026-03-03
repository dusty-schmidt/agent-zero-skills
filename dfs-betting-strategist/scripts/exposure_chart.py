
#!/usr/bin/env python3
"""
Generate driver exposure chart: my lineups vs projected field ownership.
Usage: python exposure_chart.py --pool best150.csv --proj proj.csv --out chart.png
"""
import argparse
import pandas as pd
import numpy as np
from collections import Counter

DRIVER_COLS = ["D","D.1","D.2","D.3","D.4","D.5"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--proj", required=True)
    parser.add_argument("--out", default="exposure_chart.png")
    parser.add_argument("--title", default="Driver Exposure: My Lineups vs Projected Field")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj = pd.read_csv(args.proj)
    id_name = dict(zip(proj["DFS ID"].astype(str), proj["Name"]))
    id_own = dict(zip(proj["Name"], proj["My Own"]))

    pool = pd.read_csv(args.pool)
    pool["drivers"] = pool.apply(lambda r: [id_name.get(str(int(r[c])), str(r[c])) for c in DRIVER_COLS], axis=1)
    counter = Counter([d for lst in pool["drivers"] for d in lst])
    n = len(pool)

    df = pd.DataFrame([
        {"Driver": d, "My Exposure": counter.get(d,0)/n*100, "Proj Own": id_own.get(d,0)}
        for d in proj["Name"].tolist()
    ]).sort_values("My Exposure", ascending=True)

    fig, ax = plt.subplots(figsize=(13, max(8, len(df)*0.32)))
    bg, fg, c1, c2 = "#1a1a2e", "#e0e0e0", "#00d4ff", "#ff6b35"
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    y, h = np.arange(len(df)), 0.38
    ax.barh(y+h/2, df["My Exposure"], h, label="My Exposure %", color=c1, alpha=0.9)
    ax.barh(y-h/2, df["Proj Own"], h, label="Proj Field Own %", color=c2, alpha=0.75)
    ax.set_yticks(y); ax.set_yticklabels(df["Driver"], color=fg, fontsize=8)
    ax.set_xlabel("Ownership %", color=fg)
    ax.set_title(f"{args.title}
{n} Entries", color=fg, fontweight="bold", pad=12)
    ax.tick_params(colors=fg)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#444")
    ax.grid(axis="x", color="#333", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", facecolor="#2a2a3e", edgecolor="#444", labelcolor=fg)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=bg)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
