
#!/usr/bin/env python3
"""
Generate DraftKings bulk upload CSV from finalized lineup pool.
Usage: python dk_entry_builder.py --pool best150_diverse.csv --template DKEntries.csv --proj proj.csv --out DKEntries_filled.csv
"""
import argparse
import pandas as pd

DRIVER_COLS = ["D","D.1","D.2","D.3","D.4","D.5"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--template", required=True, help="DK entries template CSV")
    parser.add_argument("--proj", required=True)
    parser.add_argument("--out", default="DKEntries_filled.csv")
    args = parser.parse_args()

    proj = pd.read_csv(args.proj)
    id_to_nameid = {str(int(row["DFS ID"])): f"{row["Name"]} ({int(row["DFS ID"])})" for _, row in proj.iterrows()}

    template = pd.read_csv(args.template, header=0, usecols=[0,1,2,3,4,5,6,7,8,9], on_bad_lines="skip")
    template = template[pd.to_numeric(template.iloc[:,0], errors="coerce").notna()].reset_index(drop=True)
    template.columns = ["Entry ID","Contest Name","Contest ID","Entry Fee","D1","D2","D3","D4","D5","D6"]

    pool = pd.read_csv(args.pool)
    n = min(len(pool), len(template))

    for i in range(n):
        drivers = [id_to_nameid.get(str(int(pool.iloc[i][c])), str(int(pool.iloc[i][c]))) for c in DRIVER_COLS]
        for j, col in enumerate(["D1","D2","D3","D4","D5","D6"]):
            template.at[i, col] = drivers[j]

    with open(args.out, "w") as f:
        f.write("Entry ID,Contest Name,Contest ID,Entry Fee,D,D,D,D,D,D
")
        for _, row in template.head(n).iterrows():
            f.write(f"{int(row["Entry ID"])},{row["Contest Name"]},{int(row["Contest ID"])},{row["Entry Fee"]},{row["D1"]},{row["D2"]},{row["D3"]},{row["D4"]},{row["D5"]},{row["D6"]}
")

    print(f"Filled {n} entries -> {args.out}")

if __name__ == "__main__":
    main()
