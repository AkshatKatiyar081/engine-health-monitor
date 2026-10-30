"""Convert the raw C-MAPSS .txt files to headered CSVs.

Train and test CSVs get an extra `rul` column:
  train: max_cycle(unit) - cycle
  test : true final RUL (from RUL_FDxxx.txt) + max_cycle(unit) - cycle
"""
import sys
from pathlib import Path
import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "DataSet"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "data" / "csv"
COLS = ["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]

OUT.mkdir(parents=True, exist_ok=True)
for fd in ("FD001", "FD002", "FD003", "FD004"):
    train = pd.read_csv(RAW / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(RAW / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    final = pd.read_csv(RAW / f"RUL_{fd}.txt", header=None, names=["rul"])
    final.insert(0, "unit", range(1, len(final) + 1))

    train["rul"] = train.groupby("unit").cycle.transform("max") - train.cycle
    last = test.groupby("unit").cycle.transform("max")
    test["rul"] = test.unit.map(final.set_index("unit").rul) + last - test.cycle

    train.to_csv(OUT / f"train_{fd}.csv", index=False)
    test.to_csv(OUT / f"test_{fd}.csv", index=False)
    final.to_csv(OUT / f"RUL_{fd}.csv", index=False)
    print(fd, train.shape, test.shape, final.shape)
