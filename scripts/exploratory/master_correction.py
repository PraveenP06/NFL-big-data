import numpy as np
import pandas as pd

COLS = ["scope", "group", "outcome", "predictor", "family", "n", "r_adj", "ci_lo", "ci_hi", "p",
        "direction", "oriented_r", "r_disc_2023_24", "r_hold_2025"]


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in zip(range(m, 0, -1), order[::-1], strict=True):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def _new_scope(scope: str, _unused: None) -> pd.DataFrame:
    d = pd.read_csv(f"reports/scope_{scope}_results.csv")
    return d[d.p.notna()].assign(scope=scope)


def main() -> None:
    frames = []
    dl = pd.read_csv("reports/scope_dl_results.csv")
    dl = dl[(dl.group == "pooled") & dl.p.notna()].assign(scope="dl")
    frames.append(dl)
    for scope in ["ol", "wrte", "db", "value"]:
        d = pd.read_csv(f"reports/scope_{scope}_results.csv")
        frames.append(d[d.p.notna()].assign(scope=scope))
    broad = pd.read_csv("reports/exploratory_scan.csv")
    frames.append(broad[broad.group == "OL"].assign(scope="broad_scan_OL"))
    frames.append(_new_scope("wrte_drills", None))
    frames.append(_new_scope("db_drills", None))
    ngs = pd.read_csv("reports/scope_ngs_agility_results.csv")
    ngs = ngs[ngs.p.notna() & (ngs.family_label.str.contains("_main_") | ngs.family_label.str.contains("allbyall"))]
    frames.append(ngs.assign(scope="ngs_agility"))
    master = pd.concat([f.reindex(columns=COLS) for f in frames], ignore_index=True)
    master = master.drop_duplicates(subset=["group", "outcome", "predictor"], keep="first").reset_index(drop=True)
    master["q_master"] = benjamini_hochberg(master.p.to_numpy())
    master.to_csv("reports/master_results.csv", index=False)
    print(len(master), "tests;", int((master.p < 0.05).sum()), "raw p<0.05;",
          int((master.q_master < 0.10).sum()), "q<0.10;", int((master.q_master < 0.05).sum()), "q<0.05")


if __name__ == "__main__":
    main()
