import sys, time, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from combine_dlol import ol_outcomes as M
from combine_dlol.reliability import permutation_null

OUT = ["peak_pressure_probability_allowed", "pressure_allowed", "snap_reaction_time", "first_0_5s_displacement"]
def groups(adj):
    return {"all": adj, "T": adj.loc[adj["nfl_position"].eq("T")], "G_C": adj.loc[adj["nfl_position"].isin(["G", "C"])]}

def null_job(args):
    key, outcome, sub = args
    adj = pd.read_parquet(f"data/processed/ol_outcomes_adjusted_{outcome}.parquet")
    col = f"{outcome}_residual" if f"{outcome}_residual" in adj.columns else outcome
    use = M._eligible(groups(adj)[sub], col)
    t = time.time()
    n = permutation_null(use, "nfl_id", col, M.MIN_PLAYS, n_permutations=500, n_splits=100, seed=M.RANDOM_SEED + 2)
    return key, float(np.nanmedian(n)), time.time() - t

if __name__ == "__main__":
    adjusted, rows = {}, []
    for o in OUT:
        adj = pd.read_parquet(f"data/processed/ol_outcomes_adjusted_{o}.parquet")
        col = f"{o}_residual" if f"{o}_residual" in adj.columns else o
        adjusted[col] = adj
    print({k: list(v.columns)[-3:] for k, v in adjusted.items()}, flush=True)
    jobs = []
    for col, adj in adjusted.items():
        for sub, data in groups(adj).items():
            use = M._eligible(data, col)
            if use["nfl_id"].nunique() < 3:
                continue
            s = M.summarize_movement(use, col, n_permutations=0)
            rows.append(M._summary_row(col, sub, s, use["nfl_id"].nunique(), len(use)))
            jobs.append((f"{col}|{sub}", col.replace("_residual", ""), sub))
    prov = pd.DataFrame(rows)
    prov.to_csv("reports/phase1_ol_reliability_provisional.csv", index=False)
    print(prov[["outcome","subgroup","eligible_players","split_half_spearman_brown_median","bootstrap_ci_lower","mixedlm_icc","verdict"]].to_string(), flush=True)
    with ProcessPoolExecutor(10) as ex:
        res = {k: m for k, m, _ in ex.map(null_job, jobs)}
    prov["permutation_null_median"] = [res[f"{r.outcome}|{r.subgroup}"] for r in prov.itertuples()]
    prov.to_csv("reports/phase1_ol_reliability.csv", index=False)
    scores = M._scores(adjusted)
    scores.to_parquet("data/processed/ol_outcomes_scores.parquet", index=False)
    M._plot_reliability(prov, Path("reports/phase1_ol_reliability.png"))
    print("DONE", flush=True)
