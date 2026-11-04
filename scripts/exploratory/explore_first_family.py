import sys, warnings; warnings.simplefilter("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from combine_dlol.reliability import split_half_reliability, player_bootstrap_ci, verdict
rng=np.random.default_rng(20261103)
c=pd.read_parquet("data/processed/combine_predictors.parquet"); c["nfl_id"]=c.nfl_id.astype(int)
dl=pd.read_parquet("data/processed/dl_outcomes_scores.parquet")
ol=pd.read_parquet("data/processed/ol_outcomes_scores.parquet")
olw=ol.pivot(index="nfl_id",columns="outcome",values="posterior_mean").reset_index()
OLO={"peak_pressure_probability_allowed_residual":"peak_pressure","snap_reaction_time_residual":"snap_reaction","first_0_5s_displacement_residual":"first_0_5s_disp"}
olw=olw.rename(columns=OLO)[["nfl_id",*OLO.values()]]
DLO=["shrunk_d_a","shrunk_d_b_displacement_0_5","shrunk_d_b_displacement_1_0","shrunk_d_b_speed_0_5"]
# production outcomes (DL)
pp=pd.read_parquet("data/processed/player_play_joined.parquet",columns=["game_id","play_id","nfl_id","nfl_position","player_get_off","pass_result","play_nullified_by_penalty","sack","tackle_for_loss","time_to_qb_hurry"])
d=pp[pp.nfl_position.isin(["DE","DT","NT","OLB"])&pp.player_get_off.notna()&pp.pass_result.notna()&(pp.play_nullified_by_penalty!="Y")].copy()
d["sack_rate"]=(d.sack.fillna(0)>0).astype(float); d["hurry_rate"]=d.time_to_qb_hurry.notna().astype(float); d["tfl_rate"]=(d.tackle_for_loss.fillna(0)>0).astype(float)
prod={}
print("PRODUCTION RELIABILITY (frozen gate)")
for o in ["sack_rate","hurry_rate","tfl_rate"]:
    cnt=d.groupby("nfl_id")[o].count(); u=d[d.nfl_id.isin(cnt[cnt>=50].index)]
    s=split_half_reliability(u,"nfl_id",o,min_n=50,n_splits=2000,seed=7); lo,hi=player_bootstrap_ci(s,n_bootstrap=1000,seed=8)
    v=verdict(s.median,lo); print(f"  {o}: SB {s.median:.3f} [{lo:.3f},{hi:.3f}] -> {v}; mean rate {u[o].mean():.4f}",flush=True)
    if v=="adequate": prod[o]=u.groupby("nfl_id")[o].mean().rename(o)
pdf=pd.concat(prod.values(),axis=1).reset_index() if prod else pd.DataFrame({"nfl_id":[]})
PRED=["forty","ten_yd_split","forty_yard_dash_top_speed","forty_yard_dash_sensor_ten_yd_burst_speed"]
def build(g):
    x=c[c.group==g]
    if g=="DL": x=x.merge(dl[["nfl_id",*DLO]],on="nfl_id").merge(pdf,on="nfl_id",how="left"); outs=DLO+list(prod); drill="pass_rush_drill_top_speed"
    else: x=x.merge(olw,on="nfl_id"); outs=list(OLO.values()); drill="pass_rush_drop_top_speed"
    return x,outs,PRED+[drill]
def partial(x,y,z):
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(z)
    bx=np.polyfit(rz,rx,1); by=np.polyfit(rz,ry,1); ex=rx-np.polyval(bx,rz); ey=ry-np.polyval(by,rz)
    return np.corrcoef(ex,ey)[0,1]
def pval(r,n,k=0):
    from scipy.stats import t
    df=n-2-k; tt=r*np.sqrt(df/(1-r*r)); return 2*t.sf(abs(tt),df)
rows=[]
for g in ("DL","OL"):
    x,outs,preds=build(g)
    for o in outs:
        for p in preds:
            m=x[[o,p,"combine_weight"]].dropna()
            if len(m)<30: continue
            for kind in ("raw","partial_weight"):
                f=(lambda a,b,w: spearmanr(a,b)[0]) if kind=="raw" else partial
                r=f(m[p].to_numpy(),m[o].to_numpy(),m["combine_weight"].to_numpy())
                bs=[]
                for _ in range(1000):
                    i=rng.integers(0,len(m),len(m)); mm=m.iloc[i]
                    try: bs.append(f(mm[p].to_numpy(),mm[o].to_numpy(),mm["combine_weight"].to_numpy()))
                    except Exception: pass
                lo,hi=np.nanquantile(bs,[0.025,0.975])
                rows.append(dict(group=g,outcome=o,predictor=p,kind=kind,n=len(m),r=r,ci_lo=lo,ci_hi=hi,p=pval(r,len(m),1 if kind!="raw" else 0)))
R=pd.DataFrame(rows)
# BH FDR across whole family
order=np.argsort(R.p.values); m=len(R); q=np.empty(m); prev=1.0
for rank,i in zip(range(m,0,-1),order[::-1]): prev=min(prev,R.p.values[i]*m/rank); q[i]=prev
R["q_fdr"]=q; R.to_csv("reports/exploratory_correlations.csv",index=False)
print(f"\nFAMILY SIZE {m} tests; q<0.05: {(R.q_fdr<0.05).sum()}, q<0.10: {(R.q_fdr<0.10).sum()}, raw p<0.05: {(R.p<0.05).sum()} (expected by chance ~{0.05*m:.1f})")
pd.set_option("display.width",250,"display.max_rows",200)
print(R.sort_values("p").head(14)[["group","outcome","predictor","kind","n","r","ci_lo","ci_hi","p","q_fdr"]].round(3).to_string())
