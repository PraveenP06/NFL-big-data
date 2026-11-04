import sys, warnings; warnings.simplefilter("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, t
rng=np.random.default_rng(11)
c=pd.read_parquet("data/processed/combine_predictors.parquet"); c["nfl_id"]=c.nfl_id.astype(int)
pp=pd.read_parquet("data/processed/player_play_joined.parquet",columns=["nfl_id","nfl_position","lined_up_position","down","yards_to_go","blitzing","player_get_off","pass_result","play_nullified_by_penalty","time_to_qb_hurry","tackle_for_loss"])
d=pp[pp.nfl_position.isin(["DE","DT","NT","OLB"])&pp.player_get_off.notna()&pp.pass_result.notna()&(pp.play_nullified_by_penalty!="Y")].copy()
d["hurry"]=d.time_to_qb_hurry.notna().astype(float); d["tfl"]=(d.tackle_for_loss.fillna(0)>0).astype(float)
import statsmodels.formula.api as smf
for o in ("hurry","tfl"):
    f=smf.ols(f"{o} ~ C(lined_up_position)+C(down)+yards_to_go+C(blitzing)",data=d).fit(); d[o+"_res"]=f.resid
    d[o+"_posres"]=d[o]-d.groupby("nfl_id")[o].transform("mean")*0  # placeholder
cnt=d.groupby("nfl_id").size(); keep=cnt[cnt>=50].index
g=d[d.nfl_id.isin(keep)].groupby("nfl_id").agg(pos=("nfl_position",lambda s:s.mode().iat[0]),hurry=("hurry","mean"),tfl=("tfl","mean"),hurry_adj=("hurry_res","mean"),tfl_adj=("tfl_res","mean")).reset_index()
g["group_pos"]=np.where(g.pos.isin(["DE","OLB"]),"EDGE","IDL")
x=c[c.group=="DL"].merge(g,on="nfl_id")
print("players",len(x),x.group_pos.value_counts().to_dict(),"| mean hurry rate by group:",x.groupby("group_pos").hurry.mean().round(3).to_dict(),"| mean forty:",x.groupby("group_pos").forty.mean().round(2).to_dict())
PRED=["forty","ten_yd_split","forty_yard_dash_top_speed","forty_yard_dash_sensor_ten_yd_burst_speed","pass_rush_drill_top_speed"]
rows=[]
for o in ("hurry","hurry_adj","tfl","tfl_adj"):
    for grp,s in [("ALL",x),("EDGE",x[x.group_pos=="EDGE"]),("IDL",x[x.group_pos=="IDL"])]:
        for p in PRED:
            m=s[[o,p]].dropna()
            if len(m)<25: continue
            r=spearmanr(m[p],m[o])[0]; bs=[spearmanr(m.iloc[i][p],m.iloc[i][o])[0] for i in (rng.integers(0,len(m),len(m)) for _ in range(500))]
            lo,hi=np.nanquantile(bs,[.025,.975]); tt=r*np.sqrt((len(m)-2)/(1-r*r)); pv=2*t.sf(abs(tt),len(m)-2)
            rows.append(dict(outcome=o,subset=grp,predictor=p,n=len(m),r=r,ci_lo=lo,ci_hi=hi,p=pv))
R=pd.DataFrame(rows); R.to_csv("reports/exploratory_robustness.csv",index=False)
pd.set_option("display.width",250,"display.max_rows",200)
for o in ("hurry","hurry_adj","tfl_adj"):
    print("\n",o); print(R[R.outcome==o].pivot_table(index="predictor",columns="subset",values="r").round(2).to_string())
print("\nhurry_adj EDGE/IDL detail"); print(R[(R.outcome=="hurry_adj")&(R.subset!="ALL")][["subset","predictor","n","r","ci_lo","ci_hi","p"]].round(3).to_string())
