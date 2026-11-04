import sys, warnings; warnings.simplefilter("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import rankdata, t as tdist
from combine_dlol.reliability import split_half_reliability, player_bootstrap_ci, verdict
MIN_N=40
c=pd.read_parquet("data/processed/combine_predictors.parquet"); c["nfl_id"]=c.nfl_id.astype(int)
meta=pd.read_csv("reports/phase2_combine_predictors.csv")
OFFICIAL=["forty","ten_yd_split","three_cone","short_shuttle","vertical","broad_jump","combine_height","combine_weight","arm_length","wing_span","hand_size","bench_reps"]
cols=["game_id","play_id","nfl_id","nfl_position","lined_up_position","down","yards_to_go","blitzing","offense_formation","dropback_duration","pass_rushers_encountered","player_get_off","pass_result","play_nullified_by_penalty","peak_pressure_probability_allowed","sack","tackle","tackle_for_loss","time_to_qb_hurry","time_to_pressure","sack_allowed","quick_pressure","unblocked_pressure"]
pp=pd.read_parquet("data/processed/player_play_joined.parquet",columns=cols)
pp=pp[pp.play_nullified_by_penalty!="Y"]
def gate(d,col):
    cnt=d.groupby("nfl_id")[col].count(); u=d[d.nfl_id.isin(cnt[cnt>=50].index)]
    s=split_half_reliability(u,"nfl_id",col,min_n=50,n_splits=1000,seed=7); lo,hi=player_bootstrap_ci(s,n_bootstrap=500,seed=8)
    return s.median,lo,verdict(s.median,lo)
def rates(group):
    if group=="DL":
        d=pp[pp.nfl_position.isin(["DE","DT","NT","OLB"])&pp.player_get_off.notna()&pp.pass_result.notna()].copy()
        d["sack_rate"]=(d.sack.fillna(0)>0)*1.0; d["hurry_rate"]=d.time_to_qb_hurry.notna()*1.0; d["tfl_rate"]=(d.tackle_for_loss.fillna(0)>0)*1.0
        d["tackle_rate"]=(d.tackle.fillna(0)>0)*1.0; d["pressure_rate"]=d.time_to_pressure.notna()*1.0
        outs=["sack_rate","hurry_rate","tfl_rate","tackle_rate","pressure_rate"]; f="C(lined_up_position)+C(down)+yards_to_go+C(blitzing)+C(offense_formation)"
    else:
        d=pp[pp.nfl_position.isin(["T","G","C"])&pp.peak_pressure_probability_allowed.notna()].copy()
        d["sack_allowed_rate"]=(d.sack_allowed.fillna(0)>0)*1.0; d["quick_pressure_rate"]=(d.quick_pressure.fillna(0)>0)*1.0; d["unblocked_pressure_rate"]=(d.unblocked_pressure.fillna(0)>0)*1.0
        outs=["sack_allowed_rate","quick_pressure_rate","unblocked_pressure_rate"]; f="C(lined_up_position)+C(down)+yards_to_go+C(offense_formation)+dropback_duration+pass_rushers_encountered"
    res={}; log=[]
    for o in outs:
        m,lo,v=gate(d,o); log.append((group,o,round(m,3),round(lo,3),v,round(d[o].mean(),4)))
        if v!="adequate": continue
        use=d.dropna(subset=["lined_up_position","down","yards_to_go","offense_formation"]+(["blitzing"] if group=="DL" else ["dropback_duration","pass_rushers_encountered"])).copy()
        use[o+"_adj"]=smf.ols(f"{o} ~ {f}",data=use).fit().resid
        cnt=use.groupby("nfl_id")[o+"_adj"].count(); u=use[use.nfl_id.isin(cnt[cnt>=50].index)]
        res[o]=u.groupby("nfl_id")[o+"_adj"].mean().rename(o)
    return res,log,d
dlsc=pd.read_parquet("data/processed/dl_outcomes_scores.parquet"); olsc=pd.read_parquet("data/processed/ol_outcomes_scores.parquet")
olw=olsc.pivot(index="nfl_id",columns="outcome",values="posterior_mean").reset_index().rename(columns={"peak_pressure_probability_allowed_residual":"peak_pressure","snap_reaction_time_residual":"snap_reaction","first_0_5s_displacement_residual":"first_0_5s_disp"})[["nfl_id","peak_pressure","snap_reaction","first_0_5s_disp"]]
gated={"DL":{"shrunk_d_a":-1,"shrunk_d_b_displacement_0_5":-1,"shrunk_d_b_displacement_1_0":-1,"shrunk_d_b_speed_0_5":-1},"OL":{"peak_pressure":-1,"snap_reaction":-1,"first_0_5s_disp":0}}
def pred_orient(p):
    if p in ("forty","ten_yd_split","three_cone","short_shuttle") or p.endswith("_time") or p.endswith("time_in_drill"): return -1
    if any(k in p for k in ("top_speed","burst","acceleration")) or p in ("vertical","broad_jump","bench_reps"): return 1
    return 0
def partial(x,y,Z):
    rx,ry=rankdata(x),rankdata(y); A=np.column_stack([np.ones(len(rx))]+[rankdata(z) if z.dtype!=bool else z.astype(float) for z in Z])
    ex=rx-A@np.linalg.lstsq(A,rx,rcond=None)[0]; ey=ry-A@np.linalg.lstsq(A,ey_ if False else ry,rcond=None)[0]; return np.corrcoef(ex,ey)[0,1]
rows=[]; glog=[]
for g in ("DL","OL"):
    res,lg,_=rates(g); glog+=lg
    pos=pp[pp.nfl_position.isin(["DE","DT","NT","OLB"] if g=="DL" else ["T","G","C"])].groupby("nfl_id").nfl_position.agg(lambda s:s.mode().iat[0])
    base=c[c.group==g].merge(dlsc[["nfl_id",*gated[g]]] if g=="DL" else olw,on="nfl_id")
    for o,s in res.items(): base=base.merge(s.reset_index(),on="nfl_id",how="left"); gated[g][o]=1 if o in("hurry_rate","tfl_rate","tackle_rate","pressure_rate","sack_rate") else -1
    base["role"]=base.nfl_id.map(pos).map(lambda p:("EDGE" if p in("DE","OLB") else "IDL") if g=="DL" else ("T" if p=="T" else "GC"))
    base["is_a"]=(base.role.isin(["EDGE","T"])).astype(bool)
    sens=list(meta[(meta.group==g)&(meta.family=="sensor")&(meta.eligible_for_M4.astype(bool))].predictor_column)
    preds=[p for p in OFFICIAL+sens if p in base.columns and base[p].notna().sum()>=MIN_N]
    for o,ori in gated[g].items():
        for p in preds:
            m=base[list(dict.fromkeys([p,o,"combine_weight","is_a","draft_year"]))].dropna()
            if len(m)<MIN_N or m[p].nunique()<5: continue
            Z=[m.is_a.to_numpy()] if p=="combine_weight" else [m.combine_weight.to_numpy(),m.is_a.to_numpy()]
            r=partial(m[p].to_numpy(float),m[o].to_numpy(float),Z) if False else None
            rx,ry=rankdata(m[p]),rankdata(m[o]); A=np.column_stack([np.ones(len(m))]+[rankdata(z) if z.dtype!=bool else z.astype(float) for z in Z])
            ex=rx-A@np.linalg.lstsq(A,rx,rcond=None)[0]; ey=ry-A@np.linalg.lstsq(A,ry,rcond=None)[0]; r=float(np.corrcoef(ex,ey)[0,1]); k=A.shape[1]-1
            df=len(m)-2-k; pv=2*tdist.sf(abs(r*np.sqrt(df/(1-r*r))),df); zz=np.arctanh(r); se=1/np.sqrt(len(m)-3-k); lo,hi=np.tanh(zz-1.96*se),np.tanh(zz+1.96*se)
            po=pred_orient(p); oriented=r*po*ori if (po!=0 and ori!=0) else np.nan
            rep=[]
            for yrs in ([2023,2024],[2025],[2023],[2024]):
                s=m[m.draft_year.isin(yrs)]
                if len(s)>=15:
                    rr=np.corrcoef(rankdata(s[p]),rankdata(s[o]))[0,1]; rep.append(rr)
                else: rep.append(np.nan)
            rows.append(dict(group=g,outcome=o,predictor=p,family="sensor" if p in sens else "official",n=len(m),r_adj=r,ci_lo=lo,ci_hi=hi,p=pv,oriented_r=oriented,r_disc_2023_24=rep[0],r_hold_2025=rep[1],r_2023=rep[2],r_2024=rep[3]))
R=pd.DataFrame(rows); m=len(R); order=np.argsort(R.p.values); q=np.empty(m); prev=1.0
for rank,i in zip(range(m,0,-1),order[::-1]): prev=min(prev,R.p.values[i]*m/rank); q[i]=prev
R["q_fdr"]=q; R["direction"]=np.where(R.oriented_r.isna(),"unclassified",np.where(R.oriented_r>0,"expected","opposite"))
R.to_csv("reports/exploratory_scan.csv",index=False); pd.DataFrame(glog,columns=["group","outcome","SB_median","lower_ci","verdict","mean_rate"]).to_csv("reports/exploratory_scan_outcome_gate.csv",index=False)
print("OUTCOME GATE (production rates):"); print(pd.DataFrame(glog,columns=["group","outcome","SB","lowCI","verdict","rate"]).to_string(index=False))
print(f"\nTESTS RUN: {m}  | by group: {R.group.value_counts().to_dict()}  | outcomes: {sorted(R.outcome.unique())}")
print(f"raw p<0.05: {(R.p<0.05).sum()} (chance expectation ~{0.05*m:.0f}) | q<0.10: {(R.q_fdr<0.10).sum()} | q<0.05: {(R.q_fdr<0.05).sum()}")
print("direction among q<0.10:",R[R.q_fdr<0.10].direction.value_counts().to_dict())
pd.set_option("display.width",260,"display.max_rows",200,"display.max_colwidth",44)
show=R[R.q_fdr<0.10].sort_values("q_fdr")
print(show[["group","outcome","predictor","n","r_adj","ci_lo","ci_hi","q_fdr","direction","r_disc_2023_24","r_hold_2025"]].round(3).to_string(index=False))
