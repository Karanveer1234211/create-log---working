#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signal_lab.py -- test the whole family of open-time entry signals on high-prob
picks in ONE pass: single signals + interactions, with a single family-wise
permutation bar across everything, an asymmetry rule, a min-cell floor, and a
correlation/duplicate check.

All signals are known at the entry-day OPEN (no lookahead): they use the entry
open price plus the prior (signal) day's OHLC-derived levels. Each test = "take
ONLY this set"; judged on held-out validation vs the take-all baseline.

A set is "real" only if its SL drop beats the best-of-everything noise bar AND it
doesn't shrink MFE/return. Interaction cells crossing gap with open-position are
flagged [dup~gap] (same event, not a real interaction).

Usage:
  python signal_lab.py --panel <panel_cache.parquet> [--picks <journal.parquet>]
        [--top-decile | --prob-floor 0.X] [--perms 1000] [--mincell 40]
"""
import argparse, numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--panel", required=True); ap.add_argument("--picks", default=None)
ap.add_argument("--symbol-col", default="symbol"); ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--prob-floor", type=float, default=0.0); ap.add_argument("--top-decile", action="store_true")
ap.add_argument("--sl-pct", type=float, default=0.06); ap.add_argument("--disc-frac", type=float, default=0.60)
ap.add_argument("--perms", type=int, default=1000); ap.add_argument("--mincell", type=int, default=40)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

def load(p):
    return pd.read_parquet(p) if p.endswith((".parquet",".pq")) else (
        pd.read_excel(p, sheet_name="OOS") if p.endswith((".xlsx",".xls")) else pd.read_csv(p))
def pick(cols, cands):
    low={c.lower():c for c in cols}
    for c in cands:
        if c in cols: return c
        if c.lower() in low: return low[c.lower()]
    return None
def mkdate(s):                       # parse + strip tz (panel is tz-aware, journal naive)
    s=pd.to_datetime(s, errors="coerce")
    if s.dt.tz is not None: s=s.dt.tz_localize(None)
    return s.dt.normalize()

# ---------- load + (optional) merge picks ----------
df=load(args.panel); S=args.symbol_col
df["_d"]=mkdate(df[pick(df.columns,["timestamp","date","datetime","pred_date"])])
for c in ["open","high","low","close"]:
    if c not in df.columns: raise SystemExit(f"[err] panel needs OHLC; missing '{c}'.")
prob_c=pick(df.columns,["prob_bigmove","prob"]); oos_c=pick(df.columns,["is_oos","oos"]); bb_c=pick(df.columns,["D_bb_bw_20","bb_bw_20","bb_bw"])
if (prob_c is None or oos_c is None) and args.picks:
    pk=load(args.picks); pk["_d"]=mkdate(pk[pick(pk.columns,["pred_date","date"])])
    pc=pick(pk.columns,["prob_bigmove","prob"]); oc=pick(pk.columns,["is_oos","oos"])
    df=df.merge(pk[[S,"_d"]+[c for c in (pc,oc) if c]].rename(columns={pc:"prob_bigmove",oc:"is_oos"}), on=[S,"_d"], how="left")
    prob_c,oos_c="prob_bigmove","is_oos"
if prob_c is None: raise SystemExit("[err] no prob column found.")
df["prob"]=pd.to_numeric(df[prob_c], errors="coerce")
df=df.sort_values([S,"_d"]).reset_index(drop=True)

# ---------- levels + next-open 5d outcomes ----------
P=(df.high+df.low+df.close)/3.0; BC=(df.high+df.low)/2.0; TC=2*P-BC
df["cpr_w"]=(TC-BC).abs()/df.close; df["P"],df["TC"],df["BC"]=P,TC,BC
df["bb"]=pd.to_numeric(df[bb_c],errors="coerce") if bb_c else np.nan
H=args.horizon; n=len(df); eo=np.full(n,np.nan); mae=np.full(n,np.nan); ret=np.full(n,np.nan); mfe=np.full(n,np.nan)
for _,g in df.groupby(S, sort=False):
    idx=g.index.to_numpy(); o=g.open.to_numpy(); hi=g.high.to_numpy(); lo=g.low.to_numpy(); cl=g.close.to_numpy()
    for k in range(len(g)-H):
        e=o[k+1]
        if not np.isfinite(e) or e<=0: continue
        eo[idx[k]]=e; mae[idx[k]]=lo[k+1:k+1+H].min()/e-1; mfe[idx[k]]=hi[k+1:k+1+H].max()/e-1; ret[idx[k]]=cl[k+1:k+1+H][-1]/e-1
df["eo"],df["mae"],df["mfe"],df["ret"]=eo,mae,mfe,ret

# ---------- restrict to OOS + high-prob trade set ----------
if oos_c is not None: df=df[df[oos_c].astype("boolean").fillna(False)]
df=df.dropna(subset=["mae","ret","eo","cpr_w","prob"]).copy()
if args.top_decile:
    thr=df.prob.quantile(0.90); df=df[df.prob>=thr]; print(f"trade set = top decile (prob >= {thr:.3f})")
else:
    df=df[df.prob>=args.prob_floor]; print(f"trade set = prob >= {args.prob_floor}")
df["gap"]=df.eo/df.close-1; df["SL"]=(df.mae<=-abs(args.sl_pct)).astype(int)
print(f"high-prob OOS picks: {len(df)}  ({df._d.min().date()} -> {df._d.max().date()})\n")

# ---------- discovery / validation ----------
dates=np.sort(df._d.unique()); cut=dates[int(len(dates)*args.disc_frac)]
disc=df[df._d<cut]; val=df[df._d>=cut].copy()
qC=disc.cpr_w.quantile(0.25)
base=val.SL.mean(); bmae,bret,bmfe=val.mae.mean(),val.ret.mean(),val.mfe.mean()
print(f"discovery {len(disc)} | validation {len(val)}   narrow-CPR width threshold = {qC:.5f}")
print(f"baseline (take ALL): n={len(val)}  SL {base*100:.1f}%  MAE {bmae*100:+.2f}  Ret {bret*100:+.2f}  MFE {bmfe*100:+.2f}\n")

# ---------- build all test sets (singles + interactions) ----------
v=val
g_up=(v.gap>0).values; c_narrow=(v.cpr_w<=qC).values
pos=np.where(v.eo>v.TC,"aboveTC",np.where(v.eo<v.BC,"belowBC","inside")); pd_green=(v.close>v.open).values
singles={
    "narrow_cpr":       c_narrow,
    "open_above_TC":    (v.eo>v.TC).values,
    "open_inside_cpr":  ((v.eo>=v.BC)&(v.eo<=v.TC)).values,
    "open_above_pivot": (v.eo>v.P).values,
    "open>prev_high":   (v.eo>v.high).values,
    "gap_up":           g_up,
    "gap_down":         (~g_up),
    "prev_day_green":   pd_green,
}
inter={}; dup=set()
for gl,gm in [("gapUP",g_up),("gapDN",~g_up)]:
    for cl,cm in [("narrowCPR",c_narrow),("wideCPR",~c_narrow)]: inter[f"{gl} x {cl}"]=gm&cm
    for pl in ["aboveTC","inside","belowBC"]:
        k=f"{gl} x {pl}"; inter[k]=gm&(pos==pl); dup.add(k)
    for dl,dm in [("prevGREEN",pd_green),("prevRED",~pd_green)]: inter[f"{gl} x {dl}"]=gm&dm
allsets={**{f"[S] {k}":m for k,m in singles.items()}, **{f"[I] {k}":m for k,m in inter.items()}}

# ---------- ONE family-wise permutation bar across everything judged ----------
slv=v.SL.values; judged={k:m for k,m in allsets.items() if m.sum()>=args.mincell}
rng=np.random.default_rng(args.seed); nb=np.empty(args.perms)
for i in range(args.perms):
    sh=rng.permutation(slv); b=sh.mean(); nb[i]=max((b-sh[m].mean()) for m in judged.values())
bar95=np.quantile(nb,0.95); bar99=np.quantile(nb,0.99)

def verdict(drop,sub):
    asym=(sub.mfe.mean()>=bmfe-0.003) and (sub.ret.mean()>=bret-0.003)
    if   drop>bar99 and asym: return "** real (beats 99% + keeps upside)"
    elif drop>bar95 and asym: return "* real (beats 95% + keeps upside)"
    elif drop>bar95:          return "beats bar BUT cuts upside (reject)"
    return "noise"

def report(title, d, flagdup=False):
    print(f"=== {title} (held-out validation; 'take ONLY this set') ===")
    print(f"{'set':24}{'n':>5}{'SL%':>7}{'SLdrop':>8}{'MAE%':>8}{'Ret%':>7}{'MFE%':>7}  verdict")
    rows=[]
    items=sorted(d.items(), key=lambda kv: -(base - v[kv[1]].SL.mean() if kv[1].sum()>=args.mincell else -9))
    for k,m in items:
        sub=v[m]; tag=" [dup~gap]" if (flagdup and any(t in k for t in("aboveTC","inside","belowBC"))) else ""
        if len(sub)<args.mincell:
            print(f"{k:24}{len(sub):5}   too few (<{args.mincell}){tag}"); rows.append((title,k,len(sub),*[np.nan]*5)); continue
        sl=sub.SL.mean(); drop=base-sl
        print(f"{k:24}{len(sub):5}{sl*100:7.1f}{drop*100:+8.1f}{sub.mae.mean()*100:+8.2f}{sub.ret.mean()*100:+7.2f}{sub.mfe.mean()*100:+7.2f}  {verdict(drop,sub)}{tag}")
        rows.append((title,k,len(sub),sl,drop,sub.mae.mean(),sub.ret.mean(),sub.mfe.mean()))
    print(); return rows

allrows =report("SINGLE SIGNALS", singles)
allrows+=report("INTERACTIONS",   inter, flagdup=True)

print(f"family-wise noise bar (best of {len(judged)} tests): 95% = +{bar95*100:.1f} SL pts | 99% = +{bar99*100:.1f} pts")
print("Real = SLdrop beats the bar AND MFE/Ret not shrinking. [dup~gap] = gap x open-position, same event.\n")

# ---------- correlation / duplicate check ----------
cc=pd.DataFrame({k:m.astype(float) for k,m in singles.items()})
if v.bb.notna().mean()>0.5: cc["bb_narrow"]=(v.bb<=disc.bb.median()).values.astype(float)
print("=== single-signal correlations (catch duplicates / the gap cluster) ===")
print(cc.corr().round(2).to_string())
pd.DataFrame(allrows,columns=["section","set","n","SL","SLdrop","MAE","Ret","MFE"]).to_csv("signal_lab_results.csv",index=False)
print("\nwrote signal_lab_results.csv")
