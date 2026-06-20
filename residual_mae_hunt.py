#!/usr/bin/env python3
r"""
residual_mae_hunt.py  --  the "crack" test, run rigorously over all 191 features.

QUESTION
    Does ANY model feature predict the drawdown (MAE) that the model's own
    probability LEFT ON THE TABLE -- i.e. downside information the upside-trained
    big-move model never optimized for (because it was trained on P(touch +5%),
    not on avoiding MAE)?

TWO GUARDS against fooling ourselves:
  (1) PERMUTATION NULL.  Searching ~191 features always turns up *something*
      in-sample by pure chance.  We shuffle the target many times to learn the
      best |corr| achievable by noise across the same 191-wide search.  A real
      feature must clear that floor.
  (2) HELD-OUT VALIDATION.  Discovery = earlier 60% of dates picks candidates;
      validation = later 40% must confirm them on data they were NOT chosen on.

KEEP CRITERION = ASYMMETRY.  A genuine risk-reducer makes the flagged trades'
MAE deeper WITHOUT shrinking their MFE (upside).  If a "filter" cuts MAE and MFE
together it is merely a volatility cut (it clips winners too) and is rejected.

USAGE
    python residual_mae_hunt.py "C:\path\to\prediction_journal.parquet"

OUTPUT
    - prints a compact report to the screen
    - writes residual_mae_results.csv  (small) -> send THAT back for interpretation
"""
import sys, numpy as np, pandas as pd

PATH       = sys.argv[1] if len(sys.argv) > 1 else "prediction_journal.parquet"
PROB_FLOOR = 0.75     # your real trade set
DISC_FRAC  = 0.60     # chronological discovery fraction
N_PERM     = 300      # permutation-null repetitions
TOPK       = 15       # how many candidates to validate
RNG        = np.random.default_rng(42)

# ----------------------------------------------------------------------
df = pd.read_parquet(PATH)
print(f"loaded {PATH}: {df.shape[0]} rows x {df.shape[1]} cols\n")

def pick(cands):
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns:        return c
        if c.lower() in low:       return low[c.lower()]
    return None

date_col = pick(["pred_date", "date"])
prob_col = pick(["prob", "prob_bigmove", "probability", "model prob"])
mae_col  = pick(["mae_5", "mae_5d", "mae"])
mfe_col  = pick(["mfe_5", "mfe_5d", "mfe"])
ret_col  = pick(["ret_5", "ret_5d", "ret5", "pct_movement", "% movement (5d)"])
oos_col  = pick(["is_oos", "oos"])

print("column mapping:")
for nm, c in [("date", date_col), ("prob", prob_col), ("mae", mae_col),
              ("mfe", mfe_col), ("ret5", ret_col), ("is_oos", oos_col)]:
    print(f"   {nm:7} -> {c}")
for nm, c in [("date", date_col), ("prob", prob_col), ("mae", mae_col),
              ("mfe", mfe_col), ("ret5", ret_col)]:
    if c is None:
        sys.exit(f"\nERROR: could not find a '{nm}' column.\nColumns present:\n{list(df.columns)}")
print()

# normalize outcome scale to FRACTIONS (handles %-stored columns)
for c in [mae_col, mfe_col, ret_col]:
    s = pd.to_numeric(df[c], errors="coerce")
    if s.abs().median() > 1.5:
        s = s / 100.0
    df[c] = s
df[date_col] = pd.to_datetime(df[date_col])
df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")

# restrict to genuine OOS rows
if oos_col is not None:
    df = df[df[oos_col].astype(bool)].copy()
    print(f"OOS rows: {len(df)}")
else:
    print("WARNING: no is_oos flag -- using ALL rows (may include in-sample!)")

# ----- candidate features = numeric cols minus known meta/outcome -----
EXCLUDE = {date_col, prob_col, mae_col, mfe_col, ret_col, oos_col}
FUZZY_DROP = ("symbol", "rank", "entry", "close_d", "close d", "open_d",
              "movement", "ret_1", "ret_3", "ret1", "ret3",
              "ev_gap", "ev_vol", "ev_large", "gap?", "volspike",
              "bigmove?", "big_move", "large_move", "pred_date", "_label", "label_")
feat_cols = []
for c in df.columns:
    if c in EXCLUDE:
        continue
    s = pd.to_numeric(df[c], errors="coerce")
    if s.notna().mean() < 0.80:      # not a usable numeric feature
        continue
    if any(p in c.lower() for p in FUZZY_DROP):
        continue
    df[c] = s
    feat_cols.append(c)

print(f"\nDetected {len(feat_cols)} candidate model features (expect ~191).")
print("   first 15:", feat_cols[:15])
print("   last  15:", feat_cols[-15:])
print("   -> if this count is far from 191, tell me and I'll adjust the filter.\n")

df = df.dropna(subset=[mae_col, mfe_col, ret_col, prob_col]).reset_index(drop=True)

# ----- chronological discovery / validation split -----
dates = np.sort(df[date_col].unique())
cut   = dates[int(len(dates) * DISC_FRAC)]
disc  = df[df[date_col] <  cut].copy()
val   = df[df[date_col] >= cut].copy()
print(f"discovery : {len(disc)} rows  {pd.Timestamp(disc[date_col].min()).date()} -> {pd.Timestamp(disc[date_col].max()).date()}")
print(f"validation: {len(val)} rows  {pd.Timestamp(val[date_col].min()).date()} -> {pd.Timestamp(val[date_col].max()).date()}\n")

# ----- residualize MAE on prob (strip what the model already knows) -----
disc["_pb"] = pd.qcut(disc[prob_col], 10, duplicates="drop")
mae_by_prob = disc.groupby("_pb", observed=True)[mae_col].mean().to_dict()
disc["_resMAE"] = disc[mae_col].values - disc["_pb"].map(mae_by_prob).astype(float).values

# feature matrix, standardized (impute median)
X = disc[feat_cols].apply(lambda s: s.fillna(s.median()))
Xs = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
Xs = Xs.fillna(0.0).values
y  = disc["_resMAE"].values
ys = (y - y.mean()) / y.std(ddof=0)
n  = len(ys)

# real correlations of every feature with residual-MAE
corr = (Xs.T @ ys) / n                      # shape (n_features,)

# ----- PERMUTATION NULL: best |corr| achievable by chance across the search -----
best_null = np.empty(N_PERM)
for i in range(N_PERM):
    yp = ys[RNG.permutation(n)]
    best_null[i] = np.abs((Xs.T @ yp) / n).max()
null_p95 = np.quantile(best_null, 0.95)
null_p50 = np.quantile(best_null, 0.50)
print("=== PERMUTATION NULL (best-of-search |corr| under pure noise) ===")
print(f"   median {null_p50:.4f}   95th pct {null_p95:.4f}")
print(f"   -> a feature is only 'real' if |corr| with residual-MAE exceeds {null_p95:.4f}\n")

order = np.argsort(-np.abs(corr))
print("=== TOP candidate features by |corr| with residual-MAE (discovery) ===")
print(f"   {'feature':28} {'corr':>8}  {'> null95?':>9}")
for j in order[:TOPK]:
    flag = "YES" if abs(corr[j]) > null_p95 else "no"
    print(f"   {feat_cols[j][:28]:28} {corr[j]:+8.4f}  {flag:>9}")
n_real = int((np.abs(corr) > null_p95).sum())
print(f"\n   {n_real} of {len(feat_cols)} features beat the noise floor.\n")

# ----- HELD-OUT VALIDATION of the top candidates (asymmetry test) -----
hp = val[val[prob_col] >= PROB_FLOOR].copy()
print(f"=== HELD-OUT VALIDATION within prob>={PROB_FLOOR}  (n={len(hp)}) ===")
print(f"   baseline: SL%={(hp[mae_col]<=-0.06).mean()*100:5.1f}  "
      f"meanMAE={hp[mae_col].mean()*100:+6.2f}  meanMFE={hp[mfe_col].mean()*100:+6.2f}  "
      f"ret5={hp[ret_col].mean()*100:+6.2f}\n")

rows = []
for j in order[:TOPK]:
    f = feat_cols[j]
    sign = np.sign(corr[j]) if corr[j] != 0 else 1.0
    # corr<0 : high feature -> deeper MAE -> flag HIGH end ; corr>0 -> flag LOW end
    thr = disc[f].quantile(0.75 if sign < 0 else 0.25)
    risky = (hp[f] > thr) if sign < 0 else (hp[f] < thr)
    keep, drop = hp[~risky], hp[risky]
    if len(drop) < 10 or len(keep) < 10:
        verdict = "too few flagged"
    else:
        dSL  = drop[mae_col].le(-0.06).mean() - keep[mae_col].le(-0.06).mean()   # +ve: dropped hit SL more (good)
        dMAE = keep[mae_col].mean() - drop[mae_col].mean()                       # +ve: dropped deeper (good)
        dMFE = keep[mfe_col].mean() - drop[mfe_col].mean()                       # ~0 good; large +ve = clipping winners
        # KEEP if dropped are worse on downside AND we are NOT mostly just cutting upside
        good_down = (dSL > 0.03) and (dMAE > 0.005)
        upside_ok = dMFE < 0.5 * dMAE
        verdict = "KEEP (real)" if (good_down and upside_ok) else \
                  ("vol-cut (clips upside)" if good_down else "no edge")
    rows.append(dict(
        feature=f, corr=round(float(corr[j]), 4),
        beats_null=bool(abs(corr[j]) > null_p95),
        n_drop=int(len(drop)),
        SL_keep=round(keep[mae_col].le(-0.06).mean()*100, 1) if len(keep) else np.nan,
        SL_drop=round(drop[mae_col].le(-0.06).mean()*100, 1) if len(drop) else np.nan,
        MAE_keep=round(keep[mae_col].mean()*100, 2) if len(keep) else np.nan,
        MAE_drop=round(drop[mae_col].mean()*100, 2) if len(drop) else np.nan,
        MFE_keep=round(keep[mfe_col].mean()*100, 2) if len(keep) else np.nan,
        MFE_drop=round(drop[mfe_col].mean()*100, 2) if len(drop) else np.nan,
        ret5_keep=round(keep[ret_col].mean()*100, 2) if len(keep) else np.nan,
        ret5_drop=round(drop[ret_col].mean()*100, 2) if len(drop) else np.nan,
        verdict=verdict))

out = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(out.to_string(index=False))
out.to_csv("residual_mae_results.csv", index=False)
print("\nwrote residual_mae_results.csv  ->  send this file back.")
print("(SL/MAE/MFE/ret in %. 'KEEP' = deepens MAE gap while sparing MFE; "
      "'vol-cut' = clips upside too; verdicts are screening only, I'll vet them.)")
