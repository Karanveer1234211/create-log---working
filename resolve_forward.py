#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resolve_forward.py  --  build the FROZEN-MODEL forward-validation track record.

Reads every dated watchlist in  <out>/watchlists/  (the predictions you froze each
day with the frozen model) and fills in what ACTUALLY happened over the next 5
trading sessions, using the same panel the model scores from.

    entry = NEXT session's OPEN   (what you'd really get; not the signal-day close)
    exit  = 5th session's CLOSE
    MAE/MFE measured over the hold, relative to entry

Only rows whose 5-session window has fully elapsed are 'resolved'; younger ones
stay 'pending' and resolve automatically on a later run. This file is never
re-scored by a new model -- it is your clean, never-rewritten live record, the
thing the backfill journal could NOT be.

Outputs:
    <out>/forward_validation.csv      every prediction + outcome (accumulating)
    prints a running SCORECARD, including the volatility-brake A/B
    (vol_skip==0 'kept' vs vol_skip==1 'flagged') so you can watch whether the
    filter actually earns its keep live BEFORE you ever make it binding.

Usage:
    python resolve_forward.py --out <pipeline --out dir> --panel <fresh panel_cache.parquet>
"""
import argparse, glob, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="the pipeline --out dir (must contain watchlists/)")
ap.add_argument("--panel", required=True, help="fresh panel_cache.parquet (carries recent prices)")
ap.add_argument("--symbol-col", default="symbol")
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--sl-pct", type=float, default=0.06, help="SL-touch proxy: MAE worse than -this")
args = ap.parse_args()

H = int(args.horizon)
wdir = os.path.join(args.out, "watchlists")
files = sorted(glob.glob(os.path.join(wdir, "watchlist_*.csv")))
if not files:
    raise SystemExit(f"no dated watchlists in {wdir} -- run the pipeline daily first.")

wl = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
wl["date"] = pd.to_datetime(wl["date"]).dt.normalize()
wl = wl.drop_duplicates(subset=[args.symbol_col, "date"], keep="first")
print(f"loaded {len(wl)} predictions from {len(files)} daily watchlists "
      f"({wl['date'].min().date()} -> {wl['date'].max().date()})")

# ---- prices ----
panel = pd.read_parquet(args.panel)
dcol = next((c for c in ("timestamp", "date", "datetime") if c in panel.columns), None)
if dcol is None:
    raise SystemExit(f"no date column in panel: {list(panel.columns)[:12]}")
d = pd.to_datetime(panel[dcol], errors="coerce")
try:
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
except (TypeError, AttributeError):
    pass
panel["_day"] = d.dt.normalize()
panel = panel.sort_values([args.symbol_col, "_day"])
by_sym = {s: g.reset_index(drop=True) for s, g in panel.groupby(args.symbol_col, sort=False)}

# ---- resolve each prediction ----
rows = []
for r in wl.itertuples(index=False):
    rec = r._asdict()
    sym = rec.get(args.symbol_col)
    pdt = rec.get("date")
    g = by_sym.get(sym)
    if g is None:
        rec["status"] = "no_price_data"; rows.append(rec); continue
    pos = g.index[g["_day"] == pd.Timestamp(pdt)]
    if len(pos) == 0:
        rec["status"] = "pred_date_not_in_panel"; rows.append(rec); continue
    i = int(pos[0])
    fut = g.iloc[i + 1: i + 1 + H]
    if len(fut) < H:
        rec["status"] = "pending"; rows.append(rec); continue
    entry = float(fut["open"].iloc[0])
    if not np.isfinite(entry) or entry <= 0:
        rec["status"] = "bad_entry"; rows.append(rec); continue
    closes = fut["close"].values.astype(float)
    hi = float(fut["high"].max()); lo = float(fut["low"].min())
    rec.update(status="resolved",
               entry_open=round(entry, 4),
               close_d5=round(float(closes[-1]), 4),
               ret_5=round(float(closes[-1]) / entry - 1.0, 4),
               mfe_5=round(hi / entry - 1.0, 4),
               mae_5=round(lo / entry - 1.0, 4))
    rows.append(rec)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(args.out, "forward_validation.csv"), index=False)

n_res = int((res["status"] == "resolved").sum())
n_pen = int((res["status"] == "pending").sum())
n_oth = int(len(res) - n_res - n_pen)
print(f"resolved {n_res}  |  pending {n_pen}  |  other {n_oth}")
print("wrote forward_validation.csv\n")

done = res[res["status"] == "resolved"].copy()
if len(done) >= 5:
    done["SL"] = (done["mae_5"] <= -abs(args.sl_pct)).astype(int)

    def card(dd, lab):
        if len(dd) == 0:
            print(f"  {lab:24} n=   0"); return
        print(f"  {lab:24} n={len(dd):4}  win {(dd.ret_5 > 0).mean() * 100:5.1f}%  "
              f"meanRet {dd.ret_5.mean() * 100:+6.2f}%  SL {dd.SL.mean() * 100:5.1f}%  "
              f"meanMAE {dd.mae_5.mean() * 100:+6.2f}%  meanMFE {dd.mfe_5.mean() * 100:+6.2f}%")

    print("=== FORWARD SCORECARD (frozen model, live) ===")
    card(done, "all resolved")
    if "vol_skip" in done.columns:
        card(done[done["vol_skip"] == 0], "kept (vol_skip=0)")
        card(done[done["vol_skip"] == 1], "flagged (vol_skip=1)")
        print("\n  Read: if 'flagged' shows worse SL / lower Ret than 'kept', the volatility")
        print("  brake is earning its keep live. Give it ~30-40 resolved trades before you")
        print("  trust it -- and keep it NON-binding (journaled) until then.")
else:
    print("(need >=5 resolved trades before the scorecard prints -- keep running daily.)")
