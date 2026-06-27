#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resolve_forward.py  --  frozen-model forward-validation track record + your-vs-system gap.

Reads every dated watchlist in <out>/watchlists/ and fills what ACTUALLY happened
over the next 5 sessions, from the same panel the model scores from:
    entry = NEXT session OPEN ; exit = 5th session CLOSE ; MAE/MFE over the hold.
Only fully-elapsed windows resolve; younger rows stay 'pending' and resolve later.
Never re-scored by a new model -- your clean, never-rewritten live record.

Adds:
  * day-by-day journey: ret_d1..ret_d5 (close each session / entry - 1)
  * calm-dip forward bucket: cpr_w, narrow_cpr, gap, gap_down, calm_dip
  * if <out>/taken_trades.csv exists: joins your real trades and computes
      sys_5d_pct           = the % change to day-5 (next-open -> day5 close)
      hold_from_my_entry   = day5 close / your entry - 1
      variance_pct/_abs    = day5 close vs your EXIT (early-exit cost/benefit)

Outputs:
  <out>/forward_validation.csv        every prediction + outcome + journey
  <out>/taken_trades_resolved.csv     your trades next to what the system did
  prints SCORECARD (vol_skip A/B, calm-dip bucket) + your-vs-system summary

Usage:
  python resolve_forward.py --out <dir> --panel <fresh panel_cache.parquet>
"""
import argparse, glob, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="pipeline --out dir (contains watchlists/)")
ap.add_argument("--panel", required=True, help="fresh panel_cache.parquet")
ap.add_argument("--symbol-col", default="symbol")
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--sl-pct", type=float, default=0.06)
ap.add_argument("--narrow-thr", type=float, default=0.00613, help="narrow-CPR width threshold |TC-BC|/close")
args = ap.parse_args()

H = int(args.horizon)
SYM = args.symbol_col
wdir = os.path.join(args.out, "watchlists")
files = sorted(glob.glob(os.path.join(wdir, "watchlist_*.csv")))
if not files:
    raise SystemExit(f"no dated watchlists in {wdir} -- run the pipeline daily first.")

def mkdate(s):
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()

wl = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
if "date" not in wl.columns:   # fall back to filename date if pipeline didn't write one
    raise SystemExit("watchlist files need a 'date' column.")
wl["date"] = mkdate(wl["date"])
wl = wl.drop_duplicates(subset=[SYM, "date"], keep="first")
print(f"loaded {len(wl)} predictions from {len(files)} daily watchlists "
      f"({wl['date'].min().date()} -> {wl['date'].max().date()})")

panel = pd.read_parquet(args.panel)
dcol = next((c for c in ("timestamp", "date", "datetime") if c in panel.columns), None)
if dcol is None:
    raise SystemExit(f"no date column in panel: {list(panel.columns)[:12]}")
panel["_day"] = mkdate(panel[dcol])
panel = panel.sort_values([SYM, "_day"])
by_sym = {s: g.reset_index(drop=True) for s, g in panel.groupby(SYM, sort=False)}

rows = []
for r in wl.itertuples(index=False):
    rec = r._asdict()
    sym = rec.get(SYM); pdt = rec.get("date")
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
    # signal-day CPR + gap (known at entry)
    sh = g.iloc[i]; sH = float(sh["high"]); sL = float(sh["low"]); sC = float(sh["close"])
    Pv = (sH + sL + sC) / 3.0; BCv = (sH + sL) / 2.0; TCv = 2 * Pv - BCv
    cprw = abs(TCv - BCv) / sC if sC > 0 else np.nan
    gap = entry / sC - 1.0 if sC > 0 else np.nan
    narrow = int(np.isfinite(cprw) and cprw <= args.narrow_thr)
    gdn = int(np.isfinite(gap) and gap < 0)
    rec.update(status="resolved",
               entry_open=round(entry, 4),
               close_d5=round(float(closes[-1]), 4),
               ret_5=round(float(closes[-1]) / entry - 1.0, 4),
               mfe_5=round(hi / entry - 1.0, 4),
               mae_5=round(lo / entry - 1.0, 4),
               cpr_w=round(cprw, 5) if np.isfinite(cprw) else np.nan,
               narrow_cpr=narrow,
               gap=round(gap, 4) if np.isfinite(gap) else np.nan,
               gap_down=gdn,
               calm_dip=int(narrow and gdn))
    for k in range(H):                       # day-by-day journey
        rec[f"ret_d{k+1}"] = round(float(closes[k]) / entry - 1.0, 4)
    rows.append(rec)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(args.out, "forward_validation.csv"), index=False)

n_res = int((res["status"] == "resolved").sum())
n_pen = int((res["status"] == "pending").sum())
print(f"resolved {n_res}  |  pending {n_pen}  |  other {len(res) - n_res - n_pen}")
print("wrote forward_validation.csv\n")

done = res[res["status"] == "resolved"].copy()
if len(done) >= 5:
    done["SL"] = (done["mae_5"] <= -abs(args.sl_pct)).astype(int)

    def card(dd, lab):
        if len(dd) == 0:
            print(f"  {lab:28} n=   0"); return
        print(f"  {lab:28} n={len(dd):4}  win {(dd.ret_5 > 0).mean()*100:5.1f}%  "
              f"meanRet {dd.ret_5.mean()*100:+6.2f}%  SL {dd.SL.mean()*100:5.1f}%  "
              f"meanMAE {dd.mae_5.mean()*100:+6.2f}%  meanMFE {dd.mfe_5.mean()*100:+6.2f}%")

    print("=== FORWARD SCORECARD (frozen model, live) ===")
    card(done, "all resolved")
    if "vol_skip" in done.columns:
        card(done[done["vol_skip"] == 0], "kept (vol_skip=0)")
        card(done[done["vol_skip"] == 1], "flagged (vol_skip=1)")
    print("  -- calm-dip candidate (on probation) --")
    card(done[done["narrow_cpr"] == 1], "narrow_cpr")
    card(done[done["gap_down"] == 1], "gap_down")
    card(done[done["calm_dip"] == 1], "calm-dip (gapDN x narrowCPR)")
    print("\n  Give any filter ~30-40 resolved trades before trusting it; keep NON-binding until then.")
else:
    print("(need >=5 resolved trades before the scorecard prints -- keep running.)")

# ---- your actual trades vs the system ----
taken_path = os.path.join(args.out, "taken_trades.csv")
if os.path.exists(taken_path) and len(done):
    tk = pd.read_csv(taken_path)
    rr = done[[SYM, "date", "entry_open", "close_d5", "ret_5", "mae_5", "mfe_5"]].copy()
    rr["_d"] = mkdate(rr["date"]); rr = rr.drop(columns=["date"])
    tk["_d"] = mkdate(tk["log_date"])
    j = tk.merge(rr, on=[SYM, "_d"], how="left")
    j["sys_5d_pct"] = j["ret_5"] * 100.0
    j["hold_from_my_entry_pct"] = (j["close_d5"] / j["entry_price"] - 1.0) * 100.0
    is_exit = j["exit_price"].notna() & j["close_d5"].notna()
    j["variance_pct"] = np.where(is_exit, (j["close_d5"] / j["exit_price"] - 1.0) * 100.0, np.nan)
    has_qty = is_exit & j["qty"].notna()
    j["variance_abs"] = np.where(has_qty, (j["close_d5"] - j["exit_price"]) * j["qty"], np.nan)
    j.to_csv(os.path.join(args.out, "taken_trades_resolved.csv"), index=False)

    matured = j[j["close_d5"].notna()]
    ex = matured[matured["exit_price"].notna()]
    print("\n=== YOUR TRADES vs SYSTEM ===")
    print(f"  logged {len(j)} | matured {len(matured)} | exited & matured {len(ex)}")
    if len(ex):
        print(f"  your realized exit : mean {ex['realized_pnl_pct'].mean():+6.2f}%")
        print(f"  % change to day 5  : mean {ex['sys_5d_pct'].mean():+6.2f}%   (next-open -> day5 close)")
        print(f"  early-exit variance: mean {ex['variance_pct'].mean():+6.2f}%   "
              f"total Rs {np.nansum(ex['variance_abs']):+,.0f}")
        print("  Read: +variance = money left on the table by exiting early; -variance = drawdown dodged.")
        print("        Watch the SIGN over many trades -- that's whether your early exits help or hurt.")
    print("  wrote taken_trades_resolved.csv")
elif os.path.exists(taken_path):
    print("\n(taken_trades.csv found, but no picks have matured yet -- your trades resolve here "
          "once their 5-day window elapses.)")
