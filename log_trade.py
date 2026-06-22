#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_trade.py -- interactive execution logger + per-trade 5-day outcome & variance.

Flow:
  1. asks for a trading date (Enter = latest)
  2. shows that date's frozen watchlist as a numbered menu; pick by number/symbol
  3. per pick: entry time/price, then holding-or-exited (only asks exit fields if exited)
  4. auto-copies the frozen watchlist row (prob, rank, every indicator)
  5. THEN reads the panel and fills, for every logged trade whose 5-session window
     has elapsed:
        sys_5d_pct            = % change next-open -> day-5 close
        hold_from_my_entry_pct= day-5 close / your entry - 1
        variance_pct/_abs     = day-5 close vs your EXIT (early-exit cost/benefit)
        ret_d1..ret_d5        = day-by-day journey (%)
     Fresh trades show 'pending' and fill on a later run (the 5 days must elapse).

Panel: auto-detected at <out>/panel_cache.parquet (override with --panel).
Writes/append-to: <out>/taken_trades.csv  (deduped on log_date + symbol)

Usage:
  python log_trade.py --out "C:\\...\\out_rank_single"
"""
import argparse, os, glob, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--panel", default=None, help="panel_cache.parquet (default: <out>/panel_cache.parquet)")
ap.add_argument("--wl-subdir", default="watchlists")
ap.add_argument("--symbol-col", default="symbol")
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--narrow-thr", type=float, default=0.00613, help="narrow-CPR width threshold |TC-BC|/close")
args = ap.parse_args()

WLDIR = os.path.join(args.out, args.wl_subdir)
LOG   = os.path.join(args.out, "taken_trades.csv")
H = int(args.horizon)
SYM = args.symbol_col

def mkdate(s):
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()

files = sorted(glob.glob(os.path.join(WLDIR, "watchlist_*.csv")))
if not files:
    sys.exit(f"[err] no watchlist_*.csv in {WLDIR}")
dates = [os.path.basename(f).replace("watchlist_", "").replace(".csv", "") for f in files]
date2path = dict(zip(dates, files))

# 1. date prompt
print(f"\nAvailable watchlist dates: {dates[-8:]}{'  ...' if len(dates) > 8 else ''}")
while True:
    d = input(f"Trading date YYYY-MM-DD (Enter = latest {dates[-1]}): ").strip() or dates[-1]
    if d in date2path:
        break
    print(f"   no watchlist for {d}. available: {dates[-8:]}")
log_date = d
wl = pd.read_csv(date2path[d])

S = SYM if SYM in wl.columns else wl.columns[0]
probcol = next((c for c in ["prob_bigmove", "prob"] if c in wl.columns), None)
if "rank" not in wl.columns and probcol:
    wl = wl.sort_values(probcol, ascending=False).reset_index(drop=True)
    wl["rank"] = wl.index + 1
wl = wl.reset_index(drop=True)
wl_syms = set(wl[S].astype(str))
idx2sym = {str(i + 1): str(wl.iloc[i][S]) for i in range(len(wl))}

# 2. menu
print(f"\nWatchlist {log_date}  --  high-prob picks:\n")
print(f"  {'#':>3}  {'symbol':12} {'prob':>6} {'vol_skip':>8}")
for i in range(min(len(wl), 25)):
    r = wl.iloc[i]
    p = f"{r[probcol]:.3f}" if probcol else ""
    vs = r['vol_skip'] if 'vol_skip' in wl.columns else ""
    print(f"  {i+1:>3}  {str(r[S]):12} {p:>6} {str(vs):>8}")

def ask(prompt, cast=str, optional=True, choices=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and optional:
            return None
        if choices is not None:
            if raw.lower() in choices:
                return choices[raw.lower()]
            print(f"   pick one of {list(choices.keys())}"); continue
        try:
            return cast(raw)
        except Exception:
            print("   invalid, try again")

TYPES   = {"1": "Intraday(MIS)", "2": "Delivery(CNC)", "3": "MTF", "4": "F&O(NRML)", "5": "other"}
REASONS = {"1": "target", "2": "stop", "3": "time/horizon", "4": "discretionary", "5": "other"}
STATUS  = {"h": "holding", "e": "exited", "holding": "holding", "exited": "exited"}

def prompt_fill(symbol, off_system):
    print(f"\n  -- {symbol} {'(OFF-SYSTEM)' if off_system else ''} --")
    entry_time  = ask("   entry time (e.g. 09:20): ")
    entry_price = ask("   entry price: ", float, optional=False)
    status = ask("   still holding or exited?  [h]olding / [e]xited: ", choices=STATUS, optional=False)
    exit_price = exit_time = reason = None
    if status == "exited":
        exit_price = ask("   exit price: ", float, optional=False)
        exit_time  = ask("   exit time/date (blank ok): ")
        reason     = ask("   exit reason  [1]target [2]stop [3]time [4]discretionary [5]other: ", choices=REASONS)
    ttype = ask("   type  [1]Intraday [2]Delivery [3]MTF [4]F&O [5]other: ", choices=TYPES)
    qty   = ask("   qty (blank ok): ", float)
    notes = ask("   note (blank ok): ")
    pnl_pct = (exit_price / entry_price - 1.0) * 100.0 if (exit_price and entry_price) else None
    pnl_abs = (exit_price - entry_price) * qty if (exit_price and entry_price and qty) else None
    row = {"log_date": log_date, "symbol": symbol, "taken": True, "off_system": off_system,
           "status": status, "entry_time": entry_time, "entry_price": entry_price,
           "exit_price": exit_price, "exit_time": exit_time, "qty": qty,
           "trade_type": ttype, "exit_reason": reason,
           "realized_pnl_pct": round(pnl_pct, 3) if pnl_pct is not None else None,
           "realized_pnl_abs": round(pnl_abs, 2) if pnl_abs is not None else None,
           "note": notes}
    if not off_system:
        rr = wl[wl[S].astype(str) == symbol]
        if len(rr):
            for c in wl.columns:
                row[f"wl_{c}"] = rr.iloc[0][c]
    return row

# 3. collect
rows = []
took = input("\nWhich did you take? row numbers or symbols, comma-separated (Enter = none): ").strip()
if took:
    for tok in [x.strip() for x in took.split(",") if x.strip()]:
        sym = idx2sym.get(tok, tok)
        if sym not in wl_syms:
            if input(f"   '{sym}' not on watchlist. log as OFF-SYSTEM? (y/n): ").strip().lower() == "y":
                rows.append(prompt_fill(sym, True))
            continue
        rows.append(prompt_fill(sym, False))
off = input("\nAny OFF-watchlist trades? symbols comma-separated (Enter = none): ").strip()
if off:
    for sym in [x.strip() for x in off.split(",") if x.strip()]:
        rows.append(prompt_fill(sym, True))

if not rows:
    print(f"\nNothing logged for {log_date}.")
    # still refresh outcomes on existing log below if panel present
    new = pd.DataFrame()
else:
    new = pd.DataFrame(rows)

# append + dedupe
if os.path.exists(LOG):
    old = pd.read_csv(LOG)
    combined = pd.concat([old, new], ignore_index=True) if len(new) else old
    combined = combined.drop_duplicates(subset=["log_date", "symbol"], keep="last")
else:
    combined = new
if not len(combined):
    sys.exit(0)

# 5. resolve 5-day outcome + variance for matured rows (this is the "it's in log_trade" part)
panel_path = args.panel or os.path.join(args.out, "panel_cache.parquet")
def resolve(df):
    panel = pd.read_parquet(panel_path)
    dcol = next((c for c in ("timestamp", "date", "datetime") if c in panel.columns), None)
    panel["_day"] = mkdate(panel[dcol])
    panel = panel.sort_values([SYM, "_day"])
    bysym = {s: g.reset_index(drop=True) for s, g in panel.groupby(SYM, sort=False)}
    out = []
    for _, row in df.iterrows():
        r = row.to_dict()
        sym = str(r["symbol"]); ld = pd.Timestamp(pd.to_datetime(r["log_date"])).normalize()
        for c in ["resolve_status", "sys_entry_open", "close_d5", "sys_5d_pct",
                  "hold_from_my_entry_pct", "variance_pct", "variance_abs"] + [f"ret_d{k+1}" for k in range(H)]:
            r.setdefault(c, np.nan)
        for c in ["cpr_P", "cpr_TC", "cpr_BC", "cpr_w", "narrow_cpr", "gap_pct", "open_pos", "gap_down", "calm_dip"]:
            r.setdefault(c, np.nan)
        g = bysym.get(sym); r["resolve_status"] = "no_price_data"
        if g is not None:
            pos = g.index[g["_day"] == ld]
            if len(pos):
                i = int(pos[0])
                # --- entry-time signals: need only the signal-day bar + entry-day open ---
                sh = g.iloc[i]; sH = float(sh["high"]); sL = float(sh["low"]); sC = float(sh["close"])
                P = (sH + sL + sC) / 3.0; BC = (sH + sL) / 2.0; TC = 2 * P - BC
                cprw = abs(TC - BC) / sC if sC > 0 else np.nan
                r["cpr_P"] = round(P, 4); r["cpr_TC"] = round(TC, 4); r["cpr_BC"] = round(BC, 4)
                r["cpr_w"] = round(cprw, 5) if np.isfinite(cprw) else np.nan
                r["narrow_cpr"] = int(np.isfinite(cprw) and cprw <= args.narrow_thr)
                if i + 1 < len(g):
                    eo = float(g.iloc[i + 1]["open"])
                    if eo > 0 and sC > 0:
                        gp = eo / sC - 1.0
                        r["gap_pct"] = round(gp * 100, 3)
                        r["gap_down"] = int(gp < 0)
                        r["open_pos"] = "above_TC" if eo > TC else ("below_BC" if eo < BC else "inside_cpr")
                        r["calm_dip"] = int((gp < 0) and r["narrow_cpr"] == 1)
                # --- 5-day outcome: needs the full window ---
                fut = g.iloc[i + 1:i + 1 + H]
                if len(fut) >= H:
                    entry = float(fut["open"].iloc[0]); closes = fut["close"].values.astype(float)
                    if entry > 0:
                        r["resolve_status"] = "resolved"
                        r["sys_entry_open"] = round(entry, 4)
                        r["close_d5"] = round(float(closes[-1]), 4)
                        r["sys_5d_pct"] = round((closes[-1] / entry - 1) * 100, 3)
                        ep, xp, q = r.get("entry_price"), r.get("exit_price"), r.get("qty")
                        if pd.notna(ep) and ep:
                            r["hold_from_my_entry_pct"] = round((closes[-1] / float(ep) - 1) * 100, 3)
                        if pd.notna(xp) and xp:
                            r["variance_pct"] = round((closes[-1] / float(xp) - 1) * 100, 3)
                            if pd.notna(q) and q:
                                r["variance_abs"] = round((closes[-1] - float(xp)) * float(q), 2)
                        for k in range(H):
                            r[f"ret_d{k+1}"] = round((closes[k] / entry - 1) * 100, 3)
                else:
                    r["resolve_status"] = "pending"
            else:
                r["resolve_status"] = "logdate_not_in_panel"
        out.append(r)
    return pd.DataFrame(out)

resolved_ok = os.path.exists(panel_path)
if resolved_ok:
    combined = resolve(combined)
combined.to_csv(LOG, index=False)

# summary for what we just logged
print(f"\nLogged {len(new)} trade(s) for {log_date} -> {LOG}" if len(new) else f"\nRefreshed outcomes -> {LOG}")
if not resolved_ok:
    print(f"  (panel not found at {panel_path}; pass --panel to fill 5-day change + variance)")
show = new if len(new) else combined[combined["log_date"] == log_date]
for sym in show["symbol"].astype(str).tolist():
    r = combined[(combined["log_date"] == log_date) & (combined["symbol"].astype(str) == sym)]
    if not len(r): continue
    r = r.iloc[0]
    st = r.get("resolve_status", "n/a")
    sig = ""
    if pd.notna(r.get("gap_pct")):
        nc = "Y" if r.get("narrow_cpr") == 1 else "n"
        sig = f"gap {r['gap_pct']:+.2f}%  open {r.get('open_pos')}  narrowCPR={nc}  ::  "
    if st == "resolved":
        if r["status"] == "exited" and pd.notna(r.get("variance_pct")):
            va = f"  variance {r['variance_pct']:+.2f}%"
            if pd.notna(r.get("variance_abs")): va += f" (Rs {r['variance_abs']:+,.0f})"
            print(f"   {sym:12} {sig}exit {r['realized_pnl_pct']:+.2f}%  |  5-day {r['sys_5d_pct']:+.2f}%{va}")
        else:
            print(f"   {sym:12} {sig}HOLDING  |  5-day so far {r['sys_5d_pct']:+.2f}%")
    elif st == "pending":
        print(f"   {sym:12} {sig}logged -- 5-day change + variance fill in once the window elapses (re-run later)")
    else:
        print(f"   {sym:12} logged -- {st}")
print("\n+variance = money left on the table by exiting early; -variance = drawdown dodged.")
