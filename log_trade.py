#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_trade.py -- interactive execution logger with full per-trade analytics.

You type only your fill (entry date/time/price, holding-or-exited, exit, qty, type).
Everything else is computed from the frozen watchlist + the panel:
  - entry-time signals (immediate):  CPR P/TC/BC, cpr_w, narrow_cpr, gap%, open_pos
  - your hold duration:              exit datetime - entry datetime (mins or days)
  - same-day view:                   day1_close + var_day1 (you exited 100, it closed 110/90)
  - holdings:                        live unrealized P&L vs the latest close
  - on maturity (5 sessions):        sys_5d_pct, variance vs your exit, ret_d1..d5 journey,
                                     days_to_profit / days_to_stop
  - totals:                          realized (absorbed) + unrealized (unabsorbed)

Previously logged trades are loaded, updated, and re-saved every run -- you never
re-enter history. Dedupe key = entry_date + symbol.

Panel auto-detected at <out>/panel_cache.parquet (override --panel).
Usage:  python log_trade.py --out "C:\\...\\out_rank_single"
"""
import argparse, os, glob, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--panel", default=None)
ap.add_argument("--wl-subdir", default="watchlists")
ap.add_argument("--symbol-col", default="symbol")
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--sl-pct", type=float, default=0.06)
ap.add_argument("--narrow-thr", type=float, default=0.00613)
args = ap.parse_args()

WLDIR = os.path.join(args.out, args.wl_subdir)
LOG = os.path.join(args.out, "taken_trades.csv")
H = int(args.horizon); SYM = args.symbol_col

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

# panel up front (for default entry date + all computed fields)
panel_path = args.panel or os.path.join(args.out, "panel_cache.parquet")
bysym, all_days = {}, []
if os.path.exists(panel_path):
    panel = pd.read_parquet(panel_path)
    dcol = next((c for c in ("timestamp", "date", "datetime") if c in panel.columns), None)
    panel["_day"] = mkdate(panel[dcol])
    panel = panel.sort_values([SYM, "_day"])
    bysym = {s: g.reset_index(drop=True) for s, g in panel.groupby(SYM, sort=False)}
    all_days = np.sort(panel["_day"].unique())
else:
    print(f"[warn] panel not found at {panel_path} -- computed fields will be blank until you pass --panel")

# 1. date prompt
print(f"\nAvailable watchlist dates: {dates[-8:]}{'  ...' if len(dates) > 8 else ''}")
while True:
    d = input(f"Watchlist date YYYY-MM-DD (Enter = latest {dates[-1]}): ").strip() or dates[-1]
    if d in date2path:
        break
    print(f"   no watchlist for {d}")
log_date = d
wl = pd.read_csv(date2path[d])
S = SYM if SYM in wl.columns else wl.columns[0]
probcol = next((c for c in ["prob_bigmove", "prob"] if c in wl.columns), None)
if "rank" not in wl.columns and probcol:
    wl = wl.sort_values(probcol, ascending=False).reset_index(drop=True); wl["rank"] = wl.index + 1
wl = wl.reset_index(drop=True)
wl_syms = set(wl[S].astype(str))
idx2sym = {str(i + 1): str(wl.iloc[i][S]) for i in range(len(wl))}

wdate = pd.Timestamp(pd.to_datetime(log_date)).normalize()
nxt = [x for x in all_days if x > np.datetime64(wdate)]
default_entry = pd.Timestamp(nxt[0]).strftime("%Y-%m-%d") if len(nxt) else log_date

# 2. menu
print(f"\nWatchlist {log_date}  --  high-prob picks:\n")
print(f"  {'#':>3}  {'symbol':12} {'prob':>6} {'vol_skip':>8}")
for i in range(min(len(wl), 25)):
    r = wl.iloc[i]; p = f"{r[probcol]:.3f}" if probcol else ""
    vs = r['vol_skip'] if 'vol_skip' in wl.columns else ""
    print(f"  {i+1:>3}  {str(r[S]):12} {p:>6} {str(vs):>8}")

def ask(prompt, cast=str, optional=True, choices=None, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
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

TYPES = {"1": "Intraday(MIS)", "2": "Delivery(CNC)", "3": "MTF", "4": "F&O(NRML)", "5": "other"}
REASONS = {"1": "target", "2": "stop", "3": "time/horizon", "4": "discretionary", "5": "other"}
STATUS = {"h": "holding", "e": "exited", "holding": "holding", "exited": "exited"}

def hold_duration(ed, et, xd, xt):
    if not xd:
        return None
    d0 = pd.to_datetime(ed); d1 = pd.to_datetime(xd)
    days = (d1.normalize() - d0.normalize()).days
    if days == 0 and et and xt:
        try:
            return f"{(int(xt[:2])*60+int(xt[3:5])) - (int(et[:2])*60+int(et[3:5]))}min"
        except Exception:
            return "same day"
    return "same day" if days == 0 else f"{days}d"

def prompt_fill(symbol, off_system):
    print(f"\n  -- {symbol} {'(OFF-SYSTEM)' if off_system else ''} --")
    entry_date = ask(f"   entry date YYYY-MM-DD (Enter = {default_entry}): ", default=default_entry)
    entry_time = ask("   entry time (e.g. 09:20): ")
    entry_price = ask("   entry price: ", float, optional=False)
    status = ask("   still holding or exited?  [h]olding / [e]xited: ", choices=STATUS, optional=False)
    exit_date = exit_time = exit_price = reason = None
    if status == "exited":
        exit_date = ask(f"   exit date YYYY-MM-DD (Enter = {entry_date}): ", default=entry_date)
        exit_time = ask("   exit time (blank ok): ")
        exit_price = ask("   exit price: ", float, optional=False)
        reason = ask("   exit reason  [1]target [2]stop [3]time [4]discretionary [5]other: ", choices=REASONS)
    ttype = ask("   type  [1]Intraday [2]Delivery [3]MTF [4]F&O [5]other: ", choices=TYPES)
    qty = ask("   qty (blank ok): ", float)
    note = ask("   note (blank ok): ")
    pnl_pct = (exit_price/entry_price - 1)*100 if (exit_price and entry_price) else None
    pnl_abs = (exit_price-entry_price)*qty if (exit_price and entry_price and qty) else None
    row = {"log_date": log_date, "symbol": symbol, "taken": True, "off_system": off_system,
           "entry_date": entry_date, "entry_time": entry_time, "entry_price": entry_price,
           "status": status, "exit_date": exit_date, "exit_time": exit_time, "exit_price": exit_price,
           "qty": qty, "trade_type": ttype, "exit_reason": reason,
           "hold_duration": hold_duration(entry_date, entry_time, exit_date, exit_time),
           "realized_pnl_pct": round(pnl_pct, 3) if pnl_pct is not None else None,
           "realized_pnl_abs": round(pnl_abs, 2) if pnl_abs is not None else None, "note": note}
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
            if input(f"   '{sym}' not on watchlist. log OFF-SYSTEM? (y/n): ").strip().lower() == "y":
                rows.append(prompt_fill(sym, True))
            continue
        rows.append(prompt_fill(sym, False))
off = input("\nAny OFF-watchlist trades? symbols comma-separated (Enter = none): ").strip()
if off:
    for sym in [x.strip() for x in off.split(",") if x.strip()]:
        rows.append(prompt_fill(sym, True))
new = pd.DataFrame(rows) if rows else pd.DataFrame()

# append + dedupe on entry_date + symbol
if os.path.exists(LOG):
    old = pd.read_csv(LOG)
    combined = pd.concat([old, new], ignore_index=True) if len(new) else old
    key = ["entry_date", "symbol"] if "entry_date" in combined.columns else ["log_date", "symbol"]
    combined = combined.drop_duplicates(subset=key, keep="last")
else:
    combined = new
if not len(combined):
    print("\nNothing logged."); sys.exit(0)

# resolve all computed fields
def resolve(df):
    out = []
    NEWC = ["cpr_P", "cpr_TC", "cpr_BC", "cpr_w", "narrow_cpr", "gap_pct", "open_pos", "gap_down",
            "calm_dip", "day1_close", "var_day1_pct", "var_day1_abs", "sys_entry_open", "close_d5",
            "sys_5d_pct", "hold_from_my_entry_pct", "variance_pct", "variance_abs", "days_to_profit",
            "days_to_stop", "last_close", "last_close_date", "unrealized_pnl_pct", "unrealized_pnl_abs"] \
           + [f"ret_d{k+1}" for k in range(H)]
    for _, row in df.iterrows():
        r = row.to_dict()
        for c in NEWC:
            r.setdefault(c, np.nan)
        sym = str(r["symbol"])
        edt = pd.Timestamp(pd.to_datetime(r.get("entry_date"))).normalize() if pd.notna(r.get("entry_date")) else None
        g = bysym.get(sym); r["resolve_status"] = "no_price_data" if g is None else "pending"
        ep = r.get("entry_price"); xp = r.get("exit_price"); q = r.get("qty")
        if g is None or edt is None:
            out.append(r); continue
        # holdings: live mark to latest close (always available)
        if str(r.get("status")) == "holding":
            last = g.iloc[-1]; lc = float(last["close"])
            r["last_close"] = round(lc, 4); r["last_close_date"] = str(pd.Timestamp(last["_day"]).date())
            if pd.notna(ep) and ep:
                r["unrealized_pnl_pct"] = round((lc/float(ep)-1)*100, 3)
                if pd.notna(q) and q:
                    r["unrealized_pnl_abs"] = round((lc-float(ep))*float(q), 2)
        pos = g.index[g["_day"] == edt]
        if not len(pos):
            r["resolve_status"] = "entry_date_not_in_panel"; out.append(r); continue
        i = int(pos[0])
        entry = float(g.iloc[i]["open"])           # entry-day OPEN
        # entry-time signals (need signal day i-1 + entry-day open)
        if i >= 1 and entry > 0:
            sh = g.iloc[i-1]; sH = float(sh["high"]); sL = float(sh["low"]); sC = float(sh["close"])
            P = (sH+sL+sC)/3.0; BC = (sH+sL)/2.0; TC = 2*P-BC
            cprw = abs(TC-BC)/sC if sC > 0 else np.nan
            r["cpr_P"] = round(P, 4); r["cpr_TC"] = round(TC, 4); r["cpr_BC"] = round(BC, 4)
            r["cpr_w"] = round(cprw, 5) if np.isfinite(cprw) else np.nan
            r["narrow_cpr"] = int(np.isfinite(cprw) and cprw <= args.narrow_thr)
            if sC > 0:
                gp = entry/sC - 1.0
                r["gap_pct"] = round(gp*100, 3); r["gap_down"] = int(gp < 0)
                r["open_pos"] = "above_TC" if entry > TC else ("below_BC" if entry < BC else "inside_cpr")
                r["calm_dip"] = int((gp < 0) and r["narrow_cpr"] == 1)
        # 5-day window from entry day (entry day = day1)
        fut = g.iloc[i:i+H]
        if len(fut) >= H and entry > 0:
            closes = fut["close"].values.astype(float); lows = fut["low"].values.astype(float)
            r["resolve_status"] = "resolved"
            r["sys_entry_open"] = round(entry, 4)
            r["day1_close"] = round(float(closes[0]), 4)
            r["close_d5"] = round(float(closes[-1]), 4)
            r["sys_5d_pct"] = round((closes[-1]/entry-1)*100, 3)
            r["days_to_profit"] = next((k+1 for k in range(H) if closes[k] > entry), None)
            r["days_to_stop"] = next((k+1 for k in range(H) if lows[k] <= entry*(1-args.sl_pct)), None)
            if pd.notna(ep) and ep:
                r["hold_from_my_entry_pct"] = round((closes[-1]/float(ep)-1)*100, 3)
            if pd.notna(xp) and xp:
                r["var_day1_pct"] = round((closes[0]/float(xp)-1)*100, 3)
                r["variance_pct"] = round((closes[-1]/float(xp)-1)*100, 3)
                if pd.notna(q) and q:
                    r["var_day1_abs"] = round((closes[0]-float(xp))*float(q), 2)
                    r["variance_abs"] = round((closes[-1]-float(xp))*float(q), 2)
            for k in range(H):
                r[f"ret_d{k+1}"] = round((closes[k]/entry-1)*100, 3)
        out.append(r)
    return pd.DataFrame(out)

if bysym:
    combined = resolve(combined)
combined.to_csv(LOG, index=False)

# ---- summary ----
print(f"\nLogged {len(new)} trade(s) for watchlist {log_date} -> {LOG}" if len(new) else f"\nRefreshed -> {LOG}")
show = new["symbol"].astype(str).tolist() if len(new) else combined[combined["log_date"] == log_date]["symbol"].astype(str).tolist()
for sym in show:
    sub = combined[combined["symbol"].astype(str) == sym]
    if not len(sub):
        continue
    r = sub.iloc[-1]; st = r.get("resolve_status", "n/a")
    sig = ""
    if pd.notna(r.get("gap_pct")):
        nc = "Y" if r.get("narrow_cpr") == 1 else "n"
        sig = f"gap {r['gap_pct']:+.2f}% open {r.get('open_pos')} narrowCPR={nc} :: "
    if str(r.get("status")) == "holding":
        u = f"unreal {r['unrealized_pnl_pct']:+.2f}%" if pd.notna(r.get("unrealized_pnl_pct")) else "open"
        print(f"   {sym:12} {sig}HOLDING  {u}  (vs {r.get('last_close_date')} close)")
    elif st == "resolved":
        d1 = f"day1-close {r['var_day1_pct']:+.2f}%" if pd.notna(r.get("var_day1_pct")) else ""
        v5 = f"5-day var {r['variance_pct']:+.2f}%" if pd.notna(r.get("variance_pct")) else ""
        dur = f"held {r['hold_duration']}" if pd.notna(r.get("hold_duration")) else ""
        print(f"   {sym:12} {sig}exit {r['realized_pnl_pct']:+.2f}% {dur}  |  {d1}  {v5}")
    else:
        print(f"   {sym:12} {sig}logged -- {st} (5-day fills once window elapses)")

# ---- totals ----
ce = combined
real = pd.to_numeric(ce.loc[ce["status"] == "exited", "realized_pnl_abs"], errors="coerce").sum()
unreal = pd.to_numeric(ce.get("unrealized_pnl_abs"), errors="coerce").sum() if "unrealized_pnl_abs" in ce.columns else 0.0
print(f"\n  TOTAL realized (absorbed)   : Rs {real:+,.0f}")
print(f"  TOTAL unrealized (unabsorbed): Rs {unreal:+,.0f}   [holdings marked to latest close]")
print(f"  TOTAL combined               : Rs {real+unreal:+,.0f}")
print("\n  +variance / +day1-close = left on the table by exiting early; - = drawdown dodged.")
