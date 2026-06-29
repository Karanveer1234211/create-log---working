#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exit_lab.py  --  EXIT-STRATEGY bake-off (one-off research, OOS only)

Same entries (top-N by prob each OOS day, fixed Rs notional each), then compare
MANY exit styles on the identical trades:

  hold_1/3/5/10d      fixed horizon
  prob_horizon        higher prob -> hold longer (>=0.80:10d, >=0.70:5d, else 3d)
  sl15                hold 5d but disaster-stop at -15%
  tp10 / tp15         full take-profit at +10% / +15%, else 5d
  bracket_15_15       +15% TP / -15% SL / 5d
  partial_tp10        sell half at +10%, rest to 5d
  partial_tp10_sl15   half at +10%, remainder with -15% disaster stop
  trail10 / trail15   trailing stop from peak
  breakeven_5         after +5%, stop to entry (protect giveback)
  predlow_close       your structural exit (close below prediction-day low)

Scored per strategy on: per-trade mean/median/win/days/MAE/worst, and a
mark-to-market PORTFOLIO (fixed notional, redeploy nothing -- every top-N pick
taken) -> total Rs P&L, CAGR on peak deployed capital, and true MAX DRAWDOWN.

OOS ONLY (in_test rows). In-sample-on-OOS-window still: one ~2yr window, no
slippage/impact modelled, assumes mechanical execution. Forward log is final judge.

Usage (one line):
  python exit_lab.py --panel "C:\\...\\bigmove_scored_panel.parquet" --top-n 3 --pos 150000
  python exit_lab.py --panel "...same..." --top-n 5 --pos 150000
"""
import argparse, os, sys
import numpy as np
import pandas as pd

MAXH = 10  # forward bars to fetch (longest horizon strategy needs 10)

# ----------------------------- load (with in_test) ----------------------------
def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def load_panel(path, prob_col=None):
    df = pd.read_parquet(path)
    cols = list(df.columns)
    ren = dict(timestamp=_pick(cols, "timestamp", "date"),
               symbol=_pick(cols, "symbol", "ticker", "tradingsymbol"),
               open=_pick(cols, "open", "o"), high=_pick(cols, "high", "h"),
               low=_pick(cols, "low", "l"), close=_pick(cols, "close", "c", "adj_close"),
               prob=prob_col or _pick(cols, "prob_bigmove", "prob", "probability", "score"))
    if ren["prob"] is None:
        for c in cols:
            if "prob" in c.lower(): ren["prob"] = c; break
    miss = [k for k, v in ren.items() if v is None]
    if miss: sys.exit(f"[error] missing {miss}. found: {cols}")
    df = df.rename(columns={v: k for k, v in ren.items()})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    try:
        if getattr(df["timestamp"].dt, "tz", None) is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df["timestamp"] = df["timestamp"].dt.normalize()
    for c in ["open", "high", "low", "close", "prob"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "open", "high", "low", "close", "prob"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    itc = _pick(cols, "in_test", "is_test", "oos")
    df["in_test"] = (df[itc].astype(str).str.lower().isin(["true", "1", "yes"])
                     if itc else False)
    return df, (itc is not None)


def select_top(df, top_n):
    out = []
    for d, g in df.groupby("timestamp"):
        out.append(g.nlargest(int(top_n), "prob"))
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


# ----------------------------- exit strategies --------------------------------
# each returns (ret, k_exit_index, reason). path arrays o,h,l,c length L (<=MAXH).
def _hold(o, h, l, c, e, prob, plow, H):
    k = min(H, len(c)) - 1
    return c[k] / e - 1, k, "horizon"

def _sl(slp):
    def f(o, h, l, c, e, prob, plow, H=5):
        lvl = e * (1 - slp); L = min(H, len(c))
        for k in range(L):
            if l[k] <= lvl: return min(lvl, o[k]) / e - 1, k, "sl"
        return c[L-1] / e - 1, L-1, "horizon"
    return f

def _tp(tpp):
    def f(o, h, l, c, e, prob, plow, H=5):
        lvl = e * (1 + tpp); L = min(H, len(c))
        for k in range(L):
            if h[k] >= lvl: return max(lvl, o[k]) / e - 1, k, "tp"
        return c[L-1] / e - 1, L-1, "horizon"
    return f

def _bracket(tpp, slp):
    def f(o, h, l, c, e, prob, plow, H=5):
        tp = e*(1+tpp); sl = e*(1-slp); L = min(H, len(c))
        for k in range(L):
            if l[k] <= sl: return min(sl, o[k])/e - 1, k, "sl"      # SL checked first (pessimistic)
            if h[k] >= tp: return max(tp, o[k])/e - 1, k, "tp"
        return c[L-1]/e - 1, L-1, "horizon"
    return f

def _partial(tpp, frac, slp=None):
    def f(o, h, l, c, e, prob, plow, H=5):
        tp = e*(1+tpp); L = min(H, len(c)); sl = e*(1-slp) if slp else None
        hit_k = None
        for k in range(L):
            if sl is not None and l[k] <= sl:
                return min(sl, o[k])/e - 1, k, "sl"                 # disaster stop on whole
            if h[k] >= tp and hit_k is None:
                hit_k = k                                          # bank partial, keep rest
        rest_ret = c[L-1]/e - 1
        if hit_k is not None:
            return frac*tpp + (1-frac)*rest_ret, L-1, "partial_tp"
        return rest_ret, L-1, "horizon"
    return f

def _trail(tr):
    def f(o, h, l, c, e, prob, plow, H=5):
        L = min(H, len(c)); peak = e
        for k in range(L):
            peak = max(peak, h[k]); stop = peak*(1-tr)
            if l[k] <= stop: return min(stop, o[k])/e - 1, k, "trail"
        return c[L-1]/e - 1, L-1, "horizon"
    return f

def _breakeven(trig):
    def f(o, h, l, c, e, prob, plow, H=5):
        L = min(H, len(c)); armed = False
        for k in range(L):
            if armed and l[k] <= e: return 0.0, k, "breakeven"
            if h[k] >= e*(1+trig): armed = True
        return c[L-1]/e - 1, L-1, "horizon"
    return f

def _prob_horizon(o, h, l, c, e, prob, plow, H=5):
    HH = 10 if prob >= 0.80 else (5 if prob >= 0.70 else 3)
    k = min(HH, len(c)) - 1
    return c[k]/e - 1, k, "prob_horizon"

def _predlow_close(o, h, l, c, e, prob, plow, H=5):
    L = min(H, len(c))
    for k in range(L):
        if c[k] < plow: return c[k]/e - 1, k, "predlow"
    return c[L-1]/e - 1, L-1, "horizon"

def strategies():
    return {
        "hold_1d":        lambda *a: _hold(*a[:7], H=1),
        "hold_3d":        lambda *a: _hold(*a[:7], H=3),
        "hold_5d":        lambda *a: _hold(*a[:7], H=5),
        "hold_10d":       lambda *a: _hold(*a[:7], H=10),
        "prob_horizon":   _prob_horizon,
        "sl15":           _sl(0.15),
        "tp10":           _tp(0.10),
        "tp15":           _tp(0.15),
        "bracket_15_15":  _bracket(0.15, 0.15),
        "partial_tp10":   _partial(0.10, 0.5),
        "partial_tp10_sl15": _partial(0.10, 0.5, slp=0.15),
        "trail10":        _trail(0.10),
        "trail15":        _trail(0.15),
        "breakeven_5":    _breakeven(0.05),
        "predlow_close":  _predlow_close,
    }


# ----------------------------- simulate paths ---------------------------------
def get_paths(df, picks):
    by = {}
    for s, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        by[s] = dict(dts=g["timestamp"].values, o=g["open"].to_numpy(float),
                     h=g["high"].to_numpy(float), l=g["low"].to_numpy(float),
                     c=g["close"].to_numpy(float),
                     idx={pd.Timestamp(t): i for i, t in enumerate(g["timestamp"].values)})
    paths = []
    for _, p in picks.iterrows():
        s = p["symbol"]; pdte = pd.Timestamp(p["timestamp"]); rec = by.get(s)
        if rec is None or pdte not in rec["idx"]: continue
        i = rec["idx"][pdte]
        fwd = list(range(i+1, min(i+1+MAXH, len(rec["c"]))))
        if len(fwd) < 1: continue
        entry = rec["o"][fwd[0]]
        if not np.isfinite(entry) or entry <= 0: continue
        paths.append(dict(symbol=s, prob=p["prob"], entry=entry, plow=rec["l"][i],
                          entry_date=pd.Timestamp(rec["dts"][fwd[0]]),
                          o=rec["o"][fwd], h=rec["h"][fwd], l=rec["l"][fwd],
                          c=rec["c"][fwd], dts=[pd.Timestamp(x) for x in rec["dts"][fwd]]))
    return paths


# ----------------------------- score one strategy -----------------------------
def score(paths, fn, pos, start_capital_floor=0.0):
    rets, days = [], []
    real_by_date = {}                  # exit_date -> realized Rs pnl
    unreal_by_date = {}                # date -> open MTM Rs (sum)
    open_by_date = {}                  # date -> count open
    for p in paths:
        ret, kx, _ = fn(p["o"], p["h"], p["l"], p["c"], p["entry"], p["prob"], p["plow"], 5)
        rets.append(ret); days.append(kx + 1)
        xdate = p["dts"][kx]
        real_by_date[xdate] = real_by_date.get(xdate, 0.0) + pos * ret
        for k in range(kx):            # days open before exit -> unrealized MTM
            d = p["dts"][k]
            unreal_by_date[d] = unreal_by_date.get(d, 0.0) + pos * (p["c"][k]/p["entry"] - 1)
            open_by_date[d] = open_by_date.get(d, 0) + 1
        open_by_date[xdate] = open_by_date.get(xdate, 0) + 1
    rets = np.array(rets); days = np.array(days)
    maes = np.array([float(np.min(p["l"][:max(1,1)] )) for p in paths])  # placeholder
    # MAE per trade over the actual hold of THIS strategy is complex; report path MAE to horizon
    mae_path = np.array([float(np.min(p["l"]/p["entry"] - 1)) for p in paths])

    # portfolio MTM equity curve
    all_dates = sorted(set(list(real_by_date) + list(unreal_by_date) + list(open_by_date)))
    peak_open = max(open_by_date.values()) if open_by_date else 1
    base = max(peak_open * pos, start_capital_floor, pos)
    eq = []; realized_cum = 0.0
    for d in all_dates:
        realized_cum += real_by_date.get(d, 0.0)
        equity = base + realized_cum + unreal_by_date.get(d, 0.0)
        eq.append(equity)
    eq = pd.Series(eq, index=pd.to_datetime(all_dates))
    total_pnl = realized_cum
    if len(eq) >= 2:
        peak = eq.cummax(); maxdd = (eq/peak - 1).min()*100
        yrs = max((eq.index[-1]-eq.index[0]).days/365.25, 1e-6)
        cagr = ((base+total_pnl)/base)**(1/yrs) - 1
        cagr *= 100
    else:
        maxdd = np.nan; cagr = np.nan
    return dict(n=len(rets), mean=rets.mean()*100, median=np.median(rets)*100,
                win=(rets>0).mean()*100, days=days.mean(), mae=mae_path.mean()*100,
                worst=rets.min()*100, total_pnl=total_pnl, cagr=cagr, maxdd=maxdd,
                peak_cap=base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--prob-col", default=None)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--pos", type=float, default=150000.0)
    ap.add_argument("--all-history", action="store_true")
    ap.add_argument("--out", default="exit_lab")
    a = ap.parse_args()
    if not os.path.exists(a.panel): sys.exit(f"[error] panel not found: {a.panel}")

    print(f"[load] {a.panel}")
    df, have_oos = load_panel(a.panel, a.prob_col)
    print(f"[load] {len(df):,} rows | {df['symbol'].nunique()} symbols | "
          f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()} | in_test={'yes' if have_oos else 'NO'}")
    pool = df if a.all_history else (df[df["in_test"]] if df["in_test"].any() else df)
    if not a.all_history and df["in_test"].any():
        print(f"[OOS ] {pool['timestamp'].min().date()} -> {pool['timestamp'].max().date()} ({len(pool):,} rows)")
    picks = select_top(pool, a.top_n)
    paths = get_paths(df, picks)
    print(f"[pick] top-{a.top_n}/day -> {len(picks):,} picks | {len(paths):,} with forward bars | "
          f"pos Rs {a.pos:,.0f}\n")
    if not paths: sys.exit("[!] no resolvable picks.")

    strat = strategies()
    rows = []
    for name, fn in strat.items():
        s = score(paths, fn, a.pos)
        s = {"strategy": name, **s}
        rows.append(s)
    t = pd.DataFrame(rows)
    show = t[["strategy","n","mean","median","win","days","mae","worst","total_pnl","cagr","maxdd"]].copy()
    show = show.sort_values("cagr", ascending=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("=" * 110)
    print(f" EXIT-STRATEGY BAKE-OFF   (OOS, top-{a.top_n}/day, Rs {a.pos:,.0f}/trade, peak-capital base)")
    print("=" * 110)
    fmt = show.copy()
    for c in ["mean","median","win","mae","worst","cagr","maxdd"]:
        fmt[c] = fmt[c].round(2)
    fmt["days"] = fmt["days"].round(1)
    fmt["total_pnl"] = fmt["total_pnl"].round(0).map(lambda x: f"{x:,.0f}")
    print(fmt.to_string(index=False))

    t.to_csv(f"{a.out}_top{a.top_n}.csv", index=False)
    print(f"\n[save] {a.out}_top{a.top_n}.csv")
    print("\nREAD: cagr = annualised return on PEAK deployed capital; maxdd = true MTM drawdown.")
    print("Best = high cagr AND shallow maxdd. 'mean' favours long holds (fat tail); judge on")
    print("cagr+maxdd together. Run --top-n 3 and 5 and compare. CONFIRM the winner FORWARD.")


if __name__ == "__main__":
    main()
