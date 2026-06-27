#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stop_research.py  --  THOROUGH comparative stop study (one-off research)

Why this exists: a per-trade MEAN-return test assumes unlimited capital and the
ability to sit through every drawdown -- which is NOT a capital-constrained,
5-day-horizon trader's reality. This compares every stop family on BOTH:

  (1) per-trade DISTRIBUTION  (mean, MEDIAN, win%, MAE, % stopped, days held,
      RETURN-PER-DAY-of-capital = capital efficiency, worst trade, Sharpe-like)
  (2) a CAPITAL-CONSTRAINED PORTFOLIO backtest: K concurrent slots, take top-prob
      picks daily, apply the stop, REDEPLOY freed capital -> equity curve, CAGR,
      MAX DRAWDOWN, final multiple. This is the metric that matches your reality:
      exiting a breached loser early frees a slot for a fresh pick.

Stop families (all: entry = next session open, horizon = 5 sessions):
  none | predlow_close | predlow_touch | ema_close | ema_touch
  fixed % sweep (-8/-12/-15/-20/-25)  |  ATR-mult sweep (1.5/2/2.5/3 x atr)
  time-only (= none, held to horizon)

IN-SAMPLE on history. Sweeping many stops and picking the best = overfitting.
Use it to see the SHAPE (does ANY family help on drawdown / capital-efficiency),
pick a ROBUST rule, and CONFIRM FORWARD before trusting it.

Usage (one line, Windows cmd):
  python stop_research.py --panel "C:\\...\\bigmove_scored_panel.parquet"
  [--top-frac 0.10] [--horizon 5] [--slots 5] [--start-capital 100000] [--out stop_research]
"""
import argparse, os, sys
import numpy as np
import pandas as pd


# ----------------------------- load -------------------------------------------
def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def load_panel(path, prob_col=None):
    df = pd.read_parquet(path)
    cols = list(df.columns)
    ren = dict(timestamp=_pick(cols, "timestamp", "date", "dt"),
               symbol=_pick(cols, "symbol", "ticker", "tradingsymbol"),
               open=_pick(cols, "open", "o"), high=_pick(cols, "high", "h"),
               low=_pick(cols, "low", "l"), close=_pick(cols, "close", "c", "adj_close"),
               prob=prob_col or _pick(cols, "prob_bigmove", "prob", "probability", "score"))
    if ren["prob"] is None:
        for c in cols:
            if "prob" in c.lower():
                ren["prob"] = c; break
    miss = [k for k, v in ren.items() if v is None]
    if miss:
        sys.exit(f"[error] missing columns {miss}. found: {cols}")
    inv = {v: k for k, v in ren.items()}
    df = df.rename(columns=inv)
    atr = _pick(cols, "atr_pct", "atrp", "atr_percent")
    df["atr_pct"] = pd.to_numeric(df[atr], errors="coerce") if atr else np.nan
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
    ema = _pick(list(df.columns), "ema20", "ema_20")
    df["ema20"] = pd.to_numeric(df[ema], errors="coerce") if ema else \
        df.groupby("symbol")["close"].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    return df, (atr is not None)


def select_top(df, top_frac=0.10, top_n=None):
    out = []
    for d, g in df.groupby("timestamp"):
        g = g.dropna(subset=["prob"])
        if g.empty:
            continue
        out.append(g.nlargest(int(top_n), "prob") if top_n
                   else g[g["prob"] >= g["prob"].quantile(1 - top_frac)])
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


# ----------------------------- stop rules -------------------------------------
def make_rules(have_atr):
    rules = {}
    def none(o, h, l, c, dts, entry, plow, pema, atr):
        return c[-1] / entry - 1, len(c), False, dts[-1]
    rules["none"] = none

    def level_close(level):
        def f(o, h, l, c, dts, entry, plow, pema, atr):
            for k in range(len(c)):
                if c[k] < level: return c[k] / entry - 1, k + 1, True, dts[k]
            return c[-1] / entry - 1, len(c), False, dts[-1]
        return f
    def level_touch(level):
        def f(o, h, l, c, dts, entry, plow, pema, atr):
            for k in range(len(c)):
                if l[k] < level:
                    fill = min(level, o[k]); return fill / entry - 1, k + 1, True, dts[k]
            return c[-1] / entry - 1, len(c), False, dts[-1]
        return f

    rules["predlow_close"] = lambda o,h,l,c,d,e,pl,pe,a: level_close(pl)(o,h,l,c,d,e,pl,pe,a)
    rules["predlow_touch"] = lambda o,h,l,c,d,e,pl,pe,a: level_touch(pl)(o,h,l,c,d,e,pl,pe,a)
    rules["ema_close"]     = lambda o,h,l,c,d,e,pl,pe,a: level_close(pe)(o,h,l,c,d,e,pl,pe,a)
    rules["ema_touch"]     = lambda o,h,l,c,d,e,pl,pe,a: level_touch(pe)(o,h,l,c,d,e,pl,pe,a)

    for x in (0.08, 0.12, 0.15, 0.20, 0.25):
        rules[f"fixed_{int(x*100)}pct"] = (
            lambda o,h,l,c,d,e,pl,pe,a, X=x: level_touch(e*(1-X))(o,h,l,c,d,e,pl,pe,a))
    if have_atr:
        for k in (1.5, 2.0, 2.5, 3.0):
            def atr_rule(o,h,l,c,d,e,pl,pe,a, K=k):
                if not np.isfinite(a) or a <= 0:                  # no atr -> no stop
                    return c[-1]/e-1, len(c), False, d[-1]
                return level_touch(e*(1 - K*a/100.0))(o,h,l,c,d,e,pl,pe,a)
            rules[f"atr_{k:g}x"] = atr_rule
    return rules


# ----------------------------- simulate per-pick paths ------------------------
def simulate(df, picks, rules, horizon=5):
    by = {}
    for s, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        by[s] = dict(dts=g["timestamp"].values, o=g["open"].to_numpy(float),
                     h=g["high"].to_numpy(float), l=g["low"].to_numpy(float),
                     c=g["close"].to_numpy(float), ema=g["ema20"].to_numpy(float),
                     idx={pd.Timestamp(t): i for i, t in enumerate(g["timestamp"].values)})
    recs = []
    for _, p in picks.iterrows():
        s = p["symbol"]; pdte = pd.Timestamp(p["timestamp"]); rec = by.get(s)
        if rec is None or pdte not in rec["idx"]:
            continue
        i = rec["idx"][pdte]
        fwd = list(range(i + 1, min(i + 1 + horizon, len(rec["c"]))))
        if len(fwd) < horizon:
            continue
        entry = rec["o"][fwd[0]]
        if not np.isfinite(entry) or entry <= 0:
            continue
        o = rec["o"][fwd]; h = rec["h"][fwd]; l = rec["l"][fwd]; c = rec["c"][fwd]
        dts = rec["dts"][fwd]
        plow = rec["l"][i]; pema = rec["ema"][i]; atr = p.get("atr_pct", np.nan)
        mae = float(np.min(l / entry - 1))
        row = dict(symbol=s, pred_date=pdte, entry_date=pd.Timestamp(dts[0]),
                   prob=p["prob"], mae=mae)
        for name, fn in rules.items():
            ret, days, hit, xdate = fn(o, h, l, c, dts, entry, plow, pema, atr)
            row[f"{name}__ret"] = ret
            row[f"{name}__days"] = days
            row[f"{name}__hit"] = hit
            row[f"{name}__exit"] = pd.Timestamp(xdate)
        recs.append(row)
    return pd.DataFrame(recs)


# ----------------------------- per-trade metrics ------------------------------
def per_trade_table(res, rules):
    rowsout = []
    for name in rules:
        r = res[f"{name}__ret"].to_numpy(float)
        d = res[f"{name}__days"].to_numpy(float)
        hit = res[f"{name}__hit"].to_numpy(bool)
        rpd = np.where(d > 0, r / d, 0.0)          # return per day of capital
        sharpe = r.mean() / r.std() if r.std() > 0 else 0.0
        rowsout.append(dict(
            rule=name, n=len(r),
            mean=r.mean()*100, median=np.median(r)*100,
            win=(r > 0).mean()*100, stopped=hit.mean()*100,
            days=d.mean(), ret_per_day=rpd.mean()*100,
            mean_mae=res["mae"].mean()*100, worst=r.min()*100,
            sharpe=sharpe))
    t = pd.DataFrame(rowsout)
    return t.sort_values("ret_per_day", ascending=False)


# ----------------------------- capital-constrained portfolio sim --------------
def portfolio_sim(res, name, slots=5, start_capital=100000.0):
    """K independent slots; each free slot takes the best-prob unheld pick entering
    that day; equity compounds per slot; freed capital (early stop) redeploys."""
    picks = res[["entry_date", f"{name}__exit", f"{name}__ret", "prob", "symbol"]].copy()
    picks.columns = ["entry", "exit", "ret", "prob", "symbol"]
    picks = picks.sort_values(["entry", "prob"], ascending=[True, False]).reset_index(drop=True)
    by_entry = {d: g for d, g in picks.groupby("entry")}
    dates = sorted(picks["entry"].unique())

    slot_eq = [start_capital / slots] * slots
    slot_busy_until = [pd.Timestamp.min] * slots
    pending = {}   # exit_date -> list of (slot, ret)
    equity_curve = []
    taken_today = set()

    for d in dates:
        # 1) process exits due on/before d
        for xd in [x for x in list(pending) if x <= d]:
            for (slot, ret) in pending.pop(xd):
                slot_eq[slot] *= (1 + ret)
                slot_busy_until[slot] = pd.Timestamp.min
        # 2) assign free slots to today's best unheld candidates
        cands = by_entry.get(d)
        if cands is not None:
            ci = 0; carr = cands.to_dict("records")
            for slot in range(slots):
                if slot_busy_until[slot] <= d:                      # free
                    while ci < len(carr) and carr[ci]["symbol"] in taken_today:
                        ci += 1
                    if ci >= len(carr):
                        break
                    pk = carr[ci]; ci += 1
                    taken_today.add(pk["symbol"])
                    slot_busy_until[slot] = pk["exit"]
                    pending.setdefault(pk["exit"], []).append((slot, pk["ret"]))
            taken_today.clear()
        equity_curve.append((d, sum(slot_eq)))
    # flush remaining
    for xd in sorted(pending):
        for (slot, ret) in pending[xd]:
            slot_eq[slot] *= (1 + ret)
    final = sum(slot_eq)
    eq = pd.Series([e for _, e in equity_curve], index=[d for d, _ in equity_curve])
    if len(eq) < 2:
        return dict(rule=name, final_mult=final/start_capital, cagr=np.nan, maxdd=np.nan, trades=len(picks))
    peak = eq.cummax(); dd = (eq/peak - 1).min()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)
    cagr = (final/start_capital)**(1/yrs) - 1
    return dict(rule=name, final_mult=final/start_capital, cagr=cagr*100,
                maxdd=dd*100, trades=len(picks))


# ----------------------------- main -------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--prob-col", default=None)
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--start-capital", type=float, default=100000.0)
    ap.add_argument("--out", default="stop_research")
    a = ap.parse_args()
    if not os.path.exists(a.panel):
        sys.exit(f"[error] panel not found: {a.panel}")

    print(f"[load] {a.panel}")
    df, have_atr = load_panel(a.panel, a.prob_col)
    print(f"[load] {len(df):,} rows | {df['symbol'].nunique()} symbols | "
          f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()} | atr={'yes' if have_atr else 'NO'}")
    picks = select_top(df, top_frac=a.top_frac, top_n=a.top_n)
    rules = make_rules(have_atr)
    print(f"[pick] {len(picks):,} picks | {len(rules)} stop rules | horizon {a.horizon} | slots {a.slots}")
    res = simulate(df, picks, rules, horizon=a.horizon)
    print(f"[sim ] {len(res):,} resolved\n")
    if res.empty:
        sys.exit("[!] nothing resolved.")

    pt = per_trade_table(res, rules)
    pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)
    print("=" * 92)
    print(" PER-TRADE (sorted by RETURN-PER-DAY of capital = capital efficiency)")
    print("=" * 92)
    print(pt.round(2).to_string(index=False))

    print("\n" + "=" * 92)
    print(f" CAPITAL-CONSTRAINED PORTFOLIO  ({a.slots} slots, redeploy freed capital, "
          f"start Rs {a.start_capital:,.0f})")
    print("=" * 92)
    port = pd.DataFrame([portfolio_sim(res, name, a.slots, a.start_capital) for name in rules])
    port = port.sort_values("final_mult", ascending=False)
    print(port.round(2).to_string(index=False))

    pt.to_csv(f"{a.out}_pertrade.csv", index=False)
    port.to_csv(f"{a.out}_portfolio.csv", index=False)
    res.to_csv(f"{a.out}_picks.csv", index=False)
    print(f"\n[save] {a.out}_pertrade.csv | {a.out}_portfolio.csv | {a.out}_picks.csv")
    print("\nHOW TO READ:")
    print(" * per-trade MEAN can favour 'none' (fat-tail winners) while a stop wins on")
    print("   MEDIAN, RET-PER-DAY (capital efficiency) and portfolio MAX-DRAWDOWN.")
    print(" * the PORTFOLIO table is the capital-aware verdict: higher final_mult / cagr")
    print("   AND shallower maxdd = the stop that actually serves a constrained 5d trader.")
    print(" * IN-SAMPLE. Pick a ROBUST rule (not the single best cell), confirm FORWARD.")


if __name__ == "__main__":
    main()
