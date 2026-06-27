#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stop_research.py  --  THOROUGH comparative stop study (one-off research)

Why this exists: a per-trade MEAN-return test assumes unlimited capital and the
ability to sit through every drawdown -- which is NOT a capital-constrained,
5-day-horizon trader's reality. This compares every stop family on BOTH:

  (1) per-trade DISTRIBUTION  (mean, MEDIAN, win%, per-rule MAE, % stopped, days
      held, RETURN-PER-DAY-of-capital = capital efficiency, worst trade, Sharpe-like)
  (2) a CAPITAL-CONSTRAINED PORTFOLIO backtest: K concurrent slots, take top-prob
      picks daily, apply the stop, REDEPLOY freed capital -> equity curve, CAGR,
      MAX DRAWDOWN (now marked-to-market DAILY, not just at trade close),
      final multiple. This matches your reality: exiting a breached loser early
      frees a slot for a fresh pick.

Stop families (all: entry = next session open, horizon = 5 sessions):
  none | predlow_close | predlow_touch | ema_close | ema_touch
  fixed % sweep (-8/-12/-15/-20/-25)  |  ATR-mult sweep (1.5/2/2.5/3 x atr)
  time-only (= none, held to horizon)

IN-SAMPLE on history. Sweeping many stops and picking the best = overfitting.
Use it to see the SHAPE (does ANY family help on drawdown / capital-efficiency),
pick a ROBUST rule, and CONFIRM FORWARD before trusting it.

Usage (one line, Windows cmd):
  python stop_research.py --panel "C:\\...\\bigmove_scored_panel.parquet"
  # selection: pass EITHER --top-frac OR --top-n (top-frac wins if both given).
  # default when neither is passed = --top-frac 0.10 (top decile per day).
  [--top-frac 0.10] [--top-n 10] [--horizon 5] [--slots 5]
  [--start-capital 100000] [--out stop_research]

=== CHANGES vs prior version ===========================================
 [FIX 1] --top-frac was unreachable: --top-n defaulted to 10 (always truthy),
         so the quantile branch never ran and you always got top-10-NAMES, not
         top-10%. Now both default to None; if neither is given we use
         top-frac=0.10. select_top branches on `top_n is not None`.
 [FIX 2] Max drawdown was computed on CLOSE-only equity (slot value updated only
         when a trade closed), understating intratrade DD -- the exact metric the
         study exists to measure. Portfolio now marks every open slot to its
         daily close return on a FULL trading-day calendar, and end-of-sample
         exits are included.
 [FIX 3] predlow/ema stop levels can sit ABOVE entry on gap-down opens, logging
         a phantom ~breakeven "stop" that flatters predlow_close (the headline
         rule). Fill logic is unchanged (realistic "stopped at open"), but the
         share of such gap-throughs is now reported per family so you can judge
         the inflation. Use --carry-gapthrough to instead hold those trades to
         horizon (removes the flatter; shows the true loss).
 [FIX 4] Per-rule MAE: MAE is now measured only up to each rule's exit day, so a
         stop that exits day 1 no longer shows a day-3 low it never experienced.
 [FIX 5] A symbol could be held concurrently in two slots (dedupe was within-day
         only). Portfolio now blocks a symbol already open in any busy slot.
 [FIX 6] Code defaults now match the documented usage (slots=5, capital=100000).
========================================================================
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
    itc = _pick(list(df.columns), "in_test", "is_test", "oos", "test")
    if itc:
        df["in_test"] = df[itc].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    else:
        df["in_test"] = False
    return df, (atr is not None), (itc is not None)


def select_top(df, top_frac=None, top_n=None):
    """Per-day cross-sectional selection.
    [FIX 1] branch on `top_n is not None` so --top-frac is actually reachable.
    """
    if top_n is None and top_frac is None:
        top_frac = 0.10                      # documented default
    out = []
    for d, g in df.groupby("timestamp"):
        g = g.dropna(subset=["prob"])
        if g.empty:
            continue
        if top_frac is not None:             # top-frac WINS if both supplied
            out.append(g[g["prob"] >= g["prob"].quantile(1 - top_frac)])
        else:
            out.append(g.nlargest(int(top_n), "prob"))
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


# which rules are level-based on prediction-day low / ema (for the gap-through diag)
_PREDLOW_RULES = ("predlow_close", "predlow_touch")
_EMA_RULES     = ("ema_close", "ema_touch")


# ----------------------------- simulate per-pick paths ------------------------
def simulate(df, picks, rules, horizon=5, carry_gapthrough=False):
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
        plow_above = bool(np.isfinite(plow) and plow >= entry)   # [FIX 3] gap-through flag
        pema_above = bool(np.isfinite(pema) and pema >= entry)
        row = dict(symbol=s, pred_date=pdte, entry_date=pd.Timestamp(dts[0]),
                   prob=p["prob"], plow_above_entry=int(plow_above),
                   pema_above_entry=int(pema_above))
        for name, fn in rules.items():
            # [FIX 3] optionally carry gap-through level stops to horizon
            if carry_gapthrough and ((name in _PREDLOW_RULES and plow_above) or
                                     (name in _EMA_RULES and pema_above)):
                ret, days, hit, xdate = c[-1]/entry - 1, len(c), False, dts[-1]
            else:
                ret, days, hit, xdate = fn(o, h, l, c, dts, entry, plow, pema, atr)
            row[f"{name}__ret"] = ret
            row[f"{name}__days"] = days
            row[f"{name}__hit"] = hit
            row[f"{name}__exit"] = pd.Timestamp(xdate)
            # [FIX 4] per-rule MAE: worst low up to THIS rule's exit day only
            row[f"{name}__mae"] = float(np.min(l[:days] / entry - 1))
            # [FIX 2] daily marks for MTM: closes on held days, realized ret on exit day
            marks = {pd.Timestamp(dts[k]): float(c[k] / entry - 1) for k in range(days - 1)}
            marks[pd.Timestamp(dts[days - 1])] = float(ret)
            row[f"{name}__marks"] = marks
        recs.append(row)
    return pd.DataFrame(recs)


# ----------------------------- per-trade metrics ------------------------------
def per_trade_table(res, rules):
    rowsout = []
    for name in rules:
        r = res[f"{name}__ret"].to_numpy(float)
        d = res[f"{name}__days"].to_numpy(float)
        hit = res[f"{name}__hit"].to_numpy(bool)
        mae = res[f"{name}__mae"].to_numpy(float)          # [FIX 4] per-rule MAE
        rpd = np.where(d > 0, r / d, 0.0)                  # return per day of capital
        sharpe = r.mean() / r.std() if r.std() > 0 else 0.0
        rowsout.append(dict(
            rule=name, n=len(r),
            mean=r.mean()*100, median=np.median(r)*100,
            win=(r > 0).mean()*100, stopped=hit.mean()*100,
            days=d.mean(), ret_per_day=rpd.mean()*100,
            mean_mae=mae.mean()*100, worst=r.min()*100,
            sharpe=sharpe))
    t = pd.DataFrame(rowsout)
    # NOTE: sorted by ret_per_day (capital efficiency) -- this rewards fast exits,
    # so the top row is partly a function of holding period. Read mean/worst too.
    return t.sort_values("ret_per_day", ascending=False)


# ----------------------------- capital-constrained portfolio sim --------------
def portfolio_sim(res, name, slots=5, start_capital=100000.0):
    """K independent slots; each free slot takes the best-prob unheld pick entering
    that day; equity compounds per slot; freed capital (early stop) redeploys.

    [FIX 2] Equity is marked to market DAILY over the full trading-day calendar
            (open positions revalued at each day's close return), so max drawdown
            reflects intratrade pain, not just realized-at-close steps.
    [FIX 5] A symbol already open in any busy slot cannot be taken again.
    """
    recs = []
    for _, r in res.iterrows():
        recs.append(dict(entry=r["entry_date"], exit=r[f"{name}__exit"],
                         ret=float(r[f"{name}__ret"]), prob=float(r["prob"]),
                         symbol=r["symbol"], marks=r[f"{name}__marks"]))
    if not recs:
        return dict(rule=name, final_mult=1.0, cagr=np.nan, maxdd=np.nan, trades=0)

    recs.sort(key=lambda x: (x["entry"], -x["prob"]))      # best prob first per day
    by_entry = {}
    for rr in recs:
        by_entry.setdefault(rr["entry"], []).append(rr)
    all_dates = sorted({d for rr in recs for d in rr["marks"].keys()})

    slot_base = [start_capital / slots] * slots            # realized capital per slot
    slot_trade = [None] * slots                            # current open pick or None
    eq_curve = []

    for d in all_dates:
        # 1) realize exits at close of day d
        for s in range(slots):
            tr = slot_trade[s]
            if tr is not None and tr["exit"] == d:
                slot_base[s] *= (1 + tr["ret"]); slot_trade[s] = None
        # 2) assign free slots to today's entrants (best prob first, no dup symbol)
        cands = by_entry.get(d, [])
        held = {slot_trade[s]["symbol"] for s in range(slots) if slot_trade[s]}
        ci = 0
        for s in range(slots):
            if slot_trade[s] is None:
                while ci < len(cands) and cands[ci]["symbol"] in held:
                    ci += 1
                if ci >= len(cands):
                    break
                slot_trade[s] = cands[ci]; held.add(cands[ci]["symbol"]); ci += 1
        # 3) mark-to-market at close of day d (open slots revalued; free slots flat)
        eq = 0.0
        for s in range(slots):
            tr = slot_trade[s]
            eq += slot_base[s] if tr is None else slot_base[s] * (1 + tr["marks"].get(d, 0.0))
        eq_curve.append((d, eq))

    final = sum(slot_base)
    eq = pd.Series([e for _, e in eq_curve], index=[d for d, _ in eq_curve])
    if len(eq) < 2:
        return dict(rule=name, final_mult=final/start_capital, cagr=np.nan,
                    maxdd=np.nan, trades=len(recs))
    peak = eq.cummax(); dd = (eq / peak - 1).min()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)
    cagr = (final / start_capital) ** (1 / yrs) - 1
    return dict(rule=name, final_mult=final/start_capital, cagr=cagr*100,
                maxdd=dd*100, trades=len(recs))


# ----------------------------- main -------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--prob-col", default=None)
    ap.add_argument("--top-frac", type=float, default=None)   # [FIX 1] both None ->
    ap.add_argument("--top-n", type=int, default=None)        #         top-frac 0.10
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--slots", type=int, default=5)           # [FIX 6] match docstring
    ap.add_argument("--start-capital", type=float, default=100000.0)   # [FIX 6]
    ap.add_argument("--carry-gapthrough", action="store_true",
                    help="[FIX 3] hold gap-through level stops to horizon instead of "
                         "logging a phantom breakeven exit")
    ap.add_argument("--oos-only", action="store_true", default=True)
    ap.add_argument("--all-history", dest="oos_only", action="store_false")
    ap.add_argument("--out", default="stop_research")
    a = ap.parse_args()
    if not os.path.exists(a.panel):
        sys.exit(f"[error] panel not found: {a.panel}")
    if a.top_frac is not None and a.top_n is not None:
        print("[warn] both --top-frac and --top-n given; --top-frac wins.")

    print(f"[load] {a.panel}")
    df, have_atr, have_oos = load_panel(a.panel, a.prob_col)
    print(f"[load] {len(df):,} rows | {df['symbol'].nunique()} symbols | "
          f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()} | "
          f"atr={'yes' if have_atr else 'NO'} | in_test col={'yes' if have_oos else 'NO'}")

    # pick pool = OOS rows only (uncontaminated); paths still simulated on FULL panel
    if a.oos_only and have_oos and df["in_test"].any():
        pool = df[df["in_test"]]
        print(f"[OOS ] picks restricted to in_test rows: {pool['timestamp'].min().date()} "
              f"-> {pool['timestamp'].max().date()}  ({len(pool):,} rows)")
    else:
        pool = df
        if a.oos_only and not df["in_test"].any():
            print("[warn] --oos-only requested but no in_test rows found; using all history.")

    picks = select_top(pool, top_frac=a.top_frac, top_n=a.top_n)
    rules = make_rules(have_atr)
    if a.top_frac is not None:
        sel = f"top-{a.top_frac*100:.0f}%"
    elif a.top_n is not None:
        sel = f"top-{a.top_n}"
    else:
        sel = "top-10%"                       # the neither-given default
    print(f"[pick] {len(picks):,} picks ({sel}/day) | {len(rules)} rules | horizon {a.horizon}")
    res = simulate(df, picks, rules, horizon=a.horizon, carry_gapthrough=a.carry_gapthrough)
    print(f"[sim ] {len(res):,} resolved  (OOS span "
          f"{res['entry_date'].min().date()} -> {res['entry_date'].max().date()})\n")
    if res.empty:
        sys.exit("[!] nothing resolved.")

    pt = per_trade_table(res, rules)
    pd.set_option("display.width", 170); pd.set_option("display.max_columns", 20)
    print("=" * 92)
    print(f" PER-TRADE  (OOS, {sel}/day)  sorted by RETURN-PER-DAY of capital")
    print("=" * 92)
    print(pt.round(2).to_string(index=False))

    # [FIX 3] gap-through diagnostic: how often was the level stop ABOVE entry?
    n = len(res)
    pl_gt = res["plow_above_entry"].mean() * 100
    pe_gt = res["pema_above_entry"].mean() * 100
    print("\n[gap-through] prediction-day stop level sat ABOVE next-open entry on:")
    print(f"   predlow rules : {pl_gt:5.1f}% of trades   ema rules : {pe_gt:5.1f}% of trades")
    if not a.carry_gapthrough:
        print("   -> these are logged as ~breakeven exits (realistic 'stopped at open'),")
        print("      which FLATTERS predlow_close/ema_close. Re-run with --carry-gapthrough")
        print("      to hold them to horizon and see the un-flattered numbers.")
    else:
        print("   -> --carry-gapthrough ON: these are held to horizon (un-flattered).")

    print("\n" + "=" * 92)
    print(f" CAPITAL-CONSTRAINED PORTFOLIO  ({a.slots} slots, redeploy freed capital, "
          f"start Rs {a.start_capital:,.0f}, OOS, {sel}/day, daily MTM)")
    print("=" * 92)
    port = pd.DataFrame([portfolio_sim(res, name, a.slots, a.start_capital) for name in rules])
    port = port.sort_values("cagr", ascending=False)
    print(port.round(2).to_string(index=False))

    # --- slot sensitivity for the rules you care about (robustness check) ---
    print("\n" + "=" * 60)
    print(" CAGR sensitivity to slot count (robustness)")
    print("=" * 60)
    key = [r for r in ["predlow_close", "none", "fixed_25pct", "predlow_touch", "ema_close"] if r in rules]
    hdr = "  rule".ljust(18) + "".join(f"{k:>10}" for k in ["slots=3", "slots=5", "slots=10"])
    print(hdr)
    for name in key:
        cells = []
        for k in (3, 5, 10):
            cells.append(portfolio_sim(res, name, k, a.start_capital)["cagr"])
        print(f"  {name:16}" + "".join(f"{c:>9.1f}%" for c in cells))

    # headline number
    hl = portfolio_sim(res, "predlow_close", a.slots, a.start_capital)
    print("\n" + "-" * 60)
    print(f" HEADLINE  (your setup: predlow_close stop, OOS, {sel}/day, "
          f"{a.slots} slots, Rs {a.start_capital:,.0f})")
    print(f"   CAGR        {hl['cagr']:+.1f}%")
    print(f"   final mult  {hl['final_mult']:.2f}x   (Rs {a.start_capital*hl['final_mult']:,.0f})")
    print(f"   max drawdown {hl['maxdd']:.1f}%   over {hl['trades']:,} trades  (daily MTM)")
    print("-" * 60)

    save = res.drop(columns=[c for c in res.columns if c.endswith("__marks")])  # drop dicts
    pt.to_csv(f"{a.out}_pertrade.csv", index=False)
    port.to_csv(f"{a.out}_portfolio.csv", index=False)
    save.to_csv(f"{a.out}_picks.csv", index=False)
    print(f"\n[save] {a.out}_pertrade.csv | {a.out}_portfolio.csv | {a.out}_picks.csv")
    print("\nNOTE: OOS still has real caveats -- one ~2yr OOS window, slippage/impact NOT")
    print("modelled, and CAGR assumes you actually execute every signal mechanically.")
    print("A robust rule holds its rank across slots=3/5/10. Forward log is the final judge.")


if __name__ == "__main__":
    main()
