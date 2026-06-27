#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structural_stop_lab.py  --  ONE-OFF STUDY (not part of the daily loop)

Question it answers (the notebook hypothesis, mechanized over ALL history):
  For top-decile model picks, is a STRUCTURAL stop healthy?
    rule A: exit if price breaks the PREDICTION-DAY candle low
    rule B: exit if price breaks the 20 EMA (measured at prediction time)
  ...tested both as CLOSE-below (wick-resistant) and TOUCH-below (intraday).

It reports the two things a handful of hand-picked trades CANNOT:
  * SAVES   : on losers (like GRMOVER -25%), how often / how much the stop rescues.
  * FALSE-STOPS: on winners (like KSHINTL +17.5%), how often the stop wicks you out
                 and how much upside that costs.

Reference bar = the prediction-day candle (the bar the model scored on), NOT the
entry candle. Entry is modeled as the NEXT session's open. Horizon = 5 sessions.

This is post-hoc EXIT-overlay analysis on the frozen scored panel. The model is
NOT touched and validation is NOT re-opened. In-sample on history -> any edge here
must still be confirmed FORWARD before trusting it.

Usage (Windows cmd, one line):
  python structural_stop_lab.py --panel "C:\\...\\bigmove_scored_panel.parquet"
Optional:
  --prob-col prob_bigmove  --top-frac 0.10  --horizon 5  --out structural_stop_report
"""

import argparse, sys, os
import numpy as np
import pandas as pd


# ----------------------------- column auto-detect -----------------------------
def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _find_prob(cols):
    c = _pick(cols, "prob_bigmove", "prob", "probability", "p_bigmove", "score")
    if c:
        return c
    for col in cols:
        if "prob" in col.lower():
            return col
    return None


def load_panel(path, prob_col=None):
    df = pd.read_parquet(path)
    cols = list(df.columns)
    ts  = _pick(cols, "timestamp", "date", "dt")
    sym = _pick(cols, "symbol", "ticker", "tradingsymbol", "scrip")
    o   = _pick(cols, "open", "o")
    h   = _pick(cols, "high", "h")
    l   = _pick(cols, "low", "l")
    c   = _pick(cols, "close", "c", "adj_close")
    pr  = prob_col or _find_prob(cols)
    missing = [n for n, v in dict(timestamp=ts, symbol=sym, open=o, high=h, low=l, close=c, prob=pr).items() if v is None]
    if missing:
        sys.exit(f"[error] panel missing required columns: {missing}\n        found: {cols}")

    df = df.rename(columns={ts: "timestamp", sym: "symbol", o: "open", h: "high",
                            l: "low", c: "close", pr: "prob"})
    # tz-strip so comparisons are clean
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    try:
        if getattr(df["timestamp"].dt, "tz", None) is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df["timestamp"] = df["timestamp"].dt.normalize()
    for cc in ["open", "high", "low", "close", "prob"]:
        df[cc] = pd.to_numeric(df[cc], errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "open", "high", "low", "close", "prob"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # 20 EMA, causal (uses only close through each row) -> known at that bar's close
    ema = _pick(list(df.columns), "ema20", "ema_20", "ema20_close")
    if ema:
        df["ema20"] = pd.to_numeric(df[ema], errors="coerce")
    else:
        df["ema20"] = df.groupby("symbol")["close"].transform(
            lambda s: s.ewm(span=20, adjust=False).mean())
    return df


# ----------------------------- pick selection ---------------------------------
def select_top(df, top_frac=0.10, top_n=None, min_prob=None):
    """Per-date top-decile (or top-N) by prob = the prediction-day rows = picks."""
    out = []
    for d, g in df.groupby("timestamp"):
        g = g.dropna(subset=["prob"])
        if g.empty:
            continue
        if top_n:
            sel = g.nlargest(int(top_n), "prob")
        else:
            cut = g["prob"].quantile(1.0 - top_frac)
            sel = g[g["prob"] >= cut]
        if min_prob is not None:
            sel = sel[sel["prob"] >= float(min_prob)]
        out.append(sel)
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


# ----------------------------- path simulation --------------------------------
def simulate(df, picks, horizon=5):
    """
    For each pick (symbol, pred_date): entry = next session open, hold up to
    `horizon` sessions. Reference levels from the prediction-day bar.
    Returns a per-pick frame with no-stop and each structural-stop outcome.
    """
    # per-symbol numpy arrays for fast forward indexing
    by = {}
    for s, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        by[s] = dict(
            dates=g["timestamp"].values,
            o=g["open"].to_numpy(float), h=g["high"].to_numpy(float),
            l=g["low"].to_numpy(float),  c=g["close"].to_numpy(float),
            ema=g["ema20"].to_numpy(float),
            idx={pd.Timestamp(t): i for i, t in enumerate(g["timestamp"].values)},
        )

    rows = []
    for _, p in picks.iterrows():
        s = p["symbol"]; pd_date = pd.Timestamp(p["timestamp"])
        rec = by.get(s)
        if rec is None or pd_date not in rec["idx"]:
            continue
        i = rec["idx"][pd_date]
        # need entry (i+1) and at least 1 forward bar; resolve as many as available up to horizon
        if i + 1 >= len(rec["c"]):
            continue  # no next session yet -> unresolved (pending)
        fwd = range(i + 1, min(i + 1 + horizon, len(rec["c"])))
        fwd = list(fwd)
        if len(fwd) < horizon:
            # not fully matured -> skip so we don't score half-baked windows
            continue

        entry = rec["o"][fwd[0]]
        if not np.isfinite(entry) or entry <= 0:
            continue
        pred_low = rec["l"][i]
        pred_ema = rec["ema"][i]
        last_close = rec["c"][fwd[-1]]
        no_stop_ret = last_close / entry - 1.0

        def run_stop(level, mode):
            # mode: 'close' exit at close on first close<level ; 'touch' exit at level on first low<level
            for k, j in enumerate(fwd):
                if mode == "close" and rec["c"][j] < level:
                    return rec["c"][j] / entry - 1.0, k + 1, True
                if mode == "touch" and rec["l"][j] < level:
                    fill = min(level, rec["o"][j])  # if gapped below, fill at open
                    return fill / entry - 1.0, k + 1, True
            return no_stop_ret, len(fwd), False  # never triggered

        a_c = run_stop(pred_low, "close")
        a_t = run_stop(pred_low, "touch")
        b_c = run_stop(pred_ema, "close")
        b_t = run_stop(pred_ema, "touch")

        rows.append(dict(
            symbol=s, pred_date=pd_date, prob=p["prob"], entry=entry,
            pred_low=pred_low, pred_ema=pred_ema,
            no_stop_ret=no_stop_ret,
            predlow_close_ret=a_c[0], predlow_close_day=a_c[1], predlow_close_hit=a_c[2],
            predlow_touch_ret=a_t[0], predlow_touch_day=a_t[1], predlow_touch_hit=a_t[2],
            ema_close_ret=b_c[0], ema_close_day=b_c[1], ema_close_hit=b_c[2],
            ema_touch_ret=b_t[0], ema_touch_day=b_t[1], ema_touch_hit=b_t[2],
        ))
    return pd.DataFrame(rows)


# ----------------------------- reporting --------------------------------------
def _rule_block(res, prefix, label):
    ret = res[f"{prefix}_ret"]; hit = res[f"{prefix}_hit"]; day = res[f"{prefix}_day"]
    ns = res["no_stop_ret"]
    win = ns > 0; los = ns <= 0
    n = len(res)
    # winners wicked out: stop hit on a winner -> upside given up
    w_hit = hit & win
    w_giveup = (ns[w_hit] - ret[w_hit])
    # losers saved: stop hit on a loser -> loss avoided (ret usually less negative)
    l_hit = hit & los
    l_save = (ret[l_hit] - ns[l_hit])
    print(f"\n--- {label} ---")
    print(f"  picks                {n}")
    print(f"  mean ret  no-stop    {ns.mean()*100:+6.2f}%")
    print(f"  mean ret  with-stop  {ret.mean()*100:+6.2f}%   (delta {(ret.mean()-ns.mean())*100:+.2f}%)")
    print(f"  total ret no-stop    {ns.sum()*100:+8.1f}%   with-stop {ret.sum()*100:+8.1f}%")
    print(f"  WINNERS  n={win.sum():4}  false-stopped {w_hit.sum():4} "
          f"({(w_hit.sum()/max(win.sum(),1))*100:4.1f}%)  avg upside given up "
          f"{(w_giveup.mean()*100 if w_hit.sum() else 0):+5.2f}%")
    print(f"  LOSERS   n={los.sum():4}  stop fired    {l_hit.sum():4} "
          f"({(l_hit.sum()/max(los.sum(),1))*100:4.1f}%)  avg loss saved      "
          f"{(l_save.mean()*100 if l_hit.sum() else 0):+5.2f}%  "
          f"(avg fire day {day[l_hit].mean():.1f})" if l_hit.sum() else
          f"  LOSERS   n={los.sum():4}  stop never fired")


def report(res, out_prefix):
    if res.empty:
        print("\n[!] no resolved picks -- nothing to score (need matured 5-session windows).")
        return
    print("\n" + "=" * 64)
    print(f" STRUCTURAL-STOP STUDY   resolved picks = {len(res)}")
    print(f" base: mean 5d ret (no stop) = {res['no_stop_ret'].mean()*100:+.2f}% | "
          f"win-rate = {(res['no_stop_ret']>0).mean()*100:.1f}%")
    print("=" * 64)

    for pref, lab in [("predlow_close", "A1  pred-day LOW, exit on CLOSE below"),
                      ("predlow_touch", "A2  pred-day LOW, exit on TOUCH below"),
                      ("ema_close",     "B1  20-EMA(@pred), exit on CLOSE below"),
                      ("ema_touch",     "B2  20-EMA(@pred), exit on TOUCH below")]:
        _rule_block(res, pref, lab)

    # 2x2 for the primary rule (pred-low close)
    ns = res["no_stop_ret"]; hit = res["predlow_close_hit"]; win = ns > 0
    print("\n--- 2x2 (rule A1: broke pred-day low on a close) ---")
    print(f"  broke & LOSER  {int((hit & ~win).sum()):4}   (good: stop saves you)")
    print(f"  broke & WINNER {int((hit &  win).sum()):4}   (bad: wicked out of a winner)")
    print(f"  held  & WINNER {int((~hit & win).sum()):4}   (good: rode it, untouched)")
    print(f"  held  & LOSER  {int((~hit & ~win).sum()):4}   (bad: stop never protected you)")

    # by probability bucket (does high-prob deserve a wider/diff stop?)
    print("\n--- rule A1 by probability bucket ---")
    res = res.copy()
    res["pb"] = pd.cut(res["prob"], [0, 0.55, 0.65, 0.75, 1.01],
                       labels=["<.55", ".55-.65", ".65-.75", ">=.75"])
    for b, g in res.groupby("pb", observed=True):
        ns_b = g["no_stop_ret"]; r_b = g["predlow_close_ret"]
        print(f"  {str(b):8} n={len(g):4}  no-stop {ns_b.mean()*100:+5.2f}%  "
              f"with-stop {r_b.mean()*100:+5.2f}%  (delta {(r_b.mean()-ns_b.mean())*100:+.2f}%)")

    path = f"{out_prefix}.csv"
    res.drop(columns=["pb"], errors="ignore").to_csv(path, index=False)
    print(f"\n[save] per-pick detail -> {path}")
    print("\nREAD ME: 'with-stop' beating 'no-stop' on TOTAL ret = the stop helps overall.")
    print("Watch the WINNERS false-stop %: high = the stop is wicking you out of KSHINTLs.")
    print("Watch LOSERS fired % + saved: high+early = it rescues GRMOVERs. Both must hold.")
    print("In-sample on history. Pick a ROBUST rule (not the exact best) and CONFIRM FORWARD.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--prob-col", default=None)
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--min-prob", type=float, default=None)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--out", default="structural_stop_report")
    a = ap.parse_args()

    if not os.path.exists(a.panel):
        sys.exit(f"[error] panel not found: {a.panel}")
    print(f"[load] {a.panel}")
    df = load_panel(a.panel, a.prob_col)
    print(f"[load] {len(df):,} rows | {df['symbol'].nunique()} symbols | "
          f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")
    picks = select_top(df, top_frac=a.top_frac, top_n=a.top_n, min_prob=a.min_prob)
    print(f"[pick] {len(picks):,} top picks "
          f"({'top-%d' % a.top_n if a.top_n else 'top-%.0f%%' % (a.top_frac*100)})")
    res = simulate(df, picks, horizon=a.horizon)
    print(f"[sim ] {len(res):,} resolved (matured {a.horizon}-session windows)")
    report(res, a.out)


if __name__ == "__main__":
    main()
