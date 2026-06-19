#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
prediction_journal.py  -  Live prediction journal + loser analysis for the model.

Sits AFTER the model. Does NOT retrain or backtest. Builds a research database of
live predictions + what happened next, so you can study why winners win and losers
lose on genuinely out-of-sample (post-training) data.

THREE MODES
-----------
  record   Snapshot today's predictions: pred_date, symbol, prob, rank, entry_ref,
           and the FULL features_train.json feature vector for each name. Appends
           PENDING rows to the journal (parquet). Safe to re-run (dedupes on
           pred_date+symbol).

  update   For rows whose 1/3/5-day windows have elapsed in panel_cache, fill in
           ret_1/3/5, MFE/MAE per horizon, and event flags (abnormal gap, volume
           spike, large daily move). Idempotent.

  analyze  The research layer on RESOLVED rows:
             - performance by probability bucket (do high probs outperform?)
             - winners vs losers feature association (Cohen's d, ranked)
             - catastrophic-loser profile (what the worst decile shares)
             - event contribution (do gaps/spikes drive losses?)
             - candidate FILTER hypotheses (with vs without, n shown)

READ THIS (the one way to misuse it)
------------------------------------
With ~191 features and a finite live sample, SOME features and filters will look
predictive by pure chance. Everything 'analyze' surfaces is a HYPOTHESIS, not a
rule. A filter that helps here must be confirmed on a fresh OOS slice before it
ever touches the model or your selection. This tool generates leads; it does not
license deployment. It also measures the MODEL's clean trade (next-open entry,
held to horizon) -> that isolates PREDICTION quality. Execution quality is the gap
vs your actual fills, which lives in trade_logger.py; cross-ref the two to answer
"bad prediction or bad execution".

USAGE (cmd, full args)
----------------------
  record:
    python prediction_journal.py --mode record ^
      --source watchlist --watchlist "...\bigmove_deploy\bigmove_watchlist.csv" ^
      --panel "...\out_rank_single\panel_cache.parquet" ^
      --features "...\out_rank_single\features_train.json" ^
      --journal "...\prediction_journal.parquet"
    (or --source scored --scored "...\bigmove_scored_panel.parquet" --min-prob 0.50)

  update:   python prediction_journal.py --mode update  --panel "...\panel_cache.parquet" --journal "...\prediction_journal.parquet"
  analyze:  python prediction_journal.py --mode analyze --journal "...\prediction_journal.parquet"

Self-test (no files needed):  python prediction_journal.py --self-test
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5)
# event thresholds (fraction)
GAP_THRESH = 0.04        # |open/prev_close - 1|
VOL_MULT = 3.0           # volume / 20d avg
MOVE_THRESH = 0.06       # |close/prev_close - 1|
CAT_LOSER = -0.08        # ret_5 below this = catastrophic loser
WIN_THRESH = 0.0         # ret_5 above this = winner


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    for c in cands:
        for col in cols:
            if c in col.lower():
                return col
    return None


def _norm_day(s):
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()


def load_features(path):
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict):
        feats = obj.get("features") or obj.get("feature_list") or list(obj.keys())
        if "impute" in obj and isinstance(obj["impute"], dict) and feats == list(obj.keys()):
            feats = list(obj["impute"].keys())
    else:
        feats = list(obj)
    return [str(f) for f in feats]


def load_panel(path):
    df = pd.read_parquet(path)
    dcol = _pick(df.columns, "timestamp", "date")
    scol = _pick(df.columns, "symbol", "ticker", "sym")
    df = df.rename(columns={dcol: "date", scol: "symbol"})
    df["date"] = _norm_day(df["date"])
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise SystemExit(f"panel missing OHLC column '{c}'")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def sym_frame(panel, sym):
    g = panel[panel["symbol"] == sym].sort_values("date").reset_index(drop=True)
    g["avg_vol20"] = g["volume"].rolling(20, min_periods=5).mean().shift(1)
    return g


def load_journal(path):
    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def save_journal(df, path):
    df.to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #
def mode_record(a):
    feats = load_features(a.features)
    panel = load_panel(a.panel)

    if a.source == "watchlist":
        w = pd.read_csv(a.watchlist)
        scol = _pick(w.columns, "symbol", "ticker", "sym")
        pcol = _pick(w.columns, "prob", "score", "p_bigmove")
        dcol = _pick(w.columns, "date", "timestamp", "pred_date")
        rcol = _pick(w.columns, "rank")
        w = w.rename(columns={scol: "symbol", pcol: "prob"})
        pred_day = (_norm_day(w[dcol]).max() if dcol else panel["date"].max())
        w["rank"] = w[rcol] if rcol else np.arange(1, len(w) + 1)
        preds = w[["symbol", "prob", "rank"]].copy()
    else:
        sc = pd.read_parquet(a.scored)
        dcol = _pick(sc.columns, "timestamp", "date")
        scol = _pick(sc.columns, "symbol")
        pcol = _pick(sc.columns, "prob", "prob_5d_mean", "score")
        sc = sc.rename(columns={dcol: "date", scol: "symbol", pcol: "prob"})
        sc["date"] = _norm_day(sc["date"])
        pred_day = sc["date"].max()
        sc = sc[(sc["date"] == pred_day) & (sc["prob"] >= a.min_prob)]
        sc = sc.sort_values("prob", ascending=False)
        if a.top_n:
            sc = sc.head(a.top_n)
        sc["rank"] = np.arange(1, len(sc) + 1)
        preds = sc[["symbol", "prob", "rank"]].copy()

    if preds.empty:
        print(f"[record] no predictions to record on {pred_day.date()}")
        return

    feat_cols = [f for f in feats if f in panel.columns]
    missing = [f for f in feats if f not in panel.columns]
    if missing:
        print(f"[record] WARNING: {len(missing)} features not in panel (stored blank), e.g. {missing[:4]}")

    jrn = load_journal(a.journal)
    existing = set((jrn["pred_date"].astype(str) + "_" + jrn["symbol"].astype(str))) if len(jrn) else set()

    rows = []
    pday = panel[panel["date"] == pred_day]
    for _, r in preds.iterrows():
        sym = r["symbol"]
        key = f"{pred_day.date()}_{sym}"
        if key in existing:
            continue
        prow = pday[pday["symbol"] == sym]
        if prow.empty:
            print(f"[record] {sym}: no panel row on {pred_day.date()}, skipped")
            continue
        prow = prow.iloc[0]
        rec = {"pred_date": pred_day.date(), "symbol": sym, "prob": round(float(r["prob"]), 4),
               "rank": int(r["rank"]), "entry_ref": round(float(prow["close"]), 4),
               "stock_regime": prow.get("stock_regime", ""), "status": "PENDING"}
        for f in feats:
            rec[f"feat__{f}"] = float(prow[f]) if f in feat_cols and pd.notna(prow[f]) else np.nan
        rows.append(rec)

    if not rows:
        print("[record] nothing new (already recorded today).")
        return
    out = pd.concat([jrn, pd.DataFrame(rows)], ignore_index=True) if len(jrn) else pd.DataFrame(rows)
    save_journal(out, a.journal)
    print(f"[record] {pred_day.date()}: stored {len(rows)} prediction(s) with {len(feat_cols)} features each.")


# --------------------------------------------------------------------------- #
# update
# --------------------------------------------------------------------------- #
def _events(seg, prev_close):
    """seg: held window rows (DataFrame). prev_close: close before entry day."""
    closes = seg["close"].to_numpy("float64")
    opens = seg["open"].to_numpy("float64")
    vols = seg["volume"].to_numpy("float64")
    avg = seg["avg_vol20"].to_numpy("float64")
    pc = np.concatenate([[prev_close], closes[:-1]])
    gaps = np.abs(opens / pc - 1.0)
    moves = np.abs(closes / pc - 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        volr = vols / avg
    gap_max = float(np.nanmax(gaps)) if len(gaps) else np.nan
    move_max = float(np.nanmax(moves)) if len(moves) else np.nan
    vol_max = float(np.nanmax(volr)) if np.isfinite(volr).any() else np.nan
    return dict(
        ev_gap=int(gap_max > GAP_THRESH) if gap_max == gap_max else 0,
        ev_vol_spike=int(vol_max > VOL_MULT) if vol_max == vol_max else 0,
        ev_large_move=int(move_max > MOVE_THRESH) if move_max == move_max else 0,
        gap_max_pct=round(gap_max * 100, 2) if gap_max == gap_max else np.nan,
        vol_spike_max=round(vol_max, 2) if vol_max == vol_max else np.nan,
        move_max_pct=round(move_max * 100, 2) if move_max == move_max else np.nan,
    )


def mode_update(a):
    jrn = load_journal(a.journal)
    if jrn.empty:
        print("[update] empty journal.")
        return
    jrn = jrn.astype(object)
    panel = load_panel(a.panel)
    maxh = max(HORIZONS)
    done = 0
    for i, r in jrn[jrn["status"] != "RESOLVED"].iterrows():
        sym = r["symbol"]
        pday = pd.Timestamp(r["pred_date"]).normalize()
        g = sym_frame(panel, sym)
        fwd = g[g["date"] > pday].reset_index(drop=True)
        if len(fwd) < maxh:
            continue
        entry = float(fwd["open"].iloc[0])
        prev_close = float(g[g["date"] <= pday]["close"].iloc[-1])
        jrn.loc[i, "entry_open"] = round(entry, 4)
        jrn.loc[i, "entry_date"] = fwd["date"].iloc[0].date()
        for h in HORIZONS:
            seg = fwd.iloc[:h]
            cl = float(seg["close"].iloc[-1])
            mfe = float(seg["high"].max() / entry - 1.0)
            mae = float(seg["low"].min() / entry - 1.0)
            jrn.loc[i, f"ret_{h}"] = round((cl / entry - 1.0) * 100, 3)
            jrn.loc[i, f"mfe_{h}"] = round(mfe * 100, 2)
            jrn.loc[i, f"mae_{h}"] = round(mae * 100, 2)
        for k, v in _events(fwd.iloc[:maxh], prev_close).items():
            jrn.loc[i, k] = v
        jrn.loc[i, "ev_any"] = int(bool(jrn.loc[i, "ev_gap"] or jrn.loc[i, "ev_vol_spike"]
                                        or jrn.loc[i, "ev_large_move"]))
        jrn.loc[i, "status"] = "RESOLVED"
        done += 1
    save_journal(jrn, a.journal)
    print(f"[update] resolved {done} prediction(s). Total resolved: {(jrn['status']=='RESOLVED').sum()}")


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def _cohend(win, los):
    a, b = win.dropna(), los.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def mode_analyze(a):
    jrn = load_journal(a.journal)
    res = jrn[jrn["status"] == "RESOLVED"].copy() if "status" in jrn else pd.DataFrame()
    if len(res) < 10:
        print(f"[analyze] only {len(res)} resolved rows. Accumulate more before trusting anything below.")
        if res.empty:
            return
    for c in ("prob", "ret_1", "ret_3", "ret_5", "mfe_5", "mae_5"):
        if c in res:
            res[c] = pd.to_numeric(res[c], errors="coerce")
    n = len(res)
    print(f"\n=== SAMPLE: {n} resolved live predictions "
          f"({res['pred_date'].min()} -> {res['pred_date'].max()}) ===")
    print("    NOTE: everything below is a hypothesis on a finite live sample. Confirm OOS before acting.\n")

    # 1) by probability bucket
    res["bucket"] = (np.floor(res["prob"] / 0.05) * 0.05).round(2)
    print("=== PERFORMANCE BY PROB BUCKET (ret_5) ===")
    print(f"  {'bucket':>8} {'n':>5} {'win%':>6} {'mean':>8} {'median':>8} {'mae_med':>8}")
    for b, d in res.groupby("bucket"):
        print(f"  {b:>8.2f} {len(d):>5} {(d['ret_5']>WIN_THRESH).mean()*100:>5.0f}% "
              f"{d['ret_5'].mean():>+8.2f} {d['ret_5'].median():>+8.2f} {d['mae_5'].median():>+8.2f}")

    # 1b) by rank band - does rank-1 actually beat rank-20 within the slice?
    if "rank" in res:
        rk = pd.to_numeric(res["rank"], errors="coerce")
        bands = [(1, 5), (6, 10), (11, 15), (16, 20)]
        print("\n=== PERFORMANCE BY RANK BAND (ret_5) ===")
        print(f"  {'rank':>8} {'n':>5} {'win%':>6} {'mean':>8} {'median':>8}")
        for lo, hi in bands:
            d = res[(rk >= lo) & (rk <= hi)]
            if len(d):
                print(f"  {f'{lo}-{hi}':>8} {len(d):>5} {(d['ret_5']>WIN_THRESH).mean()*100:>5.0f}% "
                      f"{d['ret_5'].mean():>+8.2f} {d['ret_5'].median():>+8.2f}")
        print("  (if these bands are flat, rank within the top-20 carries little extra info)")

    # 2) winners vs losers feature association
    win = res[res["ret_5"] > WIN_THRESH]
    los = res[res["ret_5"] <= WIN_THRESH]
    fcols = [c for c in res.columns if c.startswith("feat__")]
    assoc = []
    for c in fcols:
        d = _cohend(pd.to_numeric(win[c], errors="coerce"), pd.to_numeric(los[c], errors="coerce"))
        if d == d:
            assoc.append((c.replace("feat__", ""), d))
    assoc.sort(key=lambda x: -abs(x[1]))
    print(f"\n=== FEATURES: winners (n={len(win)}) vs losers (n={len(los)}), by Cohen's d ===")
    print("  (+d = higher in winners, -d = higher in losers; |d|>0.5 notable, but multiple-testing applies)")
    for name, d in assoc[:12]:
        arrow = "winners" if d > 0 else "losers "
        print(f"  {d:>+6.2f}  higher in {arrow}  {name}")

    # 3) catastrophic loser profile
    cat = res[res["ret_5"] < CAT_LOSER]
    rest = res[res["ret_5"] >= CAT_LOSER]
    print(f"\n=== CATASTROPHIC LOSERS (ret_5 < {CAT_LOSER*100:.0f}%): n={len(cat)} "
          f"({len(cat)/n*100:.0f}% of sample) ===")
    if len(cat) >= 5:
        prof = []
        for c in fcols:
            d = _cohend(pd.to_numeric(cat[c], errors="coerce"), pd.to_numeric(rest[c], errors="coerce"))
            if d == d:
                prof.append((c.replace("feat__", ""), d))
        prof.sort(key=lambda x: -abs(x[1]))
        for name, d in prof[:8]:
            print(f"  {d:>+6.2f}  {'high' if d>0 else 'low '} in big losers  {name}")
        if "ev_any" in cat:
            print(f"  event-flagged share: big losers {pd.to_numeric(cat['ev_any'],errors='coerce').mean()*100:.0f}% "
                  f"vs rest {pd.to_numeric(rest['ev_any'],errors='coerce').mean()*100:.0f}%")

    # 4) event contribution
    if "ev_any" in res:
        res["ev_any"] = pd.to_numeric(res["ev_any"], errors="coerce").fillna(0)
        print("\n=== EVENT CONTRIBUTION (gap / vol spike / large move in window) ===")
        for lab, d in [("event", res[res["ev_any"] == 1]), ("no-event", res[res["ev_any"] == 0])]:
            if len(d):
                print(f"  {lab:>9}: n={len(d):>4}  mean ret_5 {d['ret_5'].mean():>+6.2f}  "
                      f"win {(d['ret_5']>0).mean()*100:>4.0f}%  loser-rate {(d['ret_5']<=0).mean()*100:>4.0f}%")

    # 5) candidate filter hypotheses - PRE-TRADE FEATURES ONLY (known at entry)
    print("\n=== FILTER HYPOTHESES (pre-trade features only; cohort ret_5; HYPOTHESIS ONLY) ===")
    base = res["ret_5"].mean()
    print(f"  baseline (all):                  n={n:>4}  mean {base:>+6.2f}")
    for name, d in assoc[:3]:
        col = f"feat__{name}"
        v = pd.to_numeric(res[col], errors="coerce")
        if v.notna().sum() < 10:
            continue
        keep = res[v >= v.median()] if d > 0 else res[v <= v.median()]
        side = "high" if d > 0 else "low"
        print(f"  keep {side:>4} {name:<22} n={len(keep):>4}  mean {keep['ret_5'].mean():>+6.2f}  "
              f"(delta {keep['ret_5'].mean()-base:>+5.2f})")
    print("\n  These use only info known at entry, so they COULD become filters. BUT each was chosen")
    print("  because it separated winners/losers in THIS sample, so the gain shown is optimistic by")
    print("  construction - the fresh-slice retest is mandatory, not optional.")
    print("  (Event flags and MAE are DIAGNOSTIC only: they describe what happened during the hold")
    print("   and cannot select trades in advance - never use them as filters.)")


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def run_self_test():
    import tempfile
    rng = np.random.default_rng(1)
    syms = [f"S{i}" for i in range(25)]
    days = pd.bdate_range("2026-01-01", periods=60)
    featnames = ["D_WQ_19", "D_dist_from_52wh", "D_range_pct", "regime_market_trend", "D_WQ_40"]
    rows = []
    for s in syms:
        base = rng.uniform(50, 500)
        drift = rng.normal(0.001, 0.001)
        px = base * np.cumprod(1 + rng.normal(drift, 0.02, len(days)))
        fvals = {f: rng.normal(0, 1) for f in featnames}
        for k, (d, c) in enumerate(zip(days, px)):
            hi = c * (1 + abs(rng.normal(0, 0.012)))
            lo = c * (1 - abs(rng.normal(0, 0.012)))
            op = lo + (hi - lo) * rng.random()
            row = dict(symbol=s, timestamp=d, open=op, high=hi, low=lo, close=c,
                       volume=rng.uniform(1e5, 1e6), stock_regime="bull_trend")
            for f in featnames:
                row[f] = fvals[f] + rng.normal(0, 0.1)
            rows.append(row)
    panel = pd.DataFrame(rows)
    tmp = Path(tempfile.mkdtemp())
    panel.to_parquet(tmp / "panel.parquet")
    (tmp / "feats.json").write_text(json.dumps({"features": featnames, "impute": {f: 0.0 for f in featnames}}))
    # watchlists on several early dates so windows resolve
    wls = []
    for di in (10, 12, 14, 16, 18, 20):
        chosen = rng.choice(syms, 8, replace=False)
        for rank, sym in enumerate(chosen, 1):
            wls.append(dict(symbol=sym, prob=round(rng.uniform(0.65, 0.85), 3), rank=rank, date=days[di]))
    jpath = str(tmp / "journal.parquet")

    class A: pass
    for di_wl in [wls[i:i+8] for i in range(0, len(wls), 8)]:
        a = A(); a.source = "watchlist"; a.panel = str(tmp / "panel.parquet")
        a.features = str(tmp / "feats.json"); a.journal = jpath
        wlp = tmp / "wl.csv"; pd.DataFrame(di_wl).to_csv(wlp, index=False); a.watchlist = str(wlp)
        a.min_prob = 0.0; a.top_n = None; a.scored = None
        mode_record(a)
    au = A(); au.journal = jpath; au.panel = str(tmp / "panel.parquet")
    mode_update(au)
    aa = A(); aa.journal = jpath
    mode_analyze(aa)
    j = load_journal(jpath)
    assert (j["status"] == "RESOLVED").sum() >= 30, "expected most rows resolved"
    assert any(c.startswith("feat__") for c in j.columns), "feature snapshot missing"
    print("\nSELF-TEST OK")
    return 0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Prediction journal + loser analysis.")
    ap.add_argument("--mode", choices=["record", "update", "analyze"])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--source", choices=["watchlist", "scored"], default="watchlist")
    ap.add_argument("--watchlist"); ap.add_argument("--scored")
    ap.add_argument("--panel"); ap.add_argument("--features")
    ap.add_argument("--journal", default="prediction_journal.parquet")
    ap.add_argument("--min-prob", dest="min_prob", type=float, default=0.50)
    ap.add_argument("--top-n", dest="top_n", type=int, default=None)
    a = ap.parse_args()

    if a.self_test:
        sys.exit(run_self_test())
    if a.mode == "record":
        if not (a.panel and a.features) or (a.source == "watchlist" and not a.watchlist) \
                or (a.source == "scored" and not a.scored):
            raise SystemExit("record needs --panel --features and the matching --watchlist/--scored")
        mode_record(a)
    elif a.mode == "update":
        if not a.panel:
            raise SystemExit("update needs --panel")
        mode_update(a)
    elif a.mode == "analyze":
        mode_analyze(a)
    else:
        raise SystemExit("pick --mode record|update|analyze (or --self-test)")


if __name__ == "__main__":
    main()
