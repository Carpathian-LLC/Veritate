# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tests canonical long-only cross-sectional equity momentum (12-1 signal, skip most
#   recent month, top decile equal-weight, monthly + 2-week rebalance, 3bp/side all-in
#   cost) on ~500 current S&P 500 members vs SPY total return, 2001-2026, Yahoo daily
#   adjclose. Variants: full universe, top-100-by-dollar-volume, 15%-vol-target overlay.
#   Null: 20 random-decile seeds, same costs. Rule fixed a priori from literature; no
#   parameter tuning anywhere. Falsifier: momentum does not beat SPY net post-2015
#   (>= 6 of 2016-2025 beat years AND full-period Sharpe >= SPY AND maxDD gap <= 10pp).
#   Universe is survivorship-biased (delisted members unavailable free): results are an
#   UPPER BOUND on the strategy; a negative verdict is therefore stronger.
# - wall-clock estimate ~10 min (one-time Yahoo download ~6 min, cached; compute <1 min).
# SMOKE_RESULTS/equity_momentum_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import ssl
import time
import traceback
import urllib.error
import urllib.request

import certifi
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# Constants

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_PATH = os.path.join(REPO, "SMOKE_RESULTS/equity_momentum_stats.json")
CACHE_DIR = (
    "/private/tmp/claude-501/-Users-mirach-00-usc1-Development-Veritate/"
    "10655374-b645-4bca-a6e1-0208baad89ab/scratchpad/equity_cache"
)

YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    "?period1={p1}&period2={p2}&interval=1d"
)
YAHOO_P1 = 946684800
UA = {"User-Agent": "Mozilla/5.0"}
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
FETCH_SLEEP_S = 0.35
FETCH_RETRIES = 3
FETCH_TIMEOUT_S = 30
BENCH = "SPY"

START_DATE = "2000-01-01"
LOOKBACK_D = 252
SKIP_D = 21
TOP_FRAC = 0.10
N_TOP_MIN = 25
N_TOP_MAX = 50
MIN_PRICE = 5.0
MIN_ELIGIBLE = 100
COST_PER_SIDE = 0.0003
REB_MONTHLY = "monthly"
REB_BIWEEKLY_D = 10
DV_WINDOW_D = 63
DV_TOP_N = 100
VOL_TARGET_ANN = 0.15
VOL_WINDOW_D = 63
VOL_MAX_EXPOSURE = 1.0
N_NULL_SEEDS = 20
TRADING_DAYS = 252
ROLL_3Y_D = 756
FOCUS_YEARS = list(range(2015, 2027))
BAR_YEARS = list(range(2016, 2026))
BAR_MIN_BEATS = 6
BAR_MAXDD_GAP = 0.10
CRASH_2020 = ("2020-02-19", "2020-03-23")

# current S&P 500 members (approximate, mid-2025 membership; Yahoo symbols).
# survivorship caveat: delisted ex-members (TWTR, ATVI, SIVB, PXD, MRO, HES, X, CTLT,
# FRC, VMW, SGEN, ...) are NOT retrievable from Yahoo (purged) or Stooq (JS-walled /
# bulk 401), so the panel backfills only survivors.
UNIVERSE = """
A AAPL ABBV ABNB ABT ACGL ACN ADBE ADI ADM ADP ADSK AEE AEP AES AFL AIG AIZ AJG AKAM
ALB ALGN ALL ALLE AMAT AMCR AMD AME AMGN AMP AMT AMZN ANET AON AOS APA APD APH APO
APP APTV ARE ATO AVB AVGO AVY AWK AXON AXP AZO BA BAC BALL BAX BBY BDX BEN BF-B BG
BIIB BK BKNG BKR BLDR BLK BMY BR BRK-B BRO BSX BX BXP C CAG CAH CARR CAT CB CBOE CBRE
CCI CCL CDNS CDW CEG CF CFG CHD CHRW CHTR CI CINF CL CLX CMCSA CME CMG CMI CMS CNC
CNP COF COIN COO COP COR COST CPAY CPB CPRT CPT CRL CRM CRWD CSCO CSGP CSX CTAS CTRA
CTSH CTVA CVS CVX CZR D DAL DASH DAY DD DE DECK DELL DFS DG DGX DHI DHR DIS DLR DLTR
DOC DOV DOW DPZ DRI DTE DUK DVA DVN DXCM EA EBAY ECL ED EFX EG EIX EL ELV EMN EMR
ENPH EOG EPAM EQIX EQR EQT ERIE ES ESS ETN ETR EVRG EW EXC EXE EXPD EXPE EXR F FANG
FAST FCX FDS FDX FE FFIV FI FICO FIS FITB FOX FOXA FRT FSLR FTNT FTV GD GDDY GE GEHC
GEN GEV GILD GIS GL GLW GM GNRC GOOG GOOGL GPC GPN GRMN GS GWW HAL HAS HBAN HCA HD
HIG HII HLT HOLX HON HOOD HPE HPQ HRL HSIC HST HSY HUBB HUM HWM IBM ICE IDXX IEX IFF
INCY INTC INTU INVH IP IPG IQV IR IRM ISRG IT ITW IVZ J JBHT JBL JCI JKHY JNJ JPM K
KDP KEY KEYS KHC KIM KKR KLAC KMB KMI KMX KO KR KVUE L LDOS LEN LH LHX LII LIN LKQ
LLY LMT LNT LOW LRCX LULU LUV LVS LW LYB LYV MA MAA MAR MAS MCD MCHP MCK MCO MDLZ MDT
MET META MGM MHK MKC MKTX MLM MMC MMM MNST MO MOH MOS MPC MPWR MRK MRNA MS MSCI MSFT
MSI MTB MTCH MTD MU NCLH NDAQ NDSN NEE NEM NFLX NI NKE NOC NOW NRG NSC NTAP NTRS NUE
NVDA NVR NWS NWSA NXPI O ODFL OKE OMC ON ORCL ORLY OTIS OXY PANW PARA PAYC PAYX PCAR
PCG PEG PEP PFE PFG PG PGR PH PHM PKG PLD PLTR PM PNC PNR PNW PODD POOL PPG PPL PRU
PSA PSX PTC PWR PYPL QCOM RCL REG REGN RF RJF RL RMD ROK ROL ROP ROST RSG RTX RVTY
SBAC SBUX SCHW SHW SJM SLB SMCI SNA SNPS SO SOLV SPG SPGI SRE STE STLD STT STX STZ SW
SWK SWKS SYF SYK SYY T TAP TDG TDY TECH TEL TER TFC TFX TGT TJX TKO TMO TMUS TPL TPR
TRGP TRMB TROW TRV TSCO TSLA TSN TT TTD TTWO TXN TXT TYL UAL UBER UDR UHS ULTA UNH
UNP UPS URI USB V VICI VLO VLTO VMC VRSK VRSN VRTX VST VTR VTRS VZ WAB WAT WBD WDC
WEC WELL WFC WM WMB WMT WRB WSM WST WTW WY WYNN XEL XOM XYL YUM ZBH ZBRA ZTS
""".split()

# ------------------------------------------------------------------------------------
# Functions


def fetch_yahoo(sym):
    url = YAHOO_URL.format(sym=sym, p1=YAHOO_P1, p2=int(time.time()))
    for attempt in range(FETCH_RETRIES):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S, context=SSL_CTX) as resp:
                payload = json.load(resp)
            res = payload["chart"]["result"]
            if not res:
                return None
            r0 = res[0]
            ts = r0.get("timestamp")
            if not ts:
                return None
            quote = r0["indicators"]["quote"][0]
            adj = r0["indicators"].get("adjclose", [{}])[0].get("adjclose")
            if adj is None:
                adj = quote["close"]
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
                    "adj": adj,
                    "close": quote["close"],
                    "volume": quote["volume"],
                }
            ).dropna(subset=["adj", "close"])
            df = df[df["adj"] > 0]
            return df.drop_duplicates(subset="date").set_index("date")
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError):
            if attempt == FETCH_RETRIES - 1:
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def load_panel(symbols, failed):
    frames = {}
    for sym in symbols:
        path = os.path.join(CACHE_DIR, f"{sym}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = fetch_yahoo(sym)
            time.sleep(FETCH_SLEEP_S)
            if df is not None:
                df.to_csv(path)
        if df is None or df.empty:
            failed.append(sym)
            continue
        frames[sym] = df[df.index >= START_DATE]
    grid = frames[BENCH].index
    adj = pd.DataFrame({s: f["adj"] for s, f in frames.items()}).reindex(grid)
    close = pd.DataFrame({s: f["close"] for s, f in frames.items()}).reindex(grid)
    dollar_vol = (
        pd.DataFrame({s: f["close"] * f["volume"] for s, f in frames.items()})
        .reindex(grid)
        .rolling(DV_WINDOW_D, min_periods=DV_WINDOW_D // 2)
        .mean()
    )
    return adj, close, dollar_vol


def rebalance_indices(dates, mode):
    if mode == REB_MONTHLY:
        month = dates.to_period("M").astype(str).to_numpy()
        return np.flatnonzero(month[:-1] != month[1:])
    return np.arange(LOOKBACK_D, len(dates), REB_BIWEEKLY_D)


def eligible_mask(A, C, t):
    return (
        np.isfinite(A[t])
        & np.isfinite(A[t - SKIP_D])
        & np.isfinite(A[t - LOOKBACK_D])
        & (C[t] >= MIN_PRICE)
    )


def pick_momentum(A, C, DV, t, top100, rng):
    elig = eligible_mask(A, C, t)
    if top100:
        dv = np.where(elig & np.isfinite(DV[t]), DV[t], -np.inf)
        keep = np.argsort(dv)[-DV_TOP_N:]
        m = np.zeros_like(elig)
        m[keep] = elig[keep] & np.isfinite(DV[t][keep])
        elig = m
    idx = np.flatnonzero(elig)
    if len(idx) < MIN_ELIGIBLE:
        return None
    n_top = int(np.clip(int(TOP_FRAC * len(idx)), N_TOP_MIN, N_TOP_MAX))
    if rng is not None:
        return rng.choice(idx, size=n_top, replace=False)
    sig = A[t - SKIP_D][idx] / A[t - LOOKBACK_D][idx] - 1.0
    return idx[np.argsort(sig)[-n_top:]]


def run_backtest(adj, close, dollar_vol, reb_idx, top100=False, seed=None):
    A, C, DV = adj.to_numpy(), close.to_numpy(), dollar_vol.to_numpy()
    R = np.nan_to_num(adj.pct_change().to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(seed) if seed is not None else None
    n_days, n_sym = A.shape
    w = np.zeros(n_sym)
    daily = np.full(n_days, np.nan)
    reb_set = set(int(t) for t in reb_idx)
    traded_total, n_reb, names_sum, start_t = 0.0, 0, 0, None
    for t in range(int(reb_idx[0]), n_days):
        gross = float(w @ R[t])
        if w.sum() > 0:
            w = w * (1.0 + R[t]) / (1.0 + gross)
        cost = 0.0
        if t in reb_set:
            picks = pick_momentum(A, C, DV, t, top100, rng)
            if picks is not None:
                w_new = np.zeros(n_sym)
                w_new[picks] = 1.0 / len(picks)
                traded = float(np.abs(w_new - w).sum())
                cost = traded * COST_PER_SIDE
                if start_t is not None:
                    traded_total += traded
                    n_reb += 1
                names_sum += len(picks)
                w = w_new
                if start_t is None:
                    start_t = t
        if start_t is not None and t > start_t:
            daily[t] = gross - cost
    ret = pd.Series(daily, index=adj.index).dropna()
    years_span = len(ret) / TRADING_DAYS
    turnover_ann = traded_total / 2.0 / max(years_span, 1e-9)
    avg_names = names_sum / max(n_reb + 1, 1)
    return ret, turnover_ann, avg_names


def vol_overlay(ret):
    vol = ret.rolling(VOL_WINDOW_D).std() * np.sqrt(TRADING_DAYS)
    exp = (VOL_TARGET_ANN / vol).clip(upper=VOL_MAX_EXPOSURE).shift(1)
    exp = exp.fillna(VOL_MAX_EXPOSURE)
    cost = exp.diff().abs().fillna(0.0) * COST_PER_SIDE
    return ret * exp - cost


def metrics(ret):
    if len(ret) < 2:
        return {}
    eq = (1.0 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)
    sharpe = float(ret.mean() / ret.std() * np.sqrt(TRADING_DAYS))
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": round(cagr, 4), "sharpe": round(sharpe, 3), "maxdd": round(maxdd, 4)}


def per_year(ret):
    out = {}
    for y, grp in ret.groupby(ret.index.year):
        out[int(y)] = round(float((1.0 + grp).prod() - 1.0), 4)
    return out


def window_return(ret, a, b):
    seg = ret[(ret.index >= a) & (ret.index <= b)]
    return round(float((1.0 + seg).prod() - 1.0), 4) if len(seg) else None


def rolling_3y_excess(strat, spy):
    both = pd.concat([strat, spy], axis=1, keys=["s", "b"]).dropna()
    ann = lambda x: (1.0 + x).rolling(ROLL_3Y_D).apply(np.prod, raw=True) ** (
        TRADING_DAYS / ROLL_3Y_D
    ) - 1.0
    ex = (ann(both["s"]) - ann(both["b"])).dropna()
    ex_m = ex.groupby(ex.index.to_period("M")).last()
    series = [[str(p), round(float(v), 4)] for p, v in ex_m.items()]
    return {
        "min": round(float(ex.min()), 4),
        "median": round(float(ex.median()), 4),
        "last": round(float(ex.iloc[-1]), 4),
        "frac_positive": round(float((ex > 0).mean()), 3),
        "monthly_series": series,
    }


def main():
    t0 = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)
    stats = {
        "smoke": "equity_momentum",
        "date": time.strftime("%Y-%m-%d"),
        "errors": [],
        "falsifier": (
            "canonical 12-1 top-decile long-only momentum does not beat SPY net post-2015: "
            f"needs >= {BAR_MIN_BEATS} beat-years in {BAR_YEARS[0]}-{BAR_YEARS[-1]}, full-period "
            "Sharpe >= SPY, maxDD gap <= 10pp"
        ),
    }
    failed = []
    adj, close, dollar_vol = load_panel(UNIVERSE + [BENCH], failed)
    spy = adj[BENCH].pct_change().dropna()
    adj = adj.drop(columns=[BENCH])
    close = close.drop(columns=[BENCH])
    dollar_vol = dollar_vol.drop(columns=[BENCH])
    stats["data"] = {
        "source": "yahoo v8 chart adjclose (split+dividend adjusted, total-return proxy)",
        "n_symbols_ok": int(adj.shape[1]),
        "n_failed": len(failed),
        "failed_symbols": failed,
        "grid": [str(adj.index[0].date()), str(adj.index[-1].date())],
        "survivorship": (
            "current-constituent panel; delisted ex-members unavailable (Yahoo purges, Stooq "
            "JS-walled, bulk 401; probed TWTR/ATVI/SIVB/PXD/MRO/HES/X/CTLT all dead). Bias "
            "direction: inflates strategy returns; results are an upper bound."
        ),
    }
    stats["protocol"] = {
        "signal": "12-1 momentum: adj[t-21]/adj[t-252]-1, no tuning",
        "portfolio": f"top decile equal-weight, clip [{N_TOP_MIN},{N_TOP_MAX}] names, long-only",
        "cost_per_side": COST_PER_SIDE,
        "min_price": MIN_PRICE,
        "vol_overlay": f"target {VOL_TARGET_ANN}, {VOL_WINDOW_D}d realized, cap {VOL_MAX_EXPOSURE}, cash at 0%",
        "null": f"{N_NULL_SEEDS} random-decile seeds, same dates/sizes/costs",
    }

    reb_m = rebalance_indices(adj.index, REB_MONTHLY)
    reb_m = reb_m[reb_m >= LOOKBACK_D]
    reb_b = rebalance_indices(adj.index, "biweekly")

    runs = {}
    ret_m, to_m, nm_m = run_backtest(adj, close, dollar_vol, reb_m)
    runs["mom_monthly"] = (ret_m, to_m, nm_m)
    ret_b, to_b, nm_b = run_backtest(adj, close, dollar_vol, reb_b)
    runs["mom_biweekly"] = (ret_b, to_b, nm_b)
    ret_t, to_t, nm_t = run_backtest(adj, close, dollar_vol, reb_m, top100=True)
    runs["mom_monthly_top100dv"] = (ret_t, to_t, nm_t)
    ret_v = vol_overlay(ret_m)
    runs["mom_monthly_volscaled"] = (ret_v, to_m, nm_m)

    spy_al = spy.reindex(ret_m.index).dropna()
    stats["results"] = {}
    for name, (ret, to, nm) in runs.items():
        stats["results"][name] = {
            **metrics(ret),
            "turnover_oneway_ann": round(to, 2),
            "avg_names": round(nm, 1),
            "per_year": {y: v for y, v in per_year(ret).items() if y in FOCUS_YEARS},
        }
    stats["results"]["spy"] = {
        **metrics(spy_al),
        "per_year": {y: v for y, v in per_year(spy_al).items() if y in FOCUS_YEARS},
    }
    stats["full_period_start"] = str(ret_m.index[0].date())

    spy_yr = per_year(spy_al)
    mom_yr = per_year(ret_m)
    stats["per_year_net_vs_spy"] = {
        str(y): {
            "mom": mom_yr.get(y),
            "spy": spy_yr.get(y),
            "excess": round(mom_yr[y] - spy_yr[y], 4) if y in mom_yr and y in spy_yr else None,
        }
        for y in FOCUS_YEARS
        if y in mom_yr
    }

    stats["rolling_3y_excess"] = rolling_3y_excess(ret_m, spy_al)
    stats["crash_windows"] = {
        "2020_crash": {
            "mom": window_return(ret_m, *CRASH_2020),
            "mom_volscaled": window_return(ret_v, *CRASH_2020),
            "spy": window_return(spy_al, *CRASH_2020),
        },
        "2020_full_year": {
            "mom": mom_yr.get(2020),
            "mom_volscaled": per_year(ret_v).get(2020),
            "spy": spy_yr.get(2020),
        },
        "2022_bear": {
            "mom": mom_yr.get(2022),
            "mom_volscaled": per_year(ret_v).get(2022),
            "spy": spy_yr.get(2022),
        },
        "2009_reversal_echo": {
            "mom": mom_yr.get(2009),
            "spy": spy_yr.get(2009),
        },
    }

    null_cagr, null_sharpe = [], []
    for seed in range(N_NULL_SEEDS):
        ret_n, _, _ = run_backtest(adj, close, dollar_vol, reb_m, seed=seed)
        m = metrics(ret_n)
        null_cagr.append(m["cagr"])
        null_sharpe.append(m["sharpe"])
    mom_full = metrics(ret_m)
    stats["null_random_decile"] = {
        "cagr_mean": round(float(np.mean(null_cagr)), 4),
        "cagr_min_max": [min(null_cagr), max(null_cagr)],
        "sharpe_mean": round(float(np.mean(null_sharpe)), 3),
        "mom_cagr_pctile_of_null": round(
            float(np.mean([mom_full["cagr"] > c for c in null_cagr])), 3
        ),
    }

    beats = sum(
        1 for y in BAR_YEARS if y in mom_yr and y in spy_yr and mom_yr[y] > spy_yr[y]
    )
    spy_full = metrics(spy_al)
    bar = {
        "beat_years": f"{beats}/{len(BAR_YEARS)} ({BAR_YEARS[0]}-{BAR_YEARS[-1]}), need >= {BAR_MIN_BEATS}",
        "sharpe_mom_vs_spy": [mom_full["sharpe"], spy_full["sharpe"]],
        "maxdd_gap_pp": round((spy_full["maxdd"] - mom_full["maxdd"]) * 100, 1),
        "passed": bool(
            beats >= BAR_MIN_BEATS
            and mom_full["sharpe"] >= spy_full["sharpe"]
            and (spy_full["maxdd"] - mom_full["maxdd"]) <= BAR_MAXDD_GAP
        ),
    }
    for name in ("mom_monthly_volscaled", "mom_biweekly", "mom_monthly_top100dv"):
        ret, _, _ = runs[name]
        yr = per_year(ret)
        b = sum(1 for y in BAR_YEARS if y in yr and y in spy_yr and yr[y] > spy_yr[y])
        m = metrics(ret)
        bar[f"{name}_beats"] = f"{b}/{len(BAR_YEARS)}"
        bar[f"{name}_sharpe"] = m["sharpe"]
    stats["survivor_bar"] = bar
    stats["wall_clock_s"] = round(time.time() - t0, 1)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "rolling_3y_excess"}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        err = traceback.format_exc()
        with open(STATS_PATH, "w") as f:
            json.dump({"smoke": "equity_momentum", "errors": [err]}, f, indent=1)
        print(err)
        raise SystemExit(1)
