#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
worstof_single_asset.py
Single‑asset IV‑surface ➜ local‑vol ➜ Monte‑Carlo path simulation
"""
from __future__ import annotations
import os, math, functools, json, pickle, asyncio, aiohttp, time
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import inspect, textwrap
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes import black_scholes
from polygon import RESTClient, exceptions as poly_exc

from scipy.optimize import least_squares
import sqlite3
import bisect
from scipy.interpolate import CubicSpline
from urllib3.exceptions import MaxRetryError, ResponseError
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from typing import List, Callable, Tuple,Dict
import aiohttp, logging
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from functools import reduce
import pathlib

# ------------------------------------------------------------
# 0 . 全局配置
# ------------------------------------------------------------

API_KEY      = os.getenv("POLYGON_API_KEY", "ty5dEd7vswig5euSoMM728XFPM8FcYm6")
UNDERLYINGS   = ["SPY","QQQ","IWM","VOO", "IVV", "VTI", "DIA"]
VAL_DATE = "2025-05-29"           # YYYY‑MM‑DD

N_WORKERS    = 16                     # 并发抓价
N_PATHS      = 100_000
DT_STEP = 1 / 252              # 日度
T_MIN        = 7/365                 # 局部波网格到期上界（年）
GRID_NT, GRID_NK = 60, 140
BETA_FIXED  = 0.5
PRICE_MIN  = 0.05
# ========== 1. Tikhonov + 可调 bound ========== #
LAMBDA_TIK = 5e-3          # 🔹正则强度
SPL_SMOOTH = 1e-2          # 🔹样条平滑 s 参数
BOUNDS_MIN = np.array([1e-3, -0.999, 1e-3])
BOUNDS_MAX = np.array([5.0,   0.999, 5.0])  # <–– 可手动再放宽
MC_PATHS   = 250_000                        # 蒙特卡洛路径
RNG_SEED   = 42

# ------------------------------------------------------------
# 1 . USD 无风险利率曲线（FRED）——沿用你原 util，但封装缓存
# ------------------------------------------------------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
TREASURIES = {1:"DGS1",2:"DGS2",3:"DGS3",5:"DGS5",7:"DGS7",10:"DGS10"}

@functools.lru_cache(maxsize=None)


def _load_fred_series(series_id: str) -> pd.Series:
    csv = FRED_CSV.format(sid=series_id)
    df  = pd.read_csv(csv)
    date_col = next(c for c in ("DATE","date","observation_date","Date") if c in df.columns)
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    val_col = series_id if series_id in df.columns else "value"
    ser = pd.to_numeric(df[val_col], errors="coerce")/100     # %→dec
    ser.index = df[date_col]; ser = ser.sort_index()
    return ser            # 与原实现一致

#todo 搞清楚zero curve的build逻辑细节
def _zero_curve(val_date):
    if isinstance(val_date, str):
        val_date = val_date.translate({c: "-" for c in range(0x2010, 0x2016)})
    val_date = pd.to_datetime(val_date).tz_localize(None)
    tenors, rates = [0.0], [_overnight_rate(val_date)]
    for y,sid in TREASURIES.items():
        rates.append(math.log(1+_last_available(_load_fred_series(sid),val_date)))
        tenors.append(float(y))
    return np.vectorize(PchipInterpolator(tenors, rates, extrapolate=True))

def _overnight_rate(ts) -> float:
    sid = "SOFR" if ts >= pd.Timestamp("2018-04-03") else "FEDFUNDS"
    r   = _last_available(_load_fred_series(sid), ts)
    return math.log(1+r)

def _last_available(ser: pd.Series, ts: pd.Timestamp) -> float:
    if ts in ser.index: return ser[ts]
    return ser.loc[:ts].iloc[-1]

def _normalize_date(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")

# ------------------------------------------------------------
# 2 . Polygon API  —— 期权链 + 收盘价（批量并发 + 本地缓存）
# ------------------------------------------------------------
client = RESTClient(API_KEY)
CACHE_DIR    = Path("cache"); CACHE_DIR.mkdir(exist_ok=True)
_PRICE_CACHE = CACHE_DIR/"daily_close.pkl"
if _PRICE_CACHE.exists(): PRICE_MEM = pickle.loads(_PRICE_CACHE.read_bytes())
else: PRICE_MEM = {}

def _save_price_cache(): _PRICE_CACHE.write_bytes(pickle.dumps(PRICE_MEM))

def _cache_get(path: pathlib.Path, fresh_seconds: int) -> bool:
    return path.exists() and time.time() - path.stat().st_mtime < fresh_seconds

async def _fetch_price_async(session, symbol):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{VAL_DATE}/{VAL_DATE}?apiKey={API_KEY}&adjusted=true"
    async with session.get(url) as resp:
        js = await resp.json()
        if js.get("results"):
            return symbol, js["results"][0]["c"]
        return symbol, None

def batch_close(symbols):
    """异步批量拉取收盘价；自动读取 / 更新本地缓存"""
    cache_f = CACHE_DIR / "close_cache.pkl"
    mem = pickle.loads(cache_f.read_bytes()) if cache_f.exists() else {}
    miss = [s for s in symbols if s not in mem]
    if miss:
        async def _main():
            async with aiohttp.ClientSession() as sess:
                coros = [_fetch_price_async(sess, s) for s in miss]
                for sym, px in await asyncio.gather(*coros):
                    if px: mem[sym] = px
        asyncio.get_event_loop().run_until_complete(_main())
        cache_f.write_bytes(pickle.dumps(mem))
    return {s: mem[s] for s in symbols if s in mem}

def get_daily_close_batch(symbols: list[str]) -> dict[str,float]:
    miss = [s for s in symbols if s not in PRICE_MEM]
    if miss:
        loop = asyncio.get_event_loop()
        async def _main():
            async with aiohttp.ClientSession() as sess:
                coros = [_fetch_price_async(sess,s) for s in miss]
                for sym,px in await asyncio.gather(*coros):
                    if px: PRICE_MEM[sym]=px
        loop.run_until_complete(_main())
        _save_price_cache()
    return {s:PRICE_MEM.get(s) for s in symbols}

def get_spot_polygon(symbol: str,
                     val_date: str | pd.Timestamp,
                     fresh_sec: int = 3600) -> float:
    """
    先用 aggregates 抓估值日收盘；
    若当天无交易 → 回退上一交易日收盘。
    """
    val_date = pd.to_datetime(val_date).date()
    cache_f  = CACHE_DIR / f"{symbol}_{val_date}_spot.npy"
    if _cache_get(cache_f, fresh_sec):
        return float(np.load(cache_f))

    # ---- 1. 当天收盘 ---------------------------------------------------
    try:
        bars = list(client.list_aggs(
            symbol, 1, "day",
            from_=val_date.isoformat(), to=val_date.isoformat(), limit=1
        ))
        if bars:
            px = float(bars[0].close)
            np.save(cache_f, np.array(px, dtype=float))
            return px
    except Exception as e:
        logging.warning(f"{symbol}: agg fetch failed – {e}")

    # ---- 2. 回退前一交易日 --------------------------------------------
    px = get_daily_close_batch(symbol)
    np.save(cache_f, np.array(px, dtype=float))
    return px


# Batch price loader with safe event‑loop handling & local cache
    # ---------------------------------------------------------------------------
def fetch_hist_polygon(symbol: str,
                       days_back: int = 180,
                       fresh: int = 86_400) -> pd.Series | None:
    f = CACHE_DIR / f"{symbol}_hist.parquet"
    if _cache_get(f, fresh):
        return pd.read_parquet(f, columns=["close"])["close"]

    try:
        end   = pd.Timestamp.utcnow().normalize()
        start = end - pd.Timedelta(days_back, "D")
        aggs = list(client.list_aggs(
            symbol, 1, "day",
            from_=start.date().isoformat(),
            to=end.date().isoformat(),
            limit=5000
        ))
        ser = pd.Series({pd.to_datetime(a.timestamp, unit="ms"): a.close
                         for a in aggs},
                        name="close").sort_index()
        if len(ser) > 3:
            ser.to_frame().to_parquet(f)
            return ser
    except Exception as e:
        logging.warning(f"{symbol}: hist fetch failed – {e}")
    return None

def get_active_option_chain(underlying: str,
                            val_date: str | pd.Timestamp) -> pd.DataFrame:
    """
    - 拉取期权链
    - 补充 price（先 quote/trade，再 list_trades 兜底）
    - 写 parquet 缓存
    返回列：ticker, K, cp, exp, T, price
    """
    cache_f = CACHE_DIR / f"{underlying}_{val_date}_chain.parquet"
    if cache_f.exists():
        df = pd.read_parquet(cache_f)
        if "price" in df.columns and df["price"].notna().any():
            return df
        cache_f.unlink()                       # 缓存无 price → 重抓

    # 1) 合约列表 --------------------------------------------------------
    contracts = list(client.list_options_contracts(
        underlying_ticker=underlying,
        as_of=_normalize_date(val_date),
        expired=False, sort="expiration_date",
        order="asc", limit=1000))
    if not contracts:
        return pd.DataFrame()

    df = pd.DataFrame({
        "ticker": [c.ticker for c in contracts],
        "K"     : [c.strike_price for c in contracts],
        "exp"   : pd.to_datetime([c.expiration_date for c in contracts]),
        "cp"    : [1 if "C" in c.ticker else 0 for c in contracts],
    })
    val_dt = pd.to_datetime(val_date)
    df["T"] = (df["exp"] - val_dt).dt.days / 365

    # 2) 逐合约补价 ------------------------------------------------------
    def _mid_price(tkr: str) -> float | None:
        # (a) Quote
        try:
            # Get comprehensive option pricing data
            snapshot = client.get_option_contract_snapshot(underlying, tkr)

            # Access pricing information
            print(f"Break-even price: {snapshot.break_even_price}")
            print(f"Day change: {snapshot.day.change}")
            print(f"Day change %: {snapshot.day.change_percent}")
            print(f"Close price: {snapshot.day.close}")
            print(f"High: {snapshot.day.high}")

        except Exception as e:
            print(f"Error: {e}")

    prices = []
    for i, tkr in enumerate(df["ticker"]):
        if i % 50 == 0 and i:
            time.sleep(0.2)                    # 简易限速
        prices.append(_mid_price(tkr))
    df["price"] = prices                       # ★ 先写列

    # 3) 现在再打印调试信息 ---------------------------------------------
    logging.info(f"{underlying}: rows={len(df)}, "
                 f"non‑NaN price={(df['price'].notna()).sum()}")

    df.to_parquet(cache_f)
    return df

# df_chain = get_active_option_chain("SPY", VAL_DATE)
# print("price 非 NaN 行数：", df_chain["price"].notna().sum())

t = client.get_daily_open_close_agg("O:SPY250117C00600000 ","2025-01-16")
print(t.close)


def bs_vega(F, K, T, sigma):
    d1 = (np.log(F/K) + 0.5*sigma*sigma*T) / (sigma*np.sqrt(T))
    return np.sqrt(T) * norm.pdf(d1)
# ------------------------------------------------------------
# 3 . IV 表 + SVI 拟合
# ------------------------------------------------------------
def _vectorized_iv(price, S, K, r_cc, T, flag):
    """NumPy 向量化封装 py_vollib"""
    price = np.asarray(price)
    N = price.size
    S = np.broadcast_to(S, (N,))
    K, r_cc, T, flag = map(np.asarray, (K, r_cc, T, flag))
    # ---- 改用掩码，避免 0 除 ----
    r_s = np.zeros_like(T, dtype=float)
    mask = T > 0
    r_s[mask] = (np.exp(r_cc[mask] * T[mask]) - 1) / T[mask]
    out = np.full(N, np.nan, dtype=float)
    for i, (p, s_, k_, r_, t_, flg) in enumerate(zip(price, S, K, r_s, T, flag)):
        try:
            out[i] = implied_volatility(p, s_, k_, r_, t_, flg)
        except Exception:
            pass
    return out

# ------------------------------------------------------------------------
# build_iv_table  v2.2
# ------------------------------------------------------------------------
# —— 依赖的辅助函数：若已在项目中定义，可删除下面 stub ————————
# get_active_option_chain : 抓指定估值日“最近到期”的全部合约，返回 DataFrame
# get_daily_close_batch   : 批量抓收盘价，返回 {symbol: price or Series}
# _vectorized_iv          : Black‑76 (或 BS) implied‑vol 求逆函数
# _zero_curve             : 返回 r(T) callable
# ------------------------------------------------------------------------
def build_iv_table(underlying: str,
                   val_date: str | pd.Timestamp,
                   *,
                   mny_width: float = 0.3,
                   min_price: float = 0.05,
                   min_iv: float = 0.05,
                   max_iv: float = 1.5,
                   min_T: float = 7/365) -> pd.DataFrame:
    """
    返回列: ['T','K','iv','spot','k','price','r_cc']
    • spot   : 由 get_spot_polygon 抓取
    • price  : 期权合约 mid/last（已在 get_active_option_chain 插入）
    """

    def _debug_counts(tag, df):
        logging.info(f"[{tag}] rows = {len(df)}, "
                     f"price>0 = {(df['price'] > 0).sum()}, "
                     f"iv.na = {df['iv'].isna().sum() if 'iv' in df else '‑'}")

    # --- 日期 & 期权链 ---------------------------------------------------
    if isinstance(val_date, str):
        val_date = val_date.translate({c: "-" for c in range(0x2010, 0x2016)})
    val_date = pd.to_datetime(val_date).tz_localize(None)

    chain = get_active_option_chain(underlying, val_date)
    _debug_counts("00 raw", chain)
    if chain.empty or "price" not in chain.columns:
        raise ValueError("empty chain or missing price")

    # --- spot -----------------------------------------------------------
    spot = get_spot_polygon(underlying, val_date)
    if not np.isfinite(spot):
        raise ValueError("spot NaN")
    chain["spot"] = spot

    # --- 基本过滤 -------------------------------------------------------
    chain = chain.dropna(subset=["price"])
    chain = chain.loc[chain["price"] > min_price]
    chain = chain.loc[abs(chain["K"]/spot - 1) <= mny_width]
    chain = chain.loc[chain["T"] >= min_T]
    if chain.empty:
        raise ValueError("empty after filters")

    # --- implied vol -----------------------------------------------------
    r_arr   = _zero_curve(val_date)(chain["T"].to_numpy())
    flags   = np.where(chain["cp"] == 1, 'c', 'p')
    spotvec = np.repeat(spot, len(chain))
    chain["iv"] = _vectorized_iv(chain["price"].to_numpy(),
                                 spotvec, chain["K"].to_numpy(),
                                 r_arr, chain["T"].to_numpy(),
                                 flags)
    chain = chain.loc[chain["iv"].between(min_iv, max_iv)]
    _debug_counts("02 iv calc", chain)
    if chain.empty:
        raise ValueError("no valid IV")

    # --- C / P 均值 ------------------------------------------------------
    piv = (chain.pivot_table(index=["T","K"], columns="cp", values="iv")
                 .dropna(subset=[0,1]))
    piv["price"] = chain.groupby(["T","K"])["price"].first()
    piv = piv[piv["price"] > min_price]
    piv["iv"] = piv[[0,1]].mean(axis=1)
    piv = piv.drop(columns=[0,1])

    # --- log‑moneyness ---------------------------------------------------
    r_cc = _zero_curve(val_date)(piv.index.get_level_values("T").to_numpy())
    piv["r_cc"] = r_cc
    piv["k"] = (np.log(piv.index.get_level_values("K")/spot)
                - r_cc * piv.index.get_level_values("T"))
    piv = piv.reset_index()
    piv["spot"] = spot
    return piv[["T","K","iv","spot","k","price","r_cc"]]

###############################################################################
# ▄▀▀█ ▄▀▀▄ ▄▀▀▄ █▀▀▄ █▀▀   2)   Black‑76 / SABR vectorised utilities
###############################################################################

def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Φ(x) – Cumulative distribution function of standard Normal (vectorised)."""
    return 0.5 * (1.0 + np.erf(np.asarray(x) / math.sqrt(2)))


def black76_call_vec(fwd: np.ndarray, k: np.ndarray, t: np.ndarray, vol: np.ndarray):
    """Vectorised Black‑76 undiscounted call price."""
    fwd = np.asarray(fwd)
    k = np.asarray(k)
    t = np.asarray(t)
    vol = np.asarray(vol)
    sigma_sqrt_t = vol * np.sqrt(t)
    # 避免除零
    near_zero = sigma_sqrt_t < 1e-12
    d1 = np.where(
        near_zero,
        np.inf,
        (np.log(fwd / k) + 0.5 * vol ** 2 * t) / sigma_sqrt_t,
    )
    d2 = d1 - sigma_sqrt_t
    price = fwd * _norm_cdf(d1) - k * _norm_cdf(d2)
    price = np.where(near_zero, np.maximum(fwd - k, 0.0), price)
    return price


def hagan_bs_vol_vec(f, k, t, alpha, beta, rho, nu):
    """
    Vectorised Hagan 2002 SABR implied vol (log‑normal version).
    Works for any mix of scalar / array inputs (broadcast‑safe).
    """
    # --- broadcast all inputs to same shape --------------------------------
    f, k, t, alpha, rho, nu = np.broadcast_arrays(
        np.asarray(f, dtype=float),
        np.asarray(k, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(alpha, dtype=float),
        np.asarray(rho, dtype=float),
        np.asarray(nu, dtype=float),
    )

    idx_eq = np.isclose(f, k)
    vol    = np.empty_like(f)

    # f == k (ATM closed‑form)
    if np.any(idx_eq):
        fk_beta = f[idx_eq] ** (1 - beta)
        term1 = (
            ((1 - beta) ** 2 / 24) * (alpha[idx_eq] ** 2) / (fk_beta ** 2)
            + (rho[idx_eq] * beta * nu[idx_eq] * alpha[idx_eq]) / (4 * fk_beta)
            + ((2 - 3 * rho[idx_eq] ** 2) * nu[idx_eq] ** 2 / 24)
        ) * t[idx_eq]
        vol[idx_eq] = alpha[idx_eq] / fk_beta * (1 + term1)

    # f ≠ k
    if np.any(~idx_eq):
        feq, keq, teq = f[~idx_eq], k[~idx_eq], t[~idx_eq]
        aeq, req, neq = alpha[~idx_eq], rho[~idx_eq], nu[~idx_eq]
        log_fk = np.log(feq / keq)
        z   = (neq / aeq) * (feq * keq) ** ((1 - beta) / 2) * log_fk
        x_z = np.log((np.sqrt(1 - 2 * req * z + z ** 2) + z - req) / (1 - req))
        fk_beta = (feq * keq) ** ((1 - beta) / 2)
        A = aeq / (
            fk_beta
            * (1 + ((1 - beta) ** 2 / 24) * log_fk ** 2
                 + ((1 - beta) ** 4 / 1920) * log_fk ** 4)
        )
        B = 1 + (
            ((1 - beta) ** 2 / 24) * (aeq ** 2) / (fk_beta ** 2)
            + (req * beta * neq * aeq) / (4 * fk_beta)
            + ((2 - 3 * req ** 2) * neq ** 2 / 24)
        ) * teq
        vol[~idx_eq] = A * z / x_z * B

    return vol


###############################################################################
# ▄▀▀▄ █▀▀ ▄▀▀▄ █  █   3)   单资产 SABR surface calibration
###############################################################################
def fit_sabr_slice(F, Ks, T, mkt_iv, beta, p_prior):                 # 正则 λ
    vegas = bs_vega(F, Ks, T, mkt_iv)  # 已向量化
    w = np.power(vegas / vegas.max(), 0.5)

    if p_prior is None:
        p_prior = np.array([0.25, 0.0, 0.5])   # α, ρ, ν 先验
    def resid(p):
        a, r, n = p
        model_iv = hagan_bs_vol_vec(F, Ks, T, a, beta, r, n)
        tik = np.sqrt(LAMBDA_TIK) * (p - p_prior)
        return np.hstack(((model_iv - mkt_iv) * w, tik))

    lo, hi = BOUNDS_MIN.copy(), BOUNDS_MAX.copy()
    # 🔹对于 T<1M 可自定义放宽
    if T < 0.1:
        hi[2] = 10.  # nu 上限

    sol = least_squares(resid, p_prior, bounds=(lo, hi))
    return sol.x  # (a, r, n)

### ---------- replace build_iv_table() 之后 -----------------
def build_sabr_surface(iv_df: pd.DataFrame,
                       spot: float,
                       r_curve: Callable[[float], float],
                       beta: float = 0.5
                       ) -> tuple[
                           float,            # beta
                           list[float],      # slices (T 节点)
                           Callable,         # a_spl
                           Callable,         # r_spl
                           Callable          # n_spl
                       ]:
    """
    逐到期(T)校准单资产 SABR，并对 α/ρ/ν 做 C² 样条插值。
    返回值与你旧脚本一致：beta, slices, a_spl, r_spl, n_spl
    """
    def _calib_slice(k_arr, iv_arr, fwd, t_exp):
        """
        在单个到期上最小化 Hagan 隐含波 RMSE，给出 α,ρ,ν
        """
        if len(k_arr) == 0:
            raise ValueError("no strikes in slice")
        mask = (k_arr / fwd > 0.7) & (k_arr / fwd < 1.3)
        k_arr, iv_arr = k_arr[mask], iv_arr[mask]

        w = np.exp(-((k_arr / fwd - 1) / 0.15) ** 2)  # 半衰宽 15%

        def obj(x):
            a, rho, nu = x
            model_iv = hagan_bs_vol_vec(fwd, k_arr, t_exp, a, beta, rho, nu)
            rmse = np.sqrt(np.average((model_iv - iv_arr) ** 2, weights=w))
            return rmse

        bounds = [
            (1e-4, 1.5),  # α   上限 100%
            (-0.999, 0.999),  # ρ
            (1e-4, 1.5)  # ν   上限 200%
        ]
        x0 = np.array([0.2, 0.0, 0.5])  # 初始猜测
        res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")
        return res.x

        # ── slice‑by‑slice 校准 ──────────────────────────────────────────
    T_nodes, a_nodes, r_nodes, n_nodes = [], [], [], []
    for T in sorted(iv_df["T"].unique()):
        sub = iv_df.loc[iv_df["T"] == T]
        k_arr, iv_arr = sub["K"].values, sub["iv"].values
        # 假设表里没有 Fwd 列就自行计算
        fwd = sub["Fwd"].iloc[0] if "Fwd" in sub.columns else spot * math.exp(r_curve(T) * T)
        α, ρ, ν = _calib_slice(k_arr, iv_arr, fwd, T)
        T_nodes.append(T)
        a_nodes.append(α)
        r_nodes.append(ρ)
        n_nodes.append(ν)

    # ── 构造三条样条 ─────────────────────────────────────────────────
    a_spl = CubicSpline(T_nodes, a_nodes, bc_type="natural", extrapolate=True)
    r_spl = CubicSpline(T_nodes, r_nodes, bc_type="natural", extrapolate=True)
    n_spl = CubicSpline(T_nodes, n_nodes, bc_type="natural", extrapolate=True)

    return beta, T_nodes, a_spl, r_spl, n_spl


def plot_sabr_surface(beta, slices, a_spl, r_spl, n_spl, spot, r_curve):
    """
    绘 3D 隐含波动率面；示例中仅用样条，不再索引 ‘slice’ dict
    """
    # 网格
    T_grid = np.linspace(min(slices), max(slices), 30)
    K_grid = np.linspace(0.7*spot, 1.3*spot, 40)
    KK, TT = np.meshgrid(K_grid, T_grid)

    # 计算隐含波
    iv_surf = np.empty_like(KK)
    for i in range(TT.shape[0]):
        T  = TT[i,0]
        F  = spot * np.exp(r_curve(T)*T)
        a  = a_spl(T);  rho = r_spl(T);  nu = n_spl(T)
        iv_surf[i,:] = hagan_bs_vol_vec(F, K_grid, T, a, beta, rho, nu)

    # 绘图
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(KK, TT, iv_surf, cmap='viridis')
    ax.set_xlabel('Strike K'); ax.set_ylabel('T (yr)'); ax.set_zlabel('IV')
    plt.show()

###############################################################################
# ▄▀▀▄ █▀▀ ▀█▀ ▄▀▄ █▀▀ █▄ ▄█   4)   Multi‑asset SABR path simulation
###############################################################################
def simulate_sabr_multi(
    spot: np.ndarray,
    beta: np.ndarray,
    a_spl_list, r_spl_list, n_spl_list,
    r_curve, T, corr_S,
    n_paths=200_000, dt=1 / 252, seed=0,
):
    rng   = np.random.default_rng(seed)
    spot  = np.asarray(spot, dtype=float)
    beta  = np.asarray(beta, dtype=float)
    N     = spot.size
    n_steps  = max(1, int(T / dt))
    dt_exact = T / n_steps
    # ── 正确的 σ(0)：先做 1×N，再 tile 成 (paths,N) ────────────────
    sigma0 = np.array([a(1e-4) / spot[i] ** (1 - beta[i])
                       for i, a in enumerate(a_spl_list)], dtype=float)
    sigma  = np.tile(sigma0, (n_paths, 1))                 # (paths, N)
    F = np.tile(spot, (n_paths, 1))                        # 初始远期
    L = np.linalg.cholesky(corr_S)
    for step in range(n_steps):
        t_now = (step + 1) * dt_exact
        z_raw = rng.standard_normal((n_paths, N))
        dW_S  = (z_raw @ L.T) * np.sqrt(dt_exact)
        z_nu  = rng.standard_normal((n_paths, N)) * np.sqrt(dt_exact)

        for i in range(N):
            alpha = a_spl_list[i](t_now)
            rho   = np.clip(r_spl_list[i](t_now), -0.999, 0.999)   # 防超界
            nu    = n_spl_list[i](t_now)

            dW_nu = rho * dW_S[:, i] + np.sqrt(1 - rho**2) * z_nu[:, i]
            sigma[:, i] += nu * sigma[:, i] * dW_nu
            sigma[:, i]  = np.maximum(sigma[:, i], 1e-8)

            F[:, i]      = np.maximum(F[:, i], 1e-8)               # 防负价
            F[:, i]     += sigma[:, i] * F[:, i] ** beta[i] * dW_S[:, i]
    return F

###############################################################################
# ▄▀▀▄ █▀▀ █▀▀█ ▄▀▀█ ▀█▀   5)   Worst‑of pricing wrapper
###############################################################################

def price_worst_of_call(
    spot_vec: np.ndarray,
    k_strike: float,
    t_expiry: float,
    r_curve: Callable[[float], float],
    beta_vec: np.ndarray,
    a_spl_list: List[Callable[[float], float]],
    r_spl_list: List[Callable[[float], float]],
    n_spl_list: List[Callable[[float], float]],
    corr_S: np.ndarray,
    n_paths: int = 200_000,
    dt: float = 1 / 365,
    seed: int | None = 42,
    return_std_err: bool = False,
) -> float | Tuple[float, float]:
    """
    定价 worst‑of 欧式看涨期权 (payoff = max(min(F_T) − K, 0))

    若 `return_std_err=True` 则同时返回 MC 方差的标准误差。
    """
    fwd_T = simulate_sabr_multi(
        spot_vec,
        beta_vec,
        a_spl_list,
        r_spl_list,
        n_spl_list,
        r_curve,
        t_expiry,
        corr_S,
        n_paths=n_paths,
        dt=dt,
        seed=seed,
    )
    worst_T = np.min(fwd_T, axis=1)
    payoff = np.maximum(worst_T - k_strike, 0.0)
    disc = math.exp(-r_curve(t_expiry) * t_expiry)
    price = disc * payoff.mean()

    if return_std_err:
        stderr = disc * payoff.std(ddof=1) / math.sqrt(n_paths)
        return price, stderr
    return price

def intersection_tk(iv_tables: list[pd.DataFrame]) -> list[tuple]:
    """
    给定多只资产的 IV 表列表，返回其 (T,K) 共同交集。
    每张表必须至少含列 ["T","K"]。
    输出按 (T 升序, K 升序) 排序。
    """
    # 把每张表转成集合 {(T1,K1), (T2,K2), ...}
    tk_sets = [set(zip(df["T"], df["K"])) for df in iv_tables]
    common  = reduce(set.intersection, tk_sets)
    if not common:
        raise ValueError("No common (T,K) across all assets — 无法对表")
    # 排序后转回 list[tuple]
    return sorted(common, key=lambda x: (x[0], x[1]))

def plot_price_surface(df: pd.DataFrame, out_png="worst_of_price_surface.png"):
    """
    把 df_price pivot 成 (T rows, K cols) → price matrix，并保存热图。
    """
    piv = df.pivot(index="T", columns="K", values="price_mc")
    plt.figure(figsize=(10, 6))
    pcm = plt.pcolormesh(piv.columns, piv.index, piv.values,
                         shading="auto")
    plt.colorbar(pcm, label="MC Price")
    plt.xlabel("Strike K")
    plt.ylabel("Maturity T (year)")
    plt.title("Worst‑of Call Price Surface")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Price surface saved → {out_png}")

# iv_df = build_iv_table("SPY", VAL_DATE,
#                        mny_width=1.0,  # 彻底放宽
#                        min_price=0.0)
# print("after all filters:", len(iv_df))
# print(iv_df.head())
# iv_df = build_iv_table("SPY", VAL_DATE,
#                        mny_width=0.6, min_price=0.05,
#                        min_iv=0.05, max_iv=1.5)
# print("rows after filters:", len(iv_df))
# print(iv_df.head())

### === 5. MAIN ==============================================
# def main():
#     # ------- parameters & containers -----------------------------------
#     r_curve = _zero_curve(VAL_DATE)
#     good_syms, skip_syms = [], []
#     spot_vec, beta_vec   = [], []
#     a_list, r_list, n_list, iv_tables = [], [], [], []
#
#     # ------- two‑pass calibration --------------------------------------
#     strict = dict(mny_width=0.5, min_price=0.05)
#     relaxed= dict(mny_width=0.7, min_price=0.02)
#     for sym in UNDERLYINGS:
#         for params in (strict, relaxed):
#             try:
#                 iv_df = build_iv_table(sym, VAL_DATE, **params)
#                 spot  = float(iv_df["spot"].iat[0])
#                 beta, _, a_s, r_s, n_s = build_sabr_surface(iv_df, spot, r_curve)
#                 good_syms.append(sym); iv_tables.append(iv_df)
#                 spot_vec.append(spot); beta_vec.append(beta)
#                 a_list.append(a_s);   r_list.append(r_s); n_list.append(n_s)
#                 break
#             except Exception as e:
#                 err = str(e); continue
#         else:
#             skip_syms.append(f"{sym} → {err}")
#
#     N = len(good_syms)
#     if N < 2:
#         print("❌ 凑不到两只资产:", "; ".join(skip_syms)); return
#
#     spot_vec = np.asarray(spot_vec); beta_vec = np.asarray(beta_vec)
#
#     # ------- correlation via Polygon hist ------------------------------
#     logret_cols = []
#     for sym in good_syms:
#         s = fetch_hist_polygon(sym)
#         if s is not None and len(s) > 10:
#             logret_cols.append(np.log(s).diff().dropna().rename(sym))
#     if len(logret_cols) >= 2:
#         corr_S = pd.concat(logret_cols, axis=1).corr().values
#     else:
#         corr_S = np.full((N,N), 0.2); np.fill_diagonal(corr_S,1)
#
#     # ------- common (T,K) grid ----------------------------------------
#     tk_pairs = intersection_tk(iv_tables)
#     if not tk_pairs:
#         print("No common (T,K)"); return
#     T_LIST = sorted({t for t,_ in tk_pairs})
#     K_DICT = {T:[K for t,K in tk_pairs if t==T] for T in T_LIST}
#
#     # ------- Monte‑Carlo pricing --------------------------------------
#     price_rows = []
#     for T in T_LIST:
#         F_T = simulate_sabr_multi(spot_vec, beta_vec,
#                                    a_list, r_list, n_list,
#                                    r_curve, T, corr_S,
#                                    n_paths=MC_PATHS, dt=DT_STEP, seed=42)
#         worst = np.min(F_T, axis=1); disc = np.exp(-r_curve(T)*T)
#         for K in K_DICT[T]:
#             payoff = np.maximum(worst-K,0)
#             price_rows.append({"T":T,"K":K,
#                                "price_mc":disc*payoff.mean(),
#                                "stderr":disc*payoff.std(ddof=1)/np.sqrt(MC_PATHS)})
#
#     df_price = (pd.DataFrame(price_rows)
#                 .sort_values(["T","K"]).reset_index(drop=True))
#
#     # ------- diagnostics ----------------------------------------------
#     s_min = spot_vec.min()
#     print("Spot vector     :", spot_vec)
#     print("Intrinsic @K360 :", s_min-360)
#     print("Alpha 0.02y     :", [a(0.02) for a in a_list])
#     print("Nu    0.02y     :", [n(0.02) for n in n_list])
#     atm_iv = iv_tables[0].query("abs(K-@s_min)<1")["iv"].mean()
#     print("ATM market IV   :", atm_iv)
#     print(df_price.head(10))
#     print(f"Priced {len(df_price)} pts for worst‑of on {good_syms}")
#     if skip_syms: print("Skipped symbols:", "; ".join(skip_syms))
#
# if __name__ == "__main__":
#     main()