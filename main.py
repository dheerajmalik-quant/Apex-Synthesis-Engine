# ==================================================
# SCRIPT FILE: main.py
# ENVIRONMENT: Local Production V1.2 (Pure NumPy / No Numba)
# ==================================================

import json
import polars as pl
import numpy as np
import os
import pandas as pd
import glob
from datetime import date
import warnings
warnings.filterwarnings("ignore") 

os.system('cls' if os.name == 'nt' else 'clear')
print("==========================================================")
print("🪐 V1.2 APEX SYNTHESIS: HIGH-SPEED PURE NUMPY EXECUTION")
print("==========================================================")

# --- 0. LOAD SECURE CONFIGURATION ---
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    print("[*] Secure config.json loaded successfully.")
except FileNotFoundError:
    print("[!] ERROR: config.json not found. Please create it before running.")
    exit()

BASE_PATH = config["data_path"] 
STARTING_CAPITAL = config["capital"]
RISK_PER_TRADE_PCT = config["risk_per_trade_pct"]
FRICTION_PER_LOT = config["friction_per_lot"]
SLIPPAGE_POINTS = config.get("slippage_points", 0.0) 

AGENTS = ["DONCHIAN", "KELTNER", "KAMA", "ICHIMOKU", "SUPERTREND"]
AGENT_MAP = {name: i for i, name in enumerate(AGENTS)}
ATR_TRAILS = config["atr_trails"]

STEP1_ACTIVATION_ATR = config["step1_activation_atr"]
STEP2_ACTIVATION_ATR = config["step2_activation_atr"]
STEP2_TRAIL_GAP_ATR = config["step2_trail_gap_atr"]

# --- 1. HISTORICAL LOT SIZE MATRIX ---
def get_historical_lot_size_vectorized(dates_series):
    """Pre-computes lot sizes for fast array access"""
    conditions = [
        dates_series < date(2021, 7, 1),
        dates_series < date(2024, 4, 1),
        dates_series < date(2024, 11, 1),
        dates_series < date(2026, 1, 1)
    ]
    choices = [75, 50, 25, 75]
    return np.select(conditions, choices, default=65)

# --- 2. ADVANCED MATH ENGINES ---
def calc_adx(h, l, c, n):
    up_m = h - np.roll(h, 1)
    dn_m = np.roll(l, 1) - l
    pos_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0)
    neg_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0)
    
    tr1 = h - l
    tr2 = np.abs(h - np.roll(c, 1))
    tr3 = np.abs(l - np.roll(c, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_ewm = pd.Series(tr).ewm(alpha=1/n, adjust=False).mean().values
    pos_dm_ewm = pd.Series(pos_dm).ewm(alpha=1/n, adjust=False).mean().values
    neg_dm_ewm = pd.Series(neg_dm).ewm(alpha=1/n, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pos_di = 100 * pos_dm_ewm / tr_ewm
        neg_di = 100 * neg_dm_ewm / tr_ewm
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
    
    adx = pd.Series(dx).ewm(alpha=1/n, adjust=False).mean().values
    return np.nan_to_num(adx)

def calc_kama_fast(close, n, fast, slow):
    change = np.abs(close - np.roll(close, n))
    diff = np.abs(np.append(np.array([0.0]), np.diff(close)))
    volatility = np.zeros_like(close)
    for i in range(n, len(close)): volatility[i] = np.sum(diff[i-n+1:i+1])
    
    er = np.zeros_like(close)
    for i in range(len(close)):
        if volatility[i] != 0: er[i] = change[i] / volatility[i]
        
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    kama = np.zeros_like(close)
    kama[n-1] = close[n-1]
    for i in range(n, len(close)): kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calc_supertrend_fast(h, l, c, atr, multiplier):
    hl2 = (h + l) / 2.0
    final_ub = np.zeros_like(c)
    final_lb = np.zeros_like(c)
    trend = np.ones_like(c)
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)
    
    for i in range(1, len(c)):
        if basic_ub[i] < final_ub[i-1] or c[i-1] > final_ub[i-1]: final_ub[i] = basic_ub[i]
        else: final_ub[i] = final_ub[i-1]
        
        if basic_lb[i] > final_lb[i-1] or c[i-1] < final_lb[i-1]: final_lb[i] = basic_lb[i]
        else: final_lb[i] = final_lb[i-1]
        
        if final_ub[i-1] == 0: final_ub[i] = basic_ub[i]
        if final_lb[i-1] == 0: final_lb[i] = basic_lb[i]
        
        if c[i] > final_ub[i-1]: trend[i] = 1
        elif c[i] < final_lb[i-1]: trend[i] = -1
        else: trend[i] = trend[i-1]
    return trend

# --- 3. HIGH-SPEED EXECUTION ENGINE ---
def run_backtest_fast(
    o, h, l, c, atr, ema_trend, ema_macro, don_u, don_l, kel_u, kel_l,
    senk_arr, kama_arr, adx_arr, st_trend, hour_arr, lot_sizes,
    agent_id, trail_mult, start_cap, risk_pct, friction, slip_pts,
    exit_hr, chop_thresh, s1_atr, s2_atr, s2_gap
):
    in_pos = False
    pos_type = 0
    entry_p = 0.0
    stop_l = 0.0
    peak_price = 0.0
    trade_qty_active = 0
    trade_lots_active = 0
    entry_idx = 0  # Fixed: Tracking the exact entry index
    
    current_capital = start_cap
    executed_trades = []

    for i in range(1, len(c) - 1):
        if in_pos:
            exit_p = 0.0
            if pos_type == 1:
                if o[i] <= stop_l: exit_p = o[i] - slip_pts
                elif l[i] <= stop_l: exit_p = stop_l - slip_pts
                else:
                    peak_price = max(peak_price, h[i])
                    current_stop = c[i] - (atr[i] * trail_mult)
                    peak_pts = peak_price - entry_p
                    
                    if peak_pts >= (s2_atr * atr[i]): 
                        current_stop = max(current_stop, peak_price - (s2_gap * atr[i]))
                    elif peak_pts >= (s1_atr * atr[i]): 
                        current_stop = max(current_stop, entry_p + (friction / trade_qty_active) + slip_pts)
                    stop_l = max(stop_l, current_stop)
            
            elif pos_type == -1:
                if o[i] >= stop_l: exit_p = o[i] + slip_pts
                elif h[i] >= stop_l: exit_p = stop_l + slip_pts
                else:
                    peak_price = min(peak_price, l[i])
                    current_stop = c[i] + (atr[i] * trail_mult)
                    peak_pts = entry_p - peak_price
                    
                    if peak_pts >= (s2_atr * atr[i]): 
                        current_stop = min(current_stop, peak_price + (s2_gap * atr[i]))
                    elif peak_pts >= (s1_atr * atr[i]): 
                        current_stop = min(current_stop, entry_p - (friction / trade_qty_active) - slip_pts)
                    stop_l = min(stop_l, current_stop)
            
            if exit_p > 0:
                raw_pts = (exit_p - entry_p) if pos_type == 1 else (entry_p - exit_p)
                net = (raw_pts * trade_qty_active) - (friction * trade_lots_active)
                current_capital += net
                
                # Fixed: Using entry_idx instead of exit index i
                executed_trades.append({
                    "entry_idx": entry_idx, "exit_idx": i, "direction": pos_type,
                    "entry_p": entry_p, "exit_p": exit_p, "qty": trade_qty_active,
                    "pnl": net, "capital": current_capital
                })
                in_pos = False

        valid_entry_time = hour_arr[i] < exit_hr
        is_trending = adx_arr[i] > chop_thresh
        
        if not in_pos and valid_entry_time and is_trending:
            cond_long, cond_short = False, False
            trend_up = c[i] > ema_trend[i] and c[i] > ema_macro[i]
            trend_dn = c[i] < ema_trend[i] and c[i] < ema_macro[i]
            
            if agent_id == 0: 
                if c[i] > don_u[i] and trend_up: cond_long = True
                elif c[i] < don_l[i] and trend_dn: cond_short = True
            elif agent_id == 1: 
                if c[i] > kel_u[i] and trend_up: cond_long = True
                elif c[i] < kel_l[i] and trend_dn: cond_short = True
            elif agent_id == 2: 
                if c[i] > kama_arr[i] and c[i-1] <= kama_arr[i-1] and trend_up: cond_long = True
                elif c[i] < kama_arr[i] and c[i-1] >= kama_arr[i-1] and trend_dn: cond_short = True
            elif agent_id == 3: 
                if c[i] > senk_arr[i] and c[i-1] <= senk_arr[i-1] and trend_up: cond_long = True
                elif c[i] < senk_arr[i] and c[i-1] >= senk_arr[i-1] and trend_dn: cond_short = True
            elif agent_id == 4: 
                if st_trend[i] == 1 and st_trend[i-1] == -1 and trend_up: cond_long = True
                elif st_trend[i] == -1 and st_trend[i-1] == 1 and trend_dn: cond_short = True
                
            if cond_long or cond_short:
                if current_capital <= 0: break 
                
                risk_amount = current_capital * risk_pct
                stop_loss_pts = atr[i+1] * trail_mult
                raw_qty = risk_amount / stop_loss_pts
                
                base_lot = lot_sizes[i+1]
                trade_qty_active = max(base_lot, int((raw_qty // base_lot) * base_lot))
                trade_lots_active = trade_qty_active // base_lot
                
                if cond_long: 
                    pos_type, in_pos = 1, True
                    entry_p = o[i+1] + slip_pts
                    peak_price = entry_p
                    stop_l = entry_p - (atr[i+1] * trail_mult)
                    entry_idx = i + 1  # Fixed: Capture the exact bar index
                elif cond_short: 
                    pos_type, in_pos = -1, True
                    entry_p = o[i+1] - slip_pts
                    peak_price = entry_p
                    stop_l = entry_p + (atr[i+1] * trail_mult)
                    entry_idx = i + 1  # Fixed: Capture the exact bar index

    return executed_trades

master_results = []
all_trades_ledger = []

# --- 4. DYNAMIC FILE DISCOVERY ---
parquet_files = glob.glob(os.path.join(BASE_PATH, "*.parquet"))

if not parquet_files:
    print(f"[!] No .parquet files found in {BASE_PATH}.")
else:
    for filepath in parquet_files:
        filename = os.path.basename(filepath)
        tf_name = filename.split('_')[2] if len(filename.split('_')) > 2 else filename
        print(f"[*] Crunching Multi-Dimensional Physics: {tf_name}...")
        
        df = pl.read_parquet(filepath).drop_nulls()
        
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime(time_zone="Asia/Kolkata").dt.replace_time_zone(None).alias("datetime")
        ).with_columns(
            pl.col("datetime").dt.date().alias("date"),
            pl.col("datetime").dt.hour().alias("hour"),
            prev_close = pl.col("close").shift(1)
        )
        
        df = df.with_columns(
            tr1 = pl.col("high") - pl.col("low"),
            tr2 = (pl.col("high") - pl.col("prev_close")).abs(),
            tr3 = (pl.col("low") - pl.col("prev_close")).abs()
        ).with_columns(tr = pl.max_horizontal(["tr1", "tr2", "tr3"]))
        
        df = df.with_columns(
            atr = pl.col("tr").ewm_mean(span=config["atr_span"], adjust=False),
            ema_trend = pl.col("close").ewm_mean(span=config["ema_trend"], adjust=False),
            ema_macro = pl.col("close").ewm_mean(span=config["ema_macro"], adjust=False),
            sma_base = pl.col("close").rolling_mean(window_size=config["base_window"]),
            atr_base = pl.col("tr").rolling_mean(window_size=config["base_window"]),
            donchian_u = pl.col("high").shift(1).rolling_max(window_size=config["base_window"]),
            donchian_l = pl.col("low").shift(1).rolling_min(window_size=config["base_window"]),
            senkou_b = ((pl.col("high").rolling_max(window_size=config["ichimoku_window"]) + pl.col("low").rolling_min(window_size=config["ichimoku_window"])) / 2).shift(config["ichimoku_shift"])
        ).with_columns(
            keltner_u = pl.col("sma_base") + (config["keltner_mult"] * pl.col("atr_base")),
            keltner_l = pl.col("sma_base") - (config["keltner_mult"] * pl.col("atr_base"))
        ).drop_nulls()

        o, c, h, l, atr = df["open"].to_numpy(), df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["atr"].to_numpy()
        ema_trend, ema_macro = df["ema_trend"].to_numpy(), df["ema_macro"].to_numpy()
        hour_arr = df["hour"].to_numpy()
        dts = df["datetime"].to_numpy()
        lot_sizes = get_historical_lot_size_vectorized(df["date"].to_numpy())
        
        don_u, don_l = df["donchian_u"].to_numpy(), df["donchian_l"].to_numpy()
        kel_u, kel_l = df["keltner_u"].to_numpy(), df["keltner_l"].to_numpy()
        senk_arr = df["senkou_b"].to_numpy()
        
        kama_arr = calc_kama_fast(c, n=config["kama_n"], fast=config["kama_fast"], slow=config["kama_slow"])
        adx_arr = calc_adx(h, l, c, n=config["adx_period"])

        for agent_name in AGENTS:
            agent_id = AGENT_MAP[agent_name]
            for trail_mult in ATR_TRAILS:
                st_trend = calc_supertrend_fast(h, l, c, atr, multiplier=trail_mult) if agent_name == "SUPERTREND" else np.zeros_like(c)
                
                trades = run_backtest_fast(
                    o, h, l, c, atr, ema_trend, ema_macro, don_u, don_l, kel_u, kel_l,
                    senk_arr, kama_arr, adx_arr, st_trend, hour_arr, lot_sizes,
                    agent_id, float(trail_mult), float(STARTING_CAPITAL), float(RISK_PER_TRADE_PCT), 
                    float(FRICTION_PER_LOT), float(SLIPPAGE_POINTS), int(config["exit_hour"]), 
                    float(config["adx_chop_threshold"]), float(STEP1_ACTIVATION_ATR), 
                    float(STEP2_ACTIVATION_ATR), float(STEP2_TRAIL_GAP_ATR)
                )

                if trades:
                    cap_curve = np.array([STARTING_CAPITAL] + [t["capital"] for t in trades])
                    pnls = np.array([t["pnl"] for t in trades])
                    
                    max_dd_pct = np.min((cap_curve - np.maximum.accumulate(cap_curve)) / np.maximum.accumulate(cap_curve)) * 100 
                    time_delta_years = (dts[-1] - dts[0]).astype('timedelta64[D]') / np.timedelta64(365, 'D')
                    cagr = ((cap_curve[-1] / STARTING_CAPITAL) ** (1 / time_delta_years) - 1) * 100 if cap_curve[-1] > 0 and time_delta_years > 0 else 0
                    
                    master_results.append({
                        "Engine": agent_name, "Timeframe": tf_name, "ATR": trail_mult,
                        "Trades": len(trades), "Win_%": round((len(pnls[pnls > 0]) / len(trades)) * 100, 2),
                        "End_Cap_INR": round(cap_curve[-1], 2), "CAGR_%": round(cagr, 2),
                        "Max_DD_%": round(max_dd_pct, 2), "True_Calmar": round(cagr / abs(max_dd_pct) if max_dd_pct != 0 else 0, 3)
                    })
                    
                    for t in trades:
                        all_trades_ledger.append({
                            "Engine": agent_name, "Timeframe": tf_name, "ATR_Trail": trail_mult,
                            "Entry_Time": dts[t["entry_idx"]], "Exit_Time": dts[t["exit_idx"]],
                            "Direction": "LONG" if t["direction"] == 1 else "SHORT",
                            "Quantity": t["qty"], "Entry_Price": t["entry_p"], 
                            "Exit_Price": t["exit_p"], "Net_PnL": t["pnl"],
                            "Capital_After": t["capital"]
                        })

df_res = pd.DataFrame(master_results)
if not df_res.empty:
    df_res = df_res.sort_values("True_Calmar", ascending=False)
    
    # --- 5. LOCAL EXPORT ---
    summary_filename = "V1.2_Apex_Summary.csv"
    ledger_filename = "V1.2_Apex_Trade_Ledger.csv"
    
    df_res.to_csv(summary_filename, index=False)
    pd.DataFrame(all_trades_ledger).to_csv(ledger_filename, index=False)
    
    print("\n" + "="*90)
    print("🏆 V1.2 APEX SYNTHESIS EXECUTED SUCCESSFULLY")
    print(f"✅ Summary Saved to: ./{summary_filename}")
    print(f"✅ Trade Ledger Saved to: ./{ledger_filename}")
    print("="*90)
    print(df_res.head(20).to_string(index=False))
else:
    print("\n[!] No trades were executed. Check your data paths and parameters.")