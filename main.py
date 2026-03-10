# ==================================================
# SCRIPT FILE: main.py
# ENVIRONMENT: Local Production
# ==================================================

import json
import polars as pl
import numpy as np
import os
import pandas as pd
from datetime import date, datetime, timezone

os.system('cls' if os.name == 'nt' else 'clear')
print("==========================================================")
print("🪐 V1.1 APEX SYNTHESIS: SECURE LOCAL EXECUTION")
print("==========================================================")

# --- 0. LOAD SECURE CONFIGURATION ---
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    print("[*] Secure config.json loaded successfully.")
except FileNotFoundError:
    print("[!] ERROR: config.json not found. Please create it before running.")
    exit()

# Extract Configuration Variables
BASE_PATH = config["data_path"] 
STARTING_CAPITAL = config["capital"]
RISK_PER_TRADE_PCT = config["risk_per_trade_pct"]
FRICTION_PER_LOT = config["friction_per_lot"]

AGENTS = ["DONCHIAN", "KELTNER", "KAMA", "ICHIMOKU", "SUPERTREND"]
ATR_TRAILS = config["atr_trails"]

STEP1_ACTIVATION_ATR = config["step1_activation_atr"]
STEP2_ACTIVATION_ATR = config["step2_activation_atr"]
STEP2_TRAIL_GAP_ATR = config["step2_trail_gap_atr"]

# --- 1. THE DATA VAULT ---
TIMEFRAMES = {
    "5_MIN": "NIFTY_SPOT_5minute_2019_2026.parquet",
    "10_MIN": "NIFTY_SPOT_10minute_2019_2026.parquet",
    "15_MIN": "NIFTY_SPOT_15minute_2019_2026.parquet",
    "30_MIN": "NIFTY_SPOT_30minute_2019_2026.parquet",
    "60_MIN": "NIFTY_SPOT_60minute_2019_2026.parquet"
}

# --- HISTORICAL NIFTY LOT SIZE MATRIX ---
def get_historical_lot_size(trade_date):
    if trade_date < date(2021, 7, 1): return 75
    elif trade_date < date(2024, 4, 1): return 50
    elif trade_date < date(2024, 11, 1): return 25
    elif trade_date < date(2026, 1, 1): return 75
    else: return 65

# --- 2. ADVANCED COMPILED MATH ENGINES ---
def calc_adx(h, l, c, n):
    """Vectorized ADX (Average Directional Index) for Chop Filtering"""
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

def calc_kama(close, n, fast, slow):
    change = np.abs(close - np.roll(close, n))
    diff = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(n, len(close)): volatility[i] = np.sum(diff[i-n+1:i+1])
    er = np.zeros_like(close)
    np.divide(change, volatility, out=er, where=volatility!=0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    kama = np.zeros_like(close)
    kama[n-1] = close[n-1]
    for i in range(n, len(close)): kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calc_supertrend(h, l, c, atr, multiplier):
    hl2 = (h + l) / 2
    final_ub, final_lb, trend = np.zeros_like(c), np.zeros_like(c), np.ones_like(c)
    basic_ub, basic_lb = hl2 + (multiplier * atr), hl2 - (multiplier * atr)
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

master_results = []

# --- 3. EXECUTION ENGINE ---
for tf_name, filename in TIMEFRAMES.items():
    filepath = os.path.join(BASE_PATH, filename)
    if not os.path.exists(filepath): 
        print(f"[!] {filename} missing at {filepath}. Skipping {tf_name}.")
        continue
        
    print(f"[*] Crunching Multi-Dimensional Physics: {tf_name}...")
    df = pl.read_parquet(filepath).drop_nulls()
    
    # -------------------------------------------------------------
    # TIMEZONE FIX APPLIED HERE
    # -------------------------------------------------------------
    df = df.with_columns(
        pl.col("timestamp").str.to_datetime(time_zone="Asia/Kolkata").dt.replace_time_zone(None).alias("datetime")
    ).with_columns(
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.year().alias("year"),
        pl.col("datetime").dt.time().alias("time"),
        prev_close = pl.col("close").shift(1)
    )
    
    df = df.with_columns(
        tr1 = pl.col("high") - pl.col("low"),
        tr2 = (pl.col("high") - pl.col("prev_close")).abs(),
        tr3 = (pl.col("low") - pl.col("prev_close")).abs()
    ).with_columns(tr = pl.max_horizontal(["tr1", "tr2", "tr3"]))
    
    # DYNAMIC INJECTIONS FROM CONFIG
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
    dts, tms, yrs = df["date"].to_list(), df["time"].to_list(), df["year"].to_numpy()
    
    don_u, don_l = df["donchian_u"].to_numpy(), df["donchian_l"].to_numpy()
    kel_u, kel_l = df["keltner_u"].to_numpy(), df["keltner_l"].to_numpy()
    senk_arr = df["senkou_b"].to_numpy()
    
    # DYNAMIC INJECTIONS FOR MATH ENGINES
    kama_arr = calc_kama(c, n=config["kama_n"], fast=config["kama_fast"], slow=config["kama_slow"])
    adx_arr = calc_adx(h, l, c, n=config["adx_period"])

    for agent_name in AGENTS:
        for trail_mult in ATR_TRAILS:
            st_trend = calc_supertrend(h, l, c, atr, multiplier=trail_mult) if agent_name == "SUPERTREND" else None
            
            in_pos, pos_type, entry_p, stop_l, peak_price = False, 0, 0.0, 0.0, 0.0
            trade_lots_active, trade_qty_active = 0, 0
            trade_pnls, capital_curve = [], [STARTING_CAPITAL]
            
            for i in range(1, len(c) - 1):
                # EXIT LOGIC
                if in_pos:
                    exit_p = 0.0
                    if pos_type == 1:
                        if o[i] <= stop_l: exit_p = o[i] 
                        elif l[i] <= stop_l: exit_p = stop_l
                        else:
                            peak_price = max(peak_price, h[i])
                            current_stop = c[i] - (atr[i] * trail_mult)
                            peak_pts = peak_price - entry_p
                            
                            if peak_pts >= (STEP2_ACTIVATION_ATR * atr[i]): 
                                current_stop = max(current_stop, peak_price - (STEP2_TRAIL_GAP_ATR * atr[i]))
                            elif peak_pts >= (STEP1_ACTIVATION_ATR * atr[i]): 
                                current_stop = max(current_stop, entry_p + (FRICTION_PER_LOT / trade_qty_active))
                            stop_l = max(stop_l, current_stop)
                    
                    elif pos_type == -1:
                        if o[i] >= stop_l: exit_p = o[i]
                        elif h[i] >= stop_l: exit_p = stop_l
                        else:
                            peak_price = min(peak_price, l[i])
                            current_stop = c[i] + (atr[i] * trail_mult)
                            peak_pts = entry_p - peak_price
                            
                            if peak_pts >= (STEP2_ACTIVATION_ATR * atr[i]): 
                                current_stop = min(current_stop, peak_price + (STEP2_TRAIL_GAP_ATR * atr[i]))
                            elif peak_pts >= (STEP1_ACTIVATION_ATR * atr[i]): 
                                current_stop = min(current_stop, entry_p - (FRICTION_PER_LOT / trade_qty_active))
                            stop_l = min(stop_l, current_stop)
                    
                    if exit_p > 0:
                        raw_pts = (exit_p - entry_p) if pos_type == 1 else (entry_p - exit_p)
                        net = (raw_pts * trade_qty_active) - (FRICTION_PER_LOT * trade_lots_active)
                        trade_pnls.append(net)
                        capital_curve.append(capital_curve[-1] + net) 
                        in_pos = False

                # ENTRY LOGIC
                valid_entry_time = tms[i].hour < config["exit_hour"]
                is_trending = adx_arr[i] > config["adx_chop_threshold"]
                
                if not in_pos and valid_entry_time and is_trending:
                    cond_long, cond_short = False, False
                    
                    trend_up = c[i] > ema_trend[i] and c[i] > ema_macro[i]
                    trend_dn = c[i] < ema_trend[i] and c[i] < ema_macro[i]
                    
                    if agent_name == "DONCHIAN":
                        if c[i] > don_u[i] and trend_up: cond_long = True
                        elif c[i] < don_l[i] and trend_dn: cond_short = True
                    elif agent_name == "KELTNER":
                        if c[i] > kel_u[i] and trend_up: cond_long = True
                        elif c[i] < kel_l[i] and trend_dn: cond_short = True
                    elif agent_name == "KAMA":
                        if c[i] > kama_arr[i] and c[i-1] <= kama_arr[i-1] and trend_up: cond_long = True
                        elif c[i] < kama_arr[i] and c[i-1] >= kama_arr[i-1] and trend_dn: cond_short = True
                    elif agent_name == "ICHIMOKU":
                        if c[i] > senk_arr[i] and c[i-1] <= senk_arr[i-1] and trend_up: cond_long = True
                        elif c[i] < senk_arr[i] and c[i-1] >= senk_arr[i-1] and trend_dn: cond_short = True
                    elif agent_name == "SUPERTREND":
                        if st_trend[i] == 1 and st_trend[i-1] == -1 and trend_up: cond_long = True
                        elif st_trend[i] == -1 and st_trend[i-1] == 1 and trend_dn: cond_short = True
                        
                    if cond_long or cond_short:
                        current_equity = capital_curve[-1]
                        if current_equity <= 0: break 
                        
                        risk_amount = current_equity * RISK_PER_TRADE_PCT
                        stop_loss_pts = atr[i+1] * trail_mult
                        raw_qty = risk_amount / stop_loss_pts
                        
                        base_lot_size = get_historical_lot_size(dts[i+1])
                        trade_qty_active = max(base_lot_size, int((raw_qty // base_lot_size) * base_lot_size))
                        trade_lots_active = trade_qty_active // base_lot_size
                        
                        if cond_long: 
                            pos_type, entry_p, peak_price, in_pos = 1, o[i+1], o[i+1], True
                            stop_l = o[i+1] - (atr[i+1] * trail_mult)
                        elif cond_short: 
                            pos_type, entry_p, peak_price, in_pos = -1, o[i+1], o[i+1], True
                            stop_l = o[i+1] + (atr[i+1] * trail_mult)

            if trade_pnls:
                np_pnls, cap_array = np.array(trade_pnls), np.array(capital_curve)
                max_dd_pct = np.min((cap_array - np.maximum.accumulate(cap_array)) / np.maximum.accumulate(cap_array)) * 100 
                time_delta_years = (dts[-1] - dts[0]).days / 365.25
                cagr = ((cap_array[-1] / STARTING_CAPITAL) ** (1 / time_delta_years) - 1) * 100 if cap_array[-1] > 0 and time_delta_years > 0 else 0
                
                master_results.append({
                    "Engine": agent_name, "Timeframe": tf_name, "ATR": trail_mult,
                    "Trades": len(trade_pnls), "Win_%": round((len(np_pnls[np_pnls > 0]) / len(trade_pnls)) * 100, 2),
                    "End_Cap_INR": round(cap_array[-1], 2), "CAGR_%": round(cagr, 2),
                    "Max_DD_%": round(max_dd_pct, 2), "True_Calmar": round(cagr / abs(max_dd_pct) if max_dd_pct != 0 else 0, 3)
                })

df_res = pd.DataFrame(master_results)
if not df_res.empty:
    df_res = df_res.sort_values("True_Calmar", ascending=False)
    
    # --- 4. LOCAL EXPORT ---
    csv_filename = "V1.1_Apex_Synthesis.csv"
    df_res.to_csv(csv_filename, index=False)
    
    print("\n" + "="*90)
    print("🏆 V1.1 APEX SYNTHESIS COMPILED & EXECUTED")
    print(f"✅ Master Ledger Saved Successfully to: ./{csv_filename}")
    print("="*90)
    print(df_res.head(20).to_string(index=False))
else:
    print("\n[!] No trades were executed. Check your data paths and parameters.")