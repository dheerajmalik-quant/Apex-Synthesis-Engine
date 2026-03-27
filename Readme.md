# 🪐 V1.2 Apex Synthesis: High-Speed Multi-Timeframe Backtest Engine

A high-performance, multi-timeframe backtesting engine built in Python for analyzing Nifty 50 historical data. Engineered for speed and precision, V1.2 completely abandons standard Pandas iterations in favor of a heavily optimized, dependency-free `Pure NumPy` architecture.

## 🚀 Key Features
* **Dynamic Timeframe Discovery:** Automatically detects and processes all `.parquet` data files in your specified directory without hardcoding paths.
* **Vectorized Math Engines:** Custom implementations of ADX, KAMA, and SuperTrend for zero-lag execution.
* **High-Resolution Trade Ledger:** Automatically compiles a granular `Trade_Ledger.csv` recording the exact entry/exit points, sizing, and PnL of every single trade alongside the master summary.
* **Dynamic Risk & Friction Parity:** Simulates real-world institutional position sizing using historical Nifty 50 lot-size matrices, fully supporting dynamic point-based slippage.
* **Step-Trail Alpha Management:** Advanced peak-to-trough trailing stop logic to eliminate look-ahead bias.

## 🛠️ Architecture
* **Data Ingestion:** `polars`
* **Execution Core:** `numpy` (High-Speed Pure Array Iteration)
* **Environment:** Local Production 

## ⚙️ How to Run
1. Clone the repository.
2. Rename `config.example.json` to `config.json`.
3. Add your local `.parquet` data paths and proprietary indicator parameters to `config.json`.
4. Install dependencies: `pip install polars pandas numpy pyarrow`
5. Execute the engine: `python main.py`