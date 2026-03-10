# 🪐 V1.1 Apex Synthesis: Multi-Timeframe Backtest Engine

A high-performance, multi-timeframe backtesting engine built in Python for analyzing Nifty 50 historical data. Engineered for speed and precision, this project transitions from standard Pandas iterations to a heavily optimized `polars` and vectorized `numpy` architecture.

## 🚀 Key Features
* **Multi-Timeframe Processing:** Seamlessly processes 5-min, 10-min, 15-min, 30-min, and 60-min intervals.
* **Vectorized Math Engines:** Custom implementations of ADX, KAMA, and SuperTrend for zero-lag execution.
* **Dynamic Risk Parity:** Simulates real-world institutional position sizing using historical Nifty 50 lot-size matrices.
* **Step-Trail Alpha Management:** Advanced peak-to-trough trailing stop logic to eliminate look-ahead bias.
* **Secure Configuration:** Alpha parameters and directory paths are decoupled from the core logic via a secure JSON architecture.

## 🛠️ Architecture
* **Data Processing:** `polars` (Parquet integration)
* **Execution Math:** `numpy`
* **Environment:** Local Production 

## ⚙️ How to Run
1. Clone the repository.
2. Rename `config.example.json` to `config.json`.
3. Add your local `.parquet` data paths and proprietary indicator parameters to `config.json`.
4. Install dependencies: `pip install polars pandas numpy pyarrow`
5. Execute the engine: `python main.py`