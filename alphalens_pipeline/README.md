# Alphalens Research Pipeline

A modular, research-grade factor analysis pipeline built on **yfinance**, **pandas**, and **alphalens-reloaded**.

---

## Project Structure

```
alphalens_pipeline/
│
├── main.py                      # Orchestration entry point
├── config.py                    # All settings in one place
├── universe.py                  # Stock universe definition
│
├── data/
│   ├── fetcher.py               # yfinance download + parquet cache
│   └── cleaner.py               # Missing-value handling + returns
│
├── factors/
│   ├── base.py                  # Abstract Factor interface
│   ├── factory.py               # Config-driven factor builder
│   ├── momentum.py              # Cross-sectional momentum
│   ├── bollinger.py             # Bollinger z-score
│   ├── volume.py                # Relative volume
│   └── mean_reversion.py        # Short-term contrarian
│
├── pipeline/
│   ├── builder.py               # Factor computation + formatting
│   └── alphalens_adapter.py     # Alphalens tear sheets + CSV export
│
├── data/cache/                  # Parquet cache (auto-created)
└── output/                      # Plots and CSVs (auto-created)
```

---

## Quickstart

### Install dependencies
```bash
pip install yfinance alphalens-reloaded pandas pyarrow matplotlib
```

### Run with defaults (momentum factor, 2022–2024)
```bash
python main.py
```

### Override factor via CLI
```bash
python main.py --factor bollinger
python main.py --factor volume
python main.py --factor mean_reversion
```

### Override date range
```bash
python main.py --factor momentum --start 2020-01-01 --end 2023-01-01
```

---

## Configuration (`config.py`)

| Key | Default | Description |
|-----|---------|-------------|
| `factor` | `"momentum"` | Factor to use: `momentum`, `bollinger`, `volume`, `mean_reversion` |
| `start_date` | `"2022-01-01"` | Backtest start |
| `end_date` | `"2024-01-01"` | Backtest end |
| `momentum_window` | `20` | Lookback days for momentum |
| `bollinger_window` | `20` | Rolling window for Bollinger z-score |
| `volume_window` | `20` | Rolling window for relative volume |
| `mean_reversion_window` | `5` | Lookback days for mean reversion |
| `quantiles` | `5` | Number of Alphalens quantile buckets |
| `periods` | `(1, 5, 10)` | Forward return horizons (trading days) |
| `min_data_fraction` | `0.85` | Drop ticker if <85% rows are valid |

---

## Caching

Raw OHLCV data is stored as **parquet** files in `data/cache/`.  
Cache key = `<ticker>_<start>_<end>.parquet`.

- **First run**: downloads from yfinance (slow)  
- **Subsequent runs**: loads from disk (fast)

To force a re-download, delete the relevant `.parquet` files.

---

## Outputs (`output/`)

| File | Description |
|------|-------------|
| `returns_tear_sheet.png` | Quantile returns, cumulative performance |
| `ic_tear_sheet.png` | Information Coefficient over time |
| `turnover_tear_sheet.png` | Factor turnover analysis |
| `ic_by_date.csv` | Daily IC values |
| `ic_summary.csv` | IC descriptive statistics |
| `mean_return_by_quantile.csv` | Mean forward returns per quantile |

---

## Adding a New Factor

1. Create `factors/my_factor.py`:

```python
from .base import Factor
import pandas as pd

class MyFactor(Factor):
    def __init__(self, window: int = 10):
        self.window = window

    def compute(self, prices: pd.DataFrame, volume=None) -> pd.Series:
        score = ...  # your logic, same shape as prices
        return self._stack(score)
```

2. Register it in `factors/factory.py`:

```python
from .my_factor import MyFactor

FACTOR_REGISTRY = {
    ...
    "my_factor": MyFactor,
}

param_map = {
    ...
    "my_factor": {"window": config.get("my_factor_window", 10)},
}
```

3. Add the window param to `config.py` and run:

```bash
python main.py --factor my_factor
```

---

## Architecture

```
Universe
   ↓
Data Fetcher  (yfinance + parquet cache)
   ↓
Data Cleaner  (ffill, drop sparse tickers)
   ↓
Return Builder
   ↓
Factor Engine (pluggable via config)
   ↓
Pipeline Builder (format for Alphalens)
   ↓
Alphalens Adapter
   ↓
output/ (tear sheets + CSVs)
```
