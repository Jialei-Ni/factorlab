from __future__ import annotations

from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "universe_name": "default_universe",
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "signals": {
        "close_20_sma": 0.1,
        "supertrend": 0.1,
        "boll": 0.1,
        "tr": 0.1,
        "close_10_mad": 0.1,
        
        "atr": 0.1,
        "trix": 0.1,
        "kst": 0.1,
        "vr": 0.1,
        "pvo": 0.1,
        "eribull": 0.1,
        "ppo": 0.1,
        "coppock": 0.1,
    },
    "portfolio_type": "long_short",
    "long_quantile": 0.9,
    "short_quantile": 0.1,
    "holding_days": 10,
    "rebalance_frequency": 5,
    "commission_bps": 5.0,
    "slippage_bps": 5.0,
    "min_data_fraction": 0.85,
    "cache_dir": "data/cache",
}
