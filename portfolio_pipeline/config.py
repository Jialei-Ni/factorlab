from __future__ import annotations

from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "universe_name": "default_universe",
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "signals": {
        "close_20_sma": -1.0,
        # "tr": 0.25,
        # "atr": 0.25,
        # "close_10_mad": -0.25,

        # "vr": -1.0,
        # "trix": 0.5,
        # "pvo": 0.5,
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
