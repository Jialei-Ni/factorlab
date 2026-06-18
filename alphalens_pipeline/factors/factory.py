"""
factors/factory.py
------------------
Config-driven factor factory.

Adding a new factor:
  1. Create factors/<your_factor>.py extending Factor.
  2. Import it here and add to FACTOR_REGISTRY.
  3. Add its window / param mapping to param_map below.
"""

from .momentum        import MomentumFactor
from .bollinger       import BollingerFactor
from .volume          import VolumeFactor
from .mean_reversion  import MeanReversionFactor
from .stockstats_factor import StockStatsFactor
from .base            import Factor

FACTOR_REGISTRY: dict[str, type[Factor]] = {
    "momentum":       MomentumFactor,
    "bollinger":      BollingerFactor,
    "volume":         VolumeFactor,
    "mean_reversion": MeanReversionFactor,
    "stockstats":     StockStatsFactor,
}


def build_factor(config: dict) -> Factor:
    """
    Instantiate the factor named in config["factor"].

    For "stockstats", config["stockstats_indicator"] sets which indicator
    to compute (default: "rsi_14").
    """
    key = config.get("factor", "momentum")

    if key not in FACTOR_REGISTRY:
        raise ValueError(
            f"Unknown factor '{key}'. "
            f"Available: {list(FACTOR_REGISTRY.keys())}"
        )

    cls = FACTOR_REGISTRY[key]

    param_map: dict[str, dict] = {
        "momentum":       {"window": config.get("momentum_window",       20)},
        "bollinger":      {"window": config.get("bollinger_window",      20)},
        "volume":         {"window": config.get("volume_window",         20)},
        "mean_reversion": {"window": config.get("mean_reversion_window",  5)},
        "stockstats":     {"indicator": config.get("stockstats_indicator", "rsi_14")},
    }

    kwargs = param_map.get(key, {})
    return cls(**kwargs)
