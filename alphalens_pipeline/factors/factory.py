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

    For custom factors, use the registered factor builder.
    If the factor name is not registered, try to resolve it as a
    StockStats indicator.
    """
    key = config.get("factor", "momentum")

    if key in FACTOR_REGISTRY:
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

    # Attempt to resolve unknown factor names as StockStats indicators.
    try:
        return StockStatsFactor(key)
    except ValueError as exc:
        raise ValueError(
            f"Unknown factor '{key}'. Available custom factors: "
            f"{list(FACTOR_REGISTRY.keys())}. "
            f"If this name is a StockStats indicator, use a supported symbol." 
        ) from exc
