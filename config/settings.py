"""Typed configuration loader.

The YAML file is parsed into frozen dataclasses so that every component receives
validated, typed settings, and so that parameter variations (sensitivity tests,
walk-forward candidates) are created with :func:`Config.override` instead of
mutating shared state.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    start_date: str
    end_date: Optional[str]
    universe_file: str
    cache_dir: str
    snapshot_file: str
    benchmark: str
    index_ticker: str
    risk_free_ticker: str
    sec_user_agent: str
    max_daily_move_flag: float


@dataclass(frozen=True)
class PeriodConfig:
    backtest_start: str
    train_end: str
    validation_end: str


@dataclass(frozen=True)
class FactorWeights:
    momentum: float
    trend: float
    sector: float
    quality: float
    value: float
    growth: float
    risk_penalty: float


@dataclass(frozen=True)
class StrategyConfig:
    rebalance_frequency: str
    momentum_lookback: int
    momentum_skip: int
    sector_momentum_lookback: int
    trend_sma: int
    fast_sma: int
    rsi_period: int
    rsi_max_entry: float
    vol_lookback: int
    beta_lookback: int
    atr_period: int
    min_history_days: int
    min_price: float
    min_avg_dollar_volume: float
    winsorize_z: float
    weights: FactorWeights
    min_buy_score: float
    sell_score: float
    hold_rank_buffer: float


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float
    max_positions: int
    sizing_method: str
    max_position_weight: float
    max_sector_weight: float
    min_cash_weight: float
    max_position_risk: float
    target_volatility: Optional[float]
    rebalance_threshold: float
    cash_earns_risk_free: bool


@dataclass(frozen=True)
class RiskConfig:
    use_stop_loss: bool
    stop_method: str
    stop_loss_vol_multiple: float
    stop_loss_atr_multiple: float
    market_regime_filter: bool
    regime_gross_cap: float
    drawdown_limit: float
    drawdown_gross_cap: float
    drawdown_reset: float
    var_confidence: float


@dataclass(frozen=True)
class CostConfig:
    commission_per_share: float
    min_commission: float
    half_spread_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class PaperConfig:
    state_dir: str
    starting_capital: float
    fill_model: str


@dataclass(frozen=True)
class EmailConfig:
    enabled: bool
    smtp_server: str
    smtp_port: int
    sender: str
    recipient: str
    password_env_var: str


@dataclass(frozen=True)
class AlertConfig:
    enabled: bool
    drawdown_alert: float
    macos_notifications: bool
    email: EmailConfig


@dataclass(frozen=True)
class WalkForwardConfig:
    first_test_year: int
    candidates: List[Dict[str, Any]]


@dataclass(frozen=True)
class OutputConfig:
    results_dir: str
    reports_dir: str


@dataclass(frozen=True)
class Config:
    data: DataConfig
    periods: PeriodConfig
    strategy: StrategyConfig
    portfolio: PortfolioConfig
    risk: RiskConfig
    costs: CostConfig
    paper_trading: PaperConfig
    alerts: AlertConfig
    walk_forward: WalkForwardConfig
    output: OutputConfig

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def override(self, overrides: Dict[str, Any]) -> "Config":
        """Return a new Config with dotted-path overrides applied.

        Example: ``cfg.override({"portfolio.max_positions": 10})``.
        Unknown keys raise ``KeyError`` so typos cannot silently do nothing.
        """
        raw = self.to_dict()
        for dotted, value in overrides.items():
            node = raw
            parts = dotted.split(".")
            for part in parts[:-1]:
                if part not in node:
                    raise KeyError(f"Unknown config section '{part}' in '{dotted}'")
                node = node[part]
            if parts[-1] not in node:
                raise KeyError(f"Unknown config key '{dotted}'")
            node[parts[-1]] = value
        return _build(raw)

    def path(self, relative: str) -> Path:
        """Resolve a project-relative path from the config."""
        return PROJECT_ROOT / relative


def _from_dict(cls: Any, raw: Dict[str, Any]) -> Any:
    kwargs = {}
    for f in fields(cls):
        if f.name not in raw:
            raise KeyError(f"Missing config key '{f.name}' for {cls.__name__}")
        value = raw[f.name]
        ftype = f.type if not isinstance(f.type, str) else _TYPES.get(f.type)
        if ftype is not None and is_dataclass(ftype) and isinstance(value, dict):
            value = _from_dict(ftype, value)
        kwargs[f.name] = value
    return cls(**kwargs)


# Map string annotations (from `from __future__ import annotations`) to classes.
_TYPES = {
    "FactorWeights": FactorWeights,
    "EmailConfig": EmailConfig,
}


def _build(raw: Dict[str, Any]) -> Config:
    raw = copy.deepcopy(raw)
    sections = {
        "data": DataConfig,
        "periods": PeriodConfig,
        "strategy": StrategyConfig,
        "portfolio": PortfolioConfig,
        "risk": RiskConfig,
        "costs": CostConfig,
        "paper_trading": PaperConfig,
        "alerts": AlertConfig,
        "walk_forward": WalkForwardConfig,
        "output": OutputConfig,
    }
    built = {name: _from_dict(cls, raw[name]) for name, cls in sections.items()}
    cfg = Config(**built)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    p, s = cfg.portfolio, cfg.strategy
    if p.sizing_method not in {"equal", "volatility"}:
        raise ValueError("portfolio.sizing_method must be 'equal' or 'volatility'")
    if s.rebalance_frequency not in {"weekly", "biweekly", "monthly", "quarterly"}:
        raise ValueError("strategy.rebalance_frequency must be weekly/biweekly/monthly/quarterly")
    if not 0 < p.max_position_weight <= 1:
        raise ValueError("portfolio.max_position_weight must be in (0, 1]")
    if not 0 <= p.min_cash_weight < 1:
        raise ValueError("portfolio.min_cash_weight must be in [0, 1)")
    if p.max_positions < 1:
        raise ValueError("portfolio.max_positions must be >= 1")
    if cfg.risk.stop_method not in {"volatility", "atr"}:
        raise ValueError("risk.stop_method must be 'volatility' or 'atr'")
    if cfg.paper_trading.fill_model != "next_open":
        raise ValueError("Only the 'next_open' paper fill model is supported")


def load_config(path: Optional[Path] = None) -> Config:
    """Load and validate the YAML configuration."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    return _build(raw)
