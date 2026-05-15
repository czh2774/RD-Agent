import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from rdagent.scenarios.qlib.ashare_semantics import (
    QLIB_ASHARE_BANDIT_DERIVED_UTILITY_NAME,
    QLIB_ASHARE_BANDIT_METRIC_INVALID_FAILURE,
    QLIB_ASHARE_BANDIT_METRIC_MISSING_FAILURE,
    QLIB_ASHARE_BANDIT_REWARD_RULE,
    QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS,
    QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS,
)


class QlibAshareBanditMetricError(ValueError):
    """Raised when Qlib A-share bandit feedback metrics are absent or invalid."""


@dataclass
class Metrics:
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    rank_icir: float = 0.0
    arr: float = 0.0
    ir: float = 0.0
    mdd: float = 0.0
    drawdown_adjusted_return: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.ic,
                self.icir,
                self.rank_ic,
                self.rank_icir,
                self.arr,
                self.ir,
                -self.mdd,
                self.drawdown_adjusted_return,
            ]
        )


def extract_metrics_from_experiment(experiment) -> Metrics:
    """Extract Qlib-declared A-share bandit metrics from experiment feedback."""

    try:
        result = experiment.result
    except AttributeError as exc:
        raise QlibAshareBanditMetricError(f"{QLIB_ASHARE_BANDIT_METRIC_MISSING_FAILURE}: experiment.result") from exc

    ic = _required_numeric_metric(result, QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS[0])
    icir = _required_numeric_metric(result, QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS[1])
    rank_ic = _required_numeric_metric(result, QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS[2])
    rank_icir = _required_numeric_metric(result, QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS[3])
    arr = _required_numeric_metric(result, QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS[0])
    ir = _required_numeric_metric(result, QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS[1])
    mdd = _required_numeric_metric(result, QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS[2])
    drawdown_adjusted_return = arr / abs(mdd) if mdd != 0 else 0.0

    return Metrics(
        ic=ic,
        icir=icir,
        rank_ic=rank_ic,
        rank_icir=rank_icir,
        arr=arr,
        ir=ir,
        mdd=mdd,
        **{QLIB_ASHARE_BANDIT_DERIVED_UTILITY_NAME: drawdown_adjusted_return},
    )


def _required_numeric_metric(result: object, metric_path: str) -> float:
    try:
        contains_metric = metric_path in result
    except TypeError as exc:
        raise QlibAshareBanditMetricError(
            f"{QLIB_ASHARE_BANDIT_METRIC_MISSING_FAILURE}: result does not expose metric-path membership"
        ) from exc
    if not contains_metric:
        raise QlibAshareBanditMetricError(f"{QLIB_ASHARE_BANDIT_METRIC_MISSING_FAILURE}: {metric_path}")
    try:
        metric_value = float(result[metric_path])
    except (KeyError, TypeError, ValueError) as exc:
        raise QlibAshareBanditMetricError(f"{QLIB_ASHARE_BANDIT_METRIC_INVALID_FAILURE}: {metric_path}") from exc
    if not math.isfinite(metric_value):
        raise QlibAshareBanditMetricError(f"{QLIB_ASHARE_BANDIT_METRIC_INVALID_FAILURE}: {metric_path}")
    return metric_value


class LinearThompsonTwoArm:
    def __init__(self, dim: int, prior_var: float = 1.0, noise_var: float = 1.0):
        self.dim = dim
        self.noise_var = noise_var
        # Each arm has its own posterior: mean & inverse of covariance (precision matrix)
        self.mean = {
            "factor": np.zeros(dim),
            "model": np.zeros(dim),
        }
        self.precision = {
            "factor": np.eye(dim) / prior_var,
            "model": np.eye(dim) / prior_var,
        }

    def sample_reward(self, arm: str, x: np.ndarray) -> float:
        P = self.precision[arm]
        P = 0.5 * (P + P.T)

        eps = 1e-6
        try:
            cov = np.linalg.inv(P + eps * np.eye(self.dim))
            L = np.linalg.cholesky(cov)
            z = np.random.randn(self.dim)
            w_sample = self.mean[arm] + L @ z
        except np.linalg.LinAlgError:
            w_sample = self.mean[arm]

        return float(np.dot(w_sample, x))

    def update(self, arm: str, x: np.ndarray, r: float) -> None:
        P = self.precision[arm]
        P += np.outer(x, x) / self.noise_var
        self.precision[arm] = P
        self.mean[arm] = np.linalg.solve(P, P @ self.mean[arm] + (r / self.noise_var) * x)

    def next_arm(self, x: np.ndarray) -> str:
        scores = {arm: self.sample_reward(arm, x) for arm in ("factor", "model")}
        return max(scores, key=scores.get)


class EnvController:
    def __init__(self, weights: Tuple[float, ...] = None) -> None:
        if weights is not None:
            raise QlibAshareBanditMetricError(QLIB_ASHARE_BANDIT_REWARD_RULE)
        self.bandit = LinearThompsonTwoArm(dim=8, prior_var=10.0, noise_var=0.5)

    def reward(self, m: Metrics) -> float:
        return float(m.drawdown_adjusted_return)

    def decide(self, m: Metrics) -> str:
        x = m.as_vector()
        return self.bandit.next_arm(x)

    def record(self, m: Metrics, arm: str) -> None:
        r = self.reward(m)
        self.bandit.update(arm, m.as_vector(), r)
