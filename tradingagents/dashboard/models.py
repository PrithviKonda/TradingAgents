"""Neural signals to set beside the agents' ratings: one trained here, one pretrained.

- **Walk-forward MLP.** A small feed-forward network (scikit-learn) learns the
  sign of the N-session forward return from technical features. It is refit on
  an expanding window and only ever trained on rows whose label was already
  known when it predicts, so every probability it reports is out of sample. A
  logistic regression on the same features is offered as the baseline any
  network has to beat.
- **Chronos-Bolt.** Amazon's pretrained time-series transformer forecasts the
  price path zero-shot, as quantile bands. Its implied probability of a higher
  price is scored the same way, over a rolling-origin backtest, so it earns
  trust on this instrument rather than being assumed to have it.

Anything with ``fit(X, y)`` and ``predict_proba(X)`` can stand in for the MLP.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import warnings
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from tradingagents.dashboard.analytics import direction

CHRONOS_MODEL = "amazon/chronos-bolt-small"
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class SignalModel(Protocol):
    def fit(self, X, y): ...

    def predict_proba(self, X): ...


# -- features --------------------------------------------------------------------------

def technical_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Scale-free features of daily OHLCV, each computed from data through its own row."""
    close, high, low, volume = bars["Close"], bars["High"], bars["Low"], bars["Volume"]
    out = pd.DataFrame(index=bars.index)
    for n in (1, 5, 10, 21, 63):
        out[f"ret_{n}"] = close.pct_change(n)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    mid, std = close.rolling(20).mean(), close.rolling(20).std()
    out["bb_pct_b"] = (close - (mid - 2 * std)) / (4 * std)
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    out["atr_14"] = true_range.rolling(14).mean() / close
    out["vol_21"] = close.pct_change().rolling(21).std() * math.sqrt(252)
    volume_std = volume.rolling(20).std()
    volume_z = (volume - volume.rolling(20).mean()) / volume_std.where(volume_std > 0)
    # Spot FX and some indices report no volume (zero, constant or missing):
    # there is nothing to read in it, so the feature is neutral rather than
    # undefined, which would drop every row. The 20-session warm-up stays NaN.
    out["volume_z"] = volume_z.mask(volume_z.isna() & (np.arange(len(bars)) >= 19), 0.0)
    out["dist_sma50"] = close / close.rolling(50).mean() - 1
    out["dist_sma200"] = close / close.rolling(200).mean() - 1
    return out.replace([np.inf, -np.inf], np.nan)


def forward_return(bars: pd.DataFrame, horizon: int) -> pd.Series:
    return bars["Close"].shift(-horizon) / bars["Close"] - 1


def make_model(kind: str = "mlp", seed: int = 0) -> SignalModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "mlp":
        # Daily returns are mostly noise, so the network is regularized hard and
        # stopped early: with weak weight decay it fit the training window and
        # reported 90%+ probabilities that scored worse than the base rate.
        # Early stopping holds out a random 20% of the training rows only; the
        # out-of-sample rows are never seen.
        net = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1.0, early_stopping=True,
                            validation_fraction=0.2, n_iter_no_change=10, max_iter=300,
                            random_state=seed)
        return make_pipeline(StandardScaler(), net)
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    raise ValueError(f"unknown model {kind!r}; choose mlp or logistic")


# -- walk-forward evaluation -----------------------------------------------------------

@dataclass
class WalkForward:
    probability: pd.Series            # out-of-sample P(up) per session
    label: pd.Series                  # 1 if the forward return was positive, NaN if not yet known
    forward: pd.Series                # the forward return itself
    horizon: int
    model: str
    metrics: dict = field(default_factory=dict)
    calibration: pd.DataFrame | None = None
    importance: pd.DataFrame | None = None

    @property
    def latest(self) -> tuple[pd.Timestamp, float]:
        """The newest out-of-sample probability: the model's current call."""
        return self.probability.index[-1], float(self.probability.iloc[-1])


def walk_forward(bars: pd.DataFrame, horizon: int = 5, kind: str = "mlp", min_train: int = 252,
                 step: int = 21, seed: int = 0, importance: bool = True,
                 model_factory=None) -> WalkForward:
    """Expanding-window, out-of-sample P(forward return > 0) for every session after ``min_train``.

    Predicting session ``t`` uses a model fit only on rows ``i`` with ``i + horizon < t0``,
    where ``t0`` starts ``t``'s block: their labels had all traded by then.
    """
    from sklearn.exceptions import ConvergenceWarning

    factory = model_factory or (lambda: make_model(kind, seed))
    features = technical_features(bars)
    forward = forward_return(bars, horizon)
    data = features.assign(_fwd=forward).dropna(subset=list(features.columns))
    X, fwd = data[features.columns], data["_fwd"]
    label = (fwd > 0).astype(float).where(fwd.notna())
    need = min_train + horizon + step
    if len(X) < need:
        # Say whether the history is short or a feature is undefined over most of it.
        sparse = [c for c in features.columns if features[c].notna().sum() < need]
        detail = (f"; no value often enough for {', '.join(sparse)}"
                  if sparse and len(bars) >= need else "")
        raise ValueError(f"need at least {need} sessions with every feature defined, "
                         f"have {len(X)} of {len(bars)} sessions{detail}")

    probability = pd.Series(np.nan, index=X.index, name="p_up")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for t0 in range(min_train + horizon, len(X), step):
            train = slice(0, t0 - horizon)
            known = label.iloc[train].notna()
            y_train = label.iloc[train][known]
            if y_train.nunique() < 2:
                continue
            model = factory()
            model.fit(X.iloc[train][known].to_numpy(), y_train.to_numpy().astype(int))
            block = X.iloc[t0:t0 + step]
            probability.iloc[t0:t0 + step] = model.predict_proba(block.to_numpy())[:, 1]

    probability = probability.dropna()
    result = WalkForward(probability, label.reindex(probability.index), fwd.reindex(probability.index),
                         horizon, kind)
    result.metrics = classification_metrics(probability, result.label, result.forward)
    result.calibration = calibration_table(probability, result.label)
    if importance:
        result.importance = permutation_importance(X, label, horizon, factory)
    return result


def classification_metrics(probability: pd.Series, label: pd.Series, forward: pd.Series | None = None) -> dict:
    """Out-of-sample accuracy, AUC, Brier and log loss, and what the probability was worth."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    known = label.notna() & probability.notna()
    p, y = probability[known].clip(1e-6, 1 - 1e-6), label[known].astype(int)
    if len(y) == 0:
        return {"n": 0}
    metrics = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "accuracy": float(((p > 0.5).astype(int) == y).mean()),
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        # The Brier score of always forecasting the base rate: the bar to clear.
        "brier_baseline": float(y.mean() * (1 - y.mean())),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    if forward is not None:
        f = forward[known]
        metrics["mean_return_when_up"] = float(f[p > 0.5].mean()) if (p > 0.5).any() else float("nan")
        metrics["mean_return_when_down"] = float(f[p <= 0.5].mean()) if (p <= 0.5).any() else float("nan")
        ranked = pd.concat([p, f], axis=1).rank()
        metrics["ic"] = float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1]))
    return metrics


def calibration_table(probability: pd.Series, label: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability against observed frequency, by probability bin."""
    known = label.notna()
    frame = pd.DataFrame({"p": probability[known], "y": label[known]})
    if frame.empty:
        return pd.DataFrame(columns=["predicted", "observed", "count"])
    frame["bin"] = pd.cut(frame["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    table = frame.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"),
                                                    count=("y", "size"))
    return table.reset_index(drop=True)


def permutation_importance(X: pd.DataFrame, label: pd.Series, horizon: int, factory,
                           holdout: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """Drop in out-of-sample AUC when each feature is shuffled, on the latest ``holdout`` share."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.inspection import permutation_importance as sk_importance

    known = label.notna()
    Xk, yk = X[known], label[known].astype(int)
    split = int(len(Xk) * (1 - holdout))
    train_end = max(split - horizon, 0)
    if train_end < 50 or yk.iloc[:train_end].nunique() < 2 or yk.iloc[split:].nunique() < 2:
        return pd.DataFrame(columns=["feature", "importance", "std"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = factory()
        model.fit(Xk.iloc[:train_end].to_numpy(), yk.iloc[:train_end].to_numpy())
        result = sk_importance(model, Xk.iloc[split:].to_numpy(), yk.iloc[split:].to_numpy(),
                               scoring="roc_auc", n_repeats=5, random_state=seed)
    return (pd.DataFrame({"feature": X.columns, "importance": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def agreement(fr: pd.DataFrame, probabilities: dict[str, pd.Series], horizon: int = 5,
              excess: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each directional decision beside the model's probability as of its date.

    Returns the per-decision table and a summary: how decisions fared when the
    model agreed with their direction, and when it did not.
    """
    column = f"excess_{horizon}" if excess else f"fwd_{horizon}"
    rows = []
    for d in fr.itertuples(index=False):
        sign = direction(d.rating)
        series = probabilities.get(d.ticker)
        if not sign or series is None or series.empty:
            continue
        known = series[series.index <= d.date]
        if known.empty:
            continue
        p = float(known.iloc[-1])
        outcome = getattr(d, column, np.nan)
        rows.append({"date": d.date, "ticker": d.ticker, "rating": d.rating, "model_p_up": p,
                     "agrees": (p > 0.5) == (sign > 0), "outcome": outcome,
                     "right": bool(outcome * sign > 0) if pd.notna(outcome) else np.nan})
    table = pd.DataFrame(rows, columns=["date", "ticker", "rating", "model_p_up", "agrees", "outcome", "right"])
    if table.empty:
        return table, pd.DataFrame(columns=["group", "count", "hit_rate", "mean_signed_outcome"])
    summary = []
    for name, group in (("model agrees", table[table["agrees"]]), ("model disagrees", table[~table["agrees"]])):
        scored = group.dropna(subset=["outcome"])
        signed = scored["outcome"] * scored["rating"].map(direction)
        summary.append({"group": name, "count": len(group),
                        "hit_rate": float(scored["right"].astype(float).mean()) if len(scored) else np.nan,
                        "mean_signed_outcome": float(signed.mean()) if len(scored) else np.nan})
    return table, pd.DataFrame(summary)


# -- Chronos ---------------------------------------------------------------------------

def chronos_available() -> bool:
    """Whether Chronos and torch are installed, found without importing torch on each page load."""
    return all(importlib.util.find_spec(name) is not None for name in ("chronos", "torch"))


@functools.lru_cache(maxsize=2)
def _pipeline(model_id: str):
    import torch
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)


def probability_above(quantiles: dict[float, float], price: float) -> float:
    """P(value > ``price``) from a forecast's quantiles, interpolating its CDF.

    Outside the outermost quantiles the CDF is held at their level, so the
    answer never claims more confidence than the forecast's own bands.
    """
    levels = np.array(sorted(quantiles))
    values = np.maximum.accumulate(np.array([quantiles[q] for q in levels]))
    cdf = float(np.interp(price, values, levels, left=levels[0], right=levels[-1]))
    return 1.0 - cdf


@dataclass
class Forecast:
    bands: pd.DataFrame        # one row per future session: q10..q90 and mean
    prob_up: float             # P(close at the horizon > last close)
    expected_return: float     # median at the horizon against the last close


def chronos_forecast(closes: pd.Series, horizon: int = 20, context: int = 512,
                     model_id: str = CHRONOS_MODEL, pipeline=None) -> Forecast:
    """Zero-shot quantile forecast of the next ``horizon`` closes."""
    import torch

    pipe = pipeline or _pipeline(model_id)
    history = closes.dropna().astype(float)
    window = torch.tensor(history.to_numpy()[-context:], dtype=torch.float32)
    q, mean = pipe.predict_quantiles(inputs=window, prediction_length=horizon, quantile_levels=list(QUANTILES))
    dates = pd.bdate_range(history.index[-1] + pd.offsets.BDay(1), periods=horizon)
    bands = pd.DataFrame(np.asarray(q[0]), index=dates, columns=[f"q{round(lv * 100)}" for lv in QUANTILES])
    bands["mean"] = np.asarray(mean[0])
    last = float(history.iloc[-1])
    final = {lv: float(bands[f"q{round(lv * 100)}"].iloc[-1]) for lv in QUANTILES}
    return Forecast(bands, probability_above(final, last), final[0.5] / last - 1)


def chronos_backtest(closes: pd.Series, horizon: int = 5, origins: int = 36, spacing: int = 10,
                     context: int = 512, model_id: str = CHRONOS_MODEL, pipeline=None) -> pd.DataFrame:
    """Rolling-origin record: at each past origin, the forecast made then against what happened.

    Each forecast sees only the closes up to its origin. Origins are ``spacing``
    sessions apart, the latest one ``horizon`` sessions before the last close so
    every forecast can be scored.
    """
    import torch

    pipe = pipeline or _pipeline(model_id)
    history = closes.dropna().astype(float)
    values = history.to_numpy()
    ends = [len(values) - horizon - k * spacing for k in range(origins)]
    ends = sorted(e for e in ends if e >= 64)
    if not ends:
        raise ValueError("not enough history for a Chronos backtest")
    inputs = [torch.tensor(values[max(0, e - context):e], dtype=torch.float32) for e in ends]
    q, _ = pipe.predict_quantiles(inputs=inputs, prediction_length=horizon, quantile_levels=list(QUANTILES))
    q = np.asarray(q)
    rows = []
    for i, e in enumerate(ends):
        last = values[e - 1]
        final = {lv: float(q[i, -1, j]) for j, lv in enumerate(QUANTILES)}
        actual = values[e - 1 + horizon] / last - 1
        rows.append({"origin": history.index[e - 1], "prob_up": probability_above(final, last),
                     "predicted_return": final[0.5] / last - 1, "actual_return": actual,
                     "q10_return": final[0.1] / last - 1, "q90_return": final[0.9] / last - 1})
    frame = pd.DataFrame(rows)
    frame["label"] = (frame["actual_return"] > 0).astype(float)
    frame["inside_80"] = frame["actual_return"].between(frame["q10_return"], frame["q90_return"])
    return frame
