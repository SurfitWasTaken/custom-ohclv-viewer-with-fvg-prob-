"""
V2 Survival Model — FVG Fill Probability (Multi-Horizon XGBoost)

Instead of a single scikit-survival model (which is O(n²) and infeasible
at our dataset size), we train one XGBoost binary classifier per horizon:

  - filled_in_1:   P(fill within 1 candle)
  - filled_in_2:   P(fill within 2 candles)
  - filled_in_5:   P(fill within 5 candles)
  - filled_in_10:  P(fill within 10 candles)
  - filled_in_20:  P(fill within 20 candles)
  - filled_in_50:  P(fill within 50 candles)
  - filled_in_100: P(fill within 100 candles)

Each model shares the same features but has its own binary target.
Monotonicity is enforced post-hoc (P(fill_10) >= P(fill_5) etc).

Training time: ~30 seconds total for all 7 models on 8,740 samples.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.fill_feature_engineer import FEATURE_COLS

HORIZONS = [1, 2, 5, 10, 20, 50, 100]


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix from DataFrame."""
    X = df[FEATURE_COLS].values.astype(np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def make_horizon_target(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Binary target: 1 if FVG was filled within `horizon` candles."""
    return ((df['time_to_fill'] <= horizon) & df['filled']).astype(int).values


def train_all_horizons(train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame) -> dict:
    """
    Train one XGBoost classifier per horizon.

    Returns dict with 'models', 'metrics', 'feature_importance'.
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss

    X_train = prepare_features(train_df)
    X_val = prepare_features(val_df)
    X_test = prepare_features(test_df)

    models = {}
    metrics = {}
    all_importances = {}

    print(f"Training {len(HORIZONS)} horizon models...")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print()

    t0 = time.time()

    for h in HORIZONS:
        y_train = make_horizon_target(train_df, h)
        y_val = make_horizon_target(val_df, h)
        y_test = make_horizon_target(test_df, h)

        fill_rate = y_train.mean()

        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        # Predictions
        p_train = model.predict_proba(X_train)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        # Metrics
        try:
            auc_train = roc_auc_score(y_train, p_train)
            auc_val = roc_auc_score(y_val, p_val)
            auc_test = roc_auc_score(y_test, p_test)
        except ValueError:
            auc_train = auc_val = auc_test = 0.0

        brier_test = brier_score_loss(y_test, p_test)

        metrics[h] = {
            'fill_rate': round(float(fill_rate), 4),
            'auc_train': round(float(auc_train), 4),
            'auc_val': round(float(auc_val), 4),
            'auc_test': round(float(auc_test), 4),
            'brier_test': round(float(brier_test), 4),
            'pred_mean_test': round(float(p_test.mean()), 4),
            'actual_mean_test': round(float(y_test.mean()), 4),
        }

        # Feature importance
        imp = model.feature_importances_
        all_importances[h] = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in imp]))

        models[h] = model

        print(f"  Horizon {h:3d}: fill_rate={fill_rate:.3f}  "
              f"AUC={auc_test:.4f}  Brier={brier_test:.4f}  "
              f"pred={p_test.mean():.3f} vs actual={y_test.mean():.3f}")

    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed:.1f}s")

    # Aggregate feature importance across horizons
    avg_importance = {}
    for feat in FEATURE_COLS:
        avg_importance[feat] = round(
            np.mean([all_importances[h][feat] for h in HORIZONS]), 4)
    avg_sorted = sorted(avg_importance.items(), key=lambda x: -x[1])

    print(f"\n  Top 10 features (averaged across horizons):")
    for name, imp in avg_sorted[:10]:
        print(f"    {name:25s} {imp:.4f}")

    return {
        'models': models,
        'metrics': metrics,
        'feature_importance': dict(avg_sorted),
        'per_horizon_importance': all_importances,
        'training_time': round(elapsed, 1),
    }


def predict_fill_probabilities(models: dict, X: np.ndarray,
                                horizons: List[int] = None) -> List[Dict]:
    """
    Predict fill probabilities at each horizon.

    Enforces monotonicity: P(fill_h2) >= P(fill_h1) if h2 > h1.

    Returns list of dicts, one per sample.
    """
    if horizons is None:
        horizons = HORIZONS

    # Get raw predictions
    raw_preds = {}
    for h in horizons:
        if h in models:
            raw_preds[h] = models[h].predict_proba(X)[:, 1]
        else:
            raw_preds[h] = np.zeros(len(X))

    # Enforce monotonicity
    n = len(X)
    results = []
    for i in range(n):
        probs = {}
        prev = 0.0
        for h in sorted(horizons):
            p = float(raw_preds[h][i])
            p = max(p, prev)  # monotonic: can't decrease
            probs[h] = round(p, 4)
            prev = p
        results.append(probs)

    return results


def calibration_analysis(models: dict, test_df: pd.DataFrame) -> Dict:
    """Compare predicted vs actual fill rates at each horizon."""
    X_test = prepare_features(test_df)
    preds = predict_fill_probabilities(models, X_test)

    results = {}
    for h in HORIZONS:
        predicted = np.array([p[h] for p in preds])
        actual = make_horizon_target(test_df, h)
        results[h] = {
            'mean_predicted': round(float(predicted.mean()), 4),
            'actual_fill_rate': round(float(actual.mean()), 4),
            'abs_error': round(abs(float(predicted.mean()) - float(actual.mean())), 4),
        }
    return results


def run_training(data_dir: str = 'data/processed',
                 output_dir: str = 'models') -> dict:
    """Full training pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  FVG FILL PROBABILITY — MULTI-HORIZON TRAINING")
    print("=" * 60)

    train_df = pd.read_parquet(os.path.join(data_dir, 'fill_train.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'fill_val.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'fill_test.parquet'))

    result = train_all_horizons(train_df, val_df, test_df)
    models = result['models']

    # Calibration
    print("\nCalibration (predicted vs actual):")
    cal = calibration_analysis(models, test_df)
    for h, stats in cal.items():
        print(f"  {h:3d}: pred={stats['mean_predicted']:.3f}  "
              f"actual={stats['actual_fill_rate']:.3f}  "
              f"err={stats['abs_error']:.3f}")

    # Save all models in one dict
    joblib.dump(models, os.path.join(output_dir, 'survival_model.pkl'))
    with open(os.path.join(output_dir, 'feature_cols.json'), 'w') as f:
        json.dump(FEATURE_COLS, f, indent=2)

    report = {
        'model_type': 'MultiHorizon_XGBoost',
        'horizons': HORIZONS,
        'metrics': result['metrics'],
        'calibration': {str(k): v for k, v in cal.items()},
        'feature_importance': result['feature_importance'],
        'n_features': len(FEATURE_COLS),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'training_time_seconds': result['training_time'],
    }
    with open(os.path.join(output_dir, 'training_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    size_mb = os.path.getsize(os.path.join(output_dir, 'survival_model.pkl')) / 1e6
    print(f"\nModel saved: {output_dir}/survival_model.pkl ({size_mb:.1f}MB)")
    print("DONE")

    return result


if __name__ == '__main__':
    run_training()
