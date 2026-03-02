"""
Phase 5 — Component 3: Model Monitor
FVG Probability Modeling Project

Lightweight production monitoring:
  - Logs every prediction to a JSON-Lines file
  - Records actual outcomes once resolved
  - Computes rolling accuracy and detects drift
"""

import os
import json
import time
from datetime import datetime
from typing import Optional


class ModelMonitor:
    """
    Track predictions, outcomes, and detect performance drift.

    Parameters
    ----------
    log_path : str
        Path to the JSON-Lines prediction log.
    alert_threshold : float
        If rolling accuracy drops below this, an alert fires.
    rolling_window : int
        Number of recent resolved predictions used for drift check.
    """

    def __init__(self,
                 log_path: str = 'models/phase5/prediction_log.jsonl',
                 alert_threshold: float = 0.55,
                 rolling_window: int = 50):
        self.log_path = log_path
        self.alert_threshold = alert_threshold
        self.rolling_window = rolling_window

        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)

        # In-memory buffer of recent predictions
        self._predictions: list = []
        self._load_existing()

    # ── persistence ─────────────────────────────────────────────────

    def _load_existing(self) -> None:
        """Load prior predictions from the log file, if any."""
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._predictions.append(json.loads(line))

    def _append_log(self, record: dict) -> None:
        """Append a single JSON record to the log file."""
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    # ── core API ────────────────────────────────────────────────────

    def log_prediction(self,
                       timestamp: str,
                       prob_upper: float,
                       bias: str,
                       confidence: float,
                       current_price: float,
                       upper_fvg_mid: float,
                       lower_fvg_mid: float,
                       target: Optional[float] = None,
                       stop_loss: Optional[float] = None) -> dict:
        """
        Record a new prediction.

        Returns the record dict (including a unique prediction_id).
        """
        record = {
            'prediction_id': f"pred_{int(time.time()*1000)}",
            'timestamp': str(timestamp),
            'prob_upper': round(prob_upper, 6),
            'bias': bias,
            'confidence': round(confidence, 6),
            'current_price': current_price,
            'upper_fvg_mid': upper_fvg_mid,
            'lower_fvg_mid': lower_fvg_mid,
            'target': target,
            'stop_loss': stop_loss,
            'outcome': None,          # filled later
            'outcome_time': None,
            'correct': None,
            'logged_at': datetime.now(tz=__import__('datetime').timezone.utc).isoformat(),
        }
        self._predictions.append(record)
        self._append_log(record)
        return record

    def log_outcome(self,
                    prediction_id: str,
                    outcome: str) -> Optional[dict]:
        """
        Record the actual outcome for a prior prediction.

        Parameters
        ----------
        prediction_id : str
        outcome : 'upper' | 'lower' | 'neither'

        Returns the updated record, or None if not found.
        """
        for rec in reversed(self._predictions):
            if rec['prediction_id'] == prediction_id:
                rec['outcome'] = outcome
                rec['outcome_time'] = datetime.now(tz=__import__('datetime').timezone.utc).isoformat()

                # Was prediction correct?
                if outcome in ('upper', 'lower'):
                    predicted_upper = rec['prob_upper'] > 0.5
                    actual_upper = outcome == 'upper'
                    rec['correct'] = predicted_upper == actual_upper
                else:
                    rec['correct'] = None  # ambiguous

                # Rewrite log (simple approach for small logs)
                self._rewrite_log()
                self._check_drift()
                return rec

        return None

    # ── drift detection ─────────────────────────────────────────────

    def _check_drift(self) -> Optional[str]:
        """
        Check if rolling accuracy has dropped below the threshold.

        Returns an alert message if drifting, else None.
        """
        resolved = [p for p in self._predictions
                    if p.get('correct') is not None]
        if len(resolved) < self.rolling_window:
            return None

        recent = resolved[-self.rolling_window:]
        accuracy = sum(1 for p in recent if p['correct']) / len(recent)

        if accuracy < self.alert_threshold:
            msg = (f"⚠ DRIFT ALERT: rolling accuracy "
                   f"{accuracy:.1%} < {self.alert_threshold:.0%} "
                   f"(last {self.rolling_window} predictions)")
            print(msg)
            return msg
        return None

    # ── reporting ───────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Return a summary of prediction performance."""
        total = len(self._predictions)
        resolved = [p for p in self._predictions
                    if p.get('correct') is not None]
        correct = sum(1 for p in resolved if p['correct'])

        return {
            'total_predictions': total,
            'resolved': len(resolved),
            'pending': total - len(resolved),
            'accuracy': correct / len(resolved) if resolved else None,
            'log_path': self.log_path,
        }

    # ── internal ────────────────────────────────────────────────────

    def _rewrite_log(self) -> None:
        """Rewrite the full log to disk (call sparingly)."""
        with open(self.log_path, 'w') as f:
            for rec in self._predictions:
                f.write(json.dumps(rec, default=str) + '\n')
