# File: Red/regen/ascend_auto.py

import numpy as np

class AutoScaler:
    """
    Monitors training signals and automatically adjusts AURASCEND width/depth gains.
    """

    def __init__(
            self,
            ascend_hybrid,
            sensitivity=0.5,
            min_gain=0.0,
            max_gain=1.0,
            min_history=5,
    ):
        self.engine = ascend_hybrid
        self.sensitivity = sensitivity

        self.min_gain = min_gain
        self.max_gain = max_gain
        self.min_history = min_history

        self.loss_history = []
        self.perplexity_history = []
        self.reasoning_scores = []
        self.dialogue_scores = []

    def _trend(self, history, window=5):
        """Compute slope: negative slope = improving."""
        if len(history) < window:
            return 0.0
        y = np.array(history[-window:], dtype=float)
        x = np.arange(len(y), dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def update_metrics(self, loss, perplexity, reasoning_score, dialogue_score):
        self.loss_history.append(float(loss))
        self.perplexity_history.append(float(perplexity))
        self.reasoning_scores.append(float(reasoning_score))
        self.dialogue_scores.append(float(dialogue_score))

    def adjust(self):
        """
        Decide scaling direction based on signal balance.
        Returns: (new_arch, score) or (current_arch, 0.0) if not enough history.
        """
        if len(self.loss_history) < self.min_history:
            # Not enough signal yet, no scaling
            return self.engine.compute_new_architecture(), 0.0

        loss_trend = -self._trend(self.loss_history)
        ppx_trend = -self._trend(self.perplexity_history)
        reason_trend = self._trend(self.reasoning_scores)
        dialog_trend = self._trend(self.dialogue_scores)

        score = (loss_trend + ppx_trend + reason_trend + dialog_trend) / 4.0

        if not np.isfinite(score):
            # numerical safeguard
            score = 0.0

        gain_change = score * self.sensitivity

        new_width = np.clip(self.engine.width_gain + gain_change, self.min_gain, self.max_gain)
        new_depth = np.clip(self.engine.depth_gain + gain_change, self.min_gain, self.max_gain)

        self.engine.set_scaling(float(new_width), float(new_depth))

        return self.engine.compute_new_architecture(), float(score)
