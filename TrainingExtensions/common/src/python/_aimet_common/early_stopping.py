# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Early stopping utility shared across optimization loops (e.g. AdaScale)."""

from dataclasses import dataclass
from typing import Optional, Union

from .utils import AimetLogger

__all__ = [
    "_EarlyStoppingConfig",
    "_create_early_stopping",
]


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)


@dataclass
class _EarlyStoppingConfig:
    """Tunable parameters for :class:`_EarlyStopping`.

    check_interval: number of observations between successive improvement checks.
    rel_threshold: minimum relative improvement over the previous window to keep going.
    window: number of most recent observations averaged at each check.
    """

    check_interval: int = 200
    rel_threshold: float = 1e-2
    window: int = 500


class _EarlyStopping:
    """Early stopping based on the relative improvement of a windowed average loss.

    Every ``check_interval`` calls, the average loss over the most recent ``window``
    observations is compared against the previous window's average. If the relative
    improvement drops below ``rel_threshold``, ``early_stop`` is set to ``True``.
    """

    def __init__(self, config: Optional[_EarlyStoppingConfig] = None):
        config = config or _EarlyStoppingConfig()
        self.check_interval = config.check_interval
        self.rel_threshold = config.rel_threshold
        self.window = config.window

        self.early_stop = False
        self._loss_history = []
        self._window_avgs = []  # windowed average loss recorded at each check
        self._step = 0

    def __call__(self, loss: float) -> bool:
        """Record ``loss`` and update ``early_stop``. Returns the current ``early_stop`` flag."""
        self._step += 1
        self._loss_history.append(loss)

        if self._step % self.check_interval != 0:
            return self.early_stop

        window_size = min(self.window, len(self._loss_history))
        window_avg = sum(self._loss_history[-window_size:]) / window_size
        self._window_avgs.append(window_avg)

        # Relative improvement check over the window
        if self._step > window_size:
            prev_window_avg = self._window_avgs[-2]
            rel_improvement = (prev_window_avg - window_avg) / (
                abs(prev_window_avg) + 1e-8
            )
            if rel_improvement < self.rel_threshold:
                _logger.info(
                    "Early stopping: relative improvement %.6f < threshold %s "
                    "over window %d at step %d",
                    rel_improvement,
                    self.rel_threshold,
                    self.window,
                    self._step,
                )
                self.early_stop = True

        return self.early_stop


def _create_early_stopping(
    flag: Union[bool, _EarlyStoppingConfig, None],
) -> Optional[_EarlyStopping]:
    """Build an :class:`_EarlyStopping` from a flag value, or ``None`` if disabled.

    ``None``/``False`` disables it, ``True`` enables it with default parameters, and
    an :class:`_EarlyStoppingConfig` enables it with those parameters.
    """
    if not flag:
        return None
    config = flag if isinstance(flag, _EarlyStoppingConfig) else None
    return _EarlyStopping(config)
