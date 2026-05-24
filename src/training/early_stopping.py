"""Early stopping utility."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track validation improvement and signal when training should stop."""

    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")
        self.best_score: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Update state and return True when score improves."""

        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        self.should_stop = self.counter >= self.patience
        return False
