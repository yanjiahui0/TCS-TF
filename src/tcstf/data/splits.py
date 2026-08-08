from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SliceSet:
    train: slice
    validation: slice
    recalibration: slice
    test: slice


@dataclass(frozen=True)
class PurgedChronologicalSplit:
    """60/15/10/15 chronological split with a purge gap between intervals.

    The default proportions match the manuscript. The purge is specified in raw
    records; for rolling-origin time-series use at least one complete forecast
    horizon.
    """

    train_fraction: float = 0.60
    validation_fraction: float = 0.15
    recalibration_fraction: float = 0.10
    test_fraction: float = 0.15
    purge: int = 0

    def split(self, n: int) -> SliceSet:
        fracs = [
            self.train_fraction,
            self.validation_fraction,
            self.recalibration_fraction,
            self.test_fraction,
        ]
        if abs(sum(fracs) - 1.0) > 1e-8:
            raise ValueError("Split fractions must sum to one")
        effective = n - 3 * self.purge
        if effective <= 4:
            raise ValueError("Not enough observations after purge gaps")
        n_train = int(effective * self.train_fraction)
        n_val = int(effective * self.validation_fraction)
        n_cal = int(effective * self.recalibration_fraction)
        n_test = effective - n_train - n_val - n_cal

        a = 0
        train = slice(a, a + n_train)
        a = train.stop + self.purge
        validation = slice(a, a + n_val)
        a = validation.stop + self.purge
        recalibration = slice(a, a + n_cal)
        a = recalibration.stop + self.purge
        test = slice(a, a + n_test)
        return SliceSet(train, validation, recalibration, test)
