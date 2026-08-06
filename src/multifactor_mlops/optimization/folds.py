"""
Purged Expanding Fold Builder for ML tuning and OOF prediction generation.

Folds tile the development window [start_date, end_date) with no overlap and no gap.
Purge is applied by label_end_time < fold.test_start (canonical purge).
"""

import pandas as pd
from typing import List, Tuple, Optional

FREQUENCY_OFFSETS = {
    "yearly": pd.DateOffset(years=1),
    "semi_yearly": pd.DateOffset(months=6),
    "quarterly": pd.DateOffset(months=3),
    "monthly": pd.DateOffset(months=1),
}


class PurgedExpandingFoldBuilder:
    def __init__(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2023-12-31",
        frequency: str = "quarterly",
    ):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        if frequency not in FREQUENCY_OFFSETS:
            raise ValueError(f"Unsupported fold frequency: {frequency}")
        self.frequency = frequency

    def build_folds(self) -> List[dict]:
        """
        Returns list of fold dicts:
          {"fold_id": int, "test_start": Timestamp, "test_end": Timestamp}
        Train is implicitly everything with label_end_time < test_start.
        """
        folds = []
        step = FREQUENCY_OFFSETS[self.frequency]
        test_start = self.start_date
        fold_id = 0
        while test_start < self.end_date:
            test_end = test_start + step
            folds.append({
                "fold_id": fold_id,
                "test_start": test_start,
                "test_end": min(test_end, self.end_date),
            })
            fold_id += 1
            test_start = test_end
        return folds
