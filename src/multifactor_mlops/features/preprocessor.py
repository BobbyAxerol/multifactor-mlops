"""
Feature Preprocessor with STRICT train-only statistics.

- clip bounds: 1%/99% quantiles computed on the TRAINING sample only
- missing/implicit values filled with TRAIN medians
- serializable to/from dict so the exact same transform is applied in
  inner validation, OOF, outer OOS and live serving (train-serving parity).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class FeaturePreprocessor:
    def __init__(self):
        self.feature_names_: Optional[List[str]] = None
        self.clip_low_: Optional[pd.Series] = None
        self.clip_high_: Optional[pd.Series] = None
        self.medians_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        X = X.replace([np.inf, -np.inf], np.nan)
        self.feature_names_ = list(X.columns)
        self.clip_low_ = X.quantile(0.01)
        self.clip_high_ = X.quantile(0.99)
        self.medians_ = X.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise ValueError("FeaturePreprocessor must be fit() before transform().")
        X = X[self.feature_names_].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.clip(lower=self.clip_low_, upper=self.clip_high_, axis=1)
        X = X.fillna(self.medians_)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        if self.feature_names_ is None:
            raise ValueError("FeaturePreprocessor must be fit() before serialization.")
        return {
            "feature_names": self.feature_names_,
            "clip_low": self.clip_low_.to_dict(),
            "clip_high": self.clip_high_.to_dict(),
            "medians": self.medians_.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeaturePreprocessor":
        pp = cls()
        pp.feature_names_ = list(data["feature_names"])
        pp.clip_low_ = pd.Series(data["clip_low"])
        pp.clip_high_ = pd.Series(data["clip_high"])
        pp.medians_ = pd.Series(data["medians"])
        return pp
