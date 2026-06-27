"""Адаптер 3-class sentiment → индексы рейтинга для API (1/3/5)."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ThreeClassToRatingEstimator(BaseEstimator, ClassifierMixin):
    """3-class sentiment -> индексы рейтинга 0/2/4 (рейтинги 1/3/5 для API)."""

    RATING_IDX = np.array([0, 2, 4])

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        self.classes_ = self.RATING_IDX
        return self

    def predict(self, X):
        pred3 = self.base_estimator.predict(X)
        return self.RATING_IDX[pred3.astype(int)]

    def predict_proba(self, X):
        proba3 = self.base_estimator.predict_proba(X)
        out = np.zeros((proba3.shape[0], 5))
        out[:, 0] = proba3[:, 0]
        out[:, 2] = proba3[:, 1]
        out[:, 4] = proba3[:, 2]
        return out
