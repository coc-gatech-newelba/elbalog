"""PCA-based learning model for anomaly detection."""


import numpy as np
from sklearn.decomposition import PCA as skPCA
from sklearn.svm import SVC


class PCA:
    def __init__(self):
        """Initialize a PCA-based learning model."""
        self._P = None
        self._svm = None

    # This empty method is needed to train the Ensemble model.
    def get_params(self, deep=True):
        """Return parameters for this estimator."""
        return {}

    def fit(self, X, y):
        """Fit the PCA-based learning model to the training data.

        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        """
        pca = skPCA().fit(X)
        self._P = []
        for accumulate_variance in [0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95, 0.99]:
            accumulated_variance = 0.0
            k = 0
            while accumulated_variance < accumulate_variance:
                accumulated_variance += pca.explained_variance_ratio_[k]
                k += 1
            self._P.append(np.transpose(pca.components_[k:]))
        self._svm = SVC(gamma="auto").fit(
            [[np.linalg.norm(np.dot(x, p)) for p in self._P] for x in X], y
        )

    def predict(self, X):
        """Return predicted labels for the test data.

        X -- [array of shape (n_samples, n_features)] Test data.
        """
        return self._svm.predict([[np.linalg.norm(np.dot(x, p)) for p in self._P] for x in X])
