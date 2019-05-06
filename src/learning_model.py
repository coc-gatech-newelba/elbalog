"""Abstract learning model for anomaly detection."""


from sklearn.metrics import precision_recall_fscore_support


class LearningModel:
    """Abstract learning model."""

    def __init__(self, model):
        """Initialize a learning model.

        model -- [object] Learning model.
        """
        self._model = model

    def fit(self, *args, **kwargs):
        """Fit the learning model to the training data."""
        raise RuntimeError("Not implemented.")

    def predict(self, *args, **kwargs):
        """Return predicted labels for the test data."""
        raise RuntimeError("Not implemented.")

    def precision_recall_fscore(self, X, y):
        """Return a 3-tuple where the first element is the precision of the model, the second is the
        recall, and the third is the F-measure.

        For statistical learning models, the test data is represented by an array of dimension
        (n_samples, n_features); for sequence-based learning models, the test data is represented by
        an array with n_samples sequences of possibly variable length.
        
        X -- [array] Test data.
        y -- [array] Target values for the test data.
        """
        y_pred = self.predict(X)
        return tuple(precision_recall_fscore_support(y, y_pred, average="binary")[0:3])
