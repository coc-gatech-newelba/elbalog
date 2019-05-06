"""Statistical learning models for anomaly detection."""


import click
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from learning_data import LearningData
from learning_model import LearningModel
from pca import PCA


class StatisticalLearningModel(LearningModel):
    """Statistical learning model."""

    def fit(self, X, y):
        """Fit the statistical learning model to the training data.

        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        """
        self._model.fit(X, y)

    def predict(self, X):
        """Return predicted labels for the test data.

        X -- [array of shape (n_samples, n_features)] Test data.
        """
        return self._model.predict(X)


class DecisionTreeLearningModel(StatisticalLearningModel):
    """Decision tree learning model wrapper."""

    def __init__(self):
        """Initialize a decision tree learning model wrapper."""
        super().__init__(DecisionTreeClassifier())


class LogisticRegressionLearningModel(StatisticalLearningModel):
    """Logistic regression learning model wrapper."""

    def __init__(self):
        """Initialize a logistic regression learning model wrapper."""
        super().__init__(LogisticRegression(solver="lbfgs", max_iter=100000))


class SVMLearningModel(StatisticalLearningModel):
    """SVM learning model wrapper."""

    def __init__(self):
        """Initialize a SVM learning model wrapper."""
        super().__init__(SVC(gamma="auto"))


class PCABasedLearningModel(StatisticalLearningModel):
    """PCA-based learning model wrapper."""

    def __init__(self):
        """Initialize a PCA-based learning model wrapper."""
        super().__init__(PCA())


class EnsembleLearningModel(StatisticalLearningModel):
    """Ensemble learning model."""

    def __init__(self):
        """Initialize a learning model that combines the results from decision tree, PCA-based, and
        logistic regression learning models.""" 
        super().__init__(VotingClassifier(estimators=[
            ("dt", DecisionTreeLearningModel()._model),
            ("pca", PCABasedLearningModel()._model),
            ("lr", LogisticRegressionLearningModel()._model)
        ], voting="hard"))


@click.group()
def main():
    pass


@main.command()
@click.argument("dirpath", metavar="<dirpath>")
@click.argument("test_size", metavar="<test_size>", type=float)
@click.argument("vectorizer_type", metavar="<vectorizer_type>")
def decision_tree(dirpath, test_size, vectorizer_type):
    "Train a decision tree learning model to classify system/application executions as normal or " \
    "abnormal."
    ld = LearningData(dirpath, test_size, vectorizer_type)
    dt = DecisionTreeLearningModel()
    dt.fit(ld.X_train(), ld.y_train())
    precision, recall, fscore = dt.precision_recall_fscore(ld.X_test(), ld.y_test())
    print("Precision = {precision}; Recall = {recall}; F-measure = {fscore}".format(
        precision=precision,
        recall=recall,
        fscore=fscore
    ))


@main.command()
@click.argument("dirpath", metavar="<dirpath>")
@click.argument("test_size", metavar="<test_size>", type=float)
@click.argument("vectorizer_type", metavar="<vectorizer_type>")
def logistic_regression(dirpath, test_size, vectorizer_type):
    "Train a logistic regression learning model to classify system/application executions as " \
    "normal or abnormal."
    ld = LearningData(dirpath, test_size, vectorizer_type)
    lr = LogisticRegressionLearningModel()
    lr.fit(ld.X_train(), ld.y_train())
    precision, recall, fscore = lr.precision_recall_fscore(ld.X_test(), ld.y_test())
    print("Precision = {precision}; Recall = {recall}; F-measure = {fscore}".format(
        precision=precision,
        recall=recall,
        fscore=fscore
    ))


@main.command()
@click.argument("dirpath", metavar="<dirpath>")
@click.argument("test_size", metavar="<test_size>", type=float)
@click.argument("vectorizer_type", metavar="<vectorizer_type>")
def svm(dirpath, test_size, vectorizer_type):
    "Train a SVM learning model to classify system/application executions as normal or abnormal."
    ld = LearningData(dirpath, test_size, vectorizer_type)
    svm = SVMLearningModel()
    svm.fit(ld.X_train(), ld.y_train())
    precision, recall, fscore = svm.precision_recall_fscore(ld.X_test(), ld.y_test())
    print("Precision = {precision}; Recall = {recall}; F-measure = {fscore}".format(
        precision=precision,
        recall=recall,
        fscore=fscore
    ))


@main.command()
@click.argument("dirpath", metavar="<dirpath>")
@click.argument("test_size", metavar="<test_size>", type=float)
@click.argument("vectorizer_type", metavar="<vectorizer_type>")
def pca(dirpath, test_size, vectorizer_type):
    "Train a PCA-based learning model to classify system/application executions as normal or " \
    "abnormal."
    ld = LearningData(dirpath, test_size, vectorizer_type)
    pca = PCABasedLearningModel()
    pca.fit(ld.X_train(), ld.y_train())
    precision, recall, fscore = pca.precision_recall_fscore(ld.X_test(), ld.y_test())
    print("Precision = {precision}; Recall = {recall}; F-measure = {fscore}".format(
        precision=precision,
        recall=recall,
        fscore=fscore
    ))


@main.command()
@click.argument("dirpath", metavar="<dirpath>")
@click.argument("test_size", metavar="<test_size>", type=float)
@click.argument("vectorizer_type", metavar="<vectorizer_type>")
def ensemble(dirpath, test_size, vectorizer_type):
    "Train an ensemble learning model to classify system/application executions as normal or " \
    "abnormal."
    ld = LearningData(dirpath, test_size, vectorizer_type)
    ensemble = EnsembleLearningModel()
    ensemble.fit(ld.X_train(), ld.y_train())
    precision, recall, fscore = ensemble.precision_recall_fscore(ld.X_test(), ld.y_test())
    print("Precision = {precision}; Recall = {recall}; F-measure = {fscore}".format(
        precision=precision,
        recall=recall,
        fscore=fscore
    ))


if __name__ == "__main__":
    main()
