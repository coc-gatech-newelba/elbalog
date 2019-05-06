"""A data model for anomaly detection learning models."""


import os
import random

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


class LearningData:
    def __init__(self, dirpath, test_size, vectorizer_type=None):
        """Load, vectorize, and split labeled data into training and test subsets.
        
        The specified directory must contain two files: "normal", containing sequences of log keys
        resulting from normal executions; and "abnormal", containing sequences of log keys resulting
        from abnormal executions. Each line of these files contains a space-separated sequence of
        log keys representing an execution session.

        dirpath -- [str] Path to the directory containing labeled data.
        test_size -- [float] Proportion of test data.
        vectorizer_type -- [None or "count" or "tfidf"] Option "count" converts a sequence of log
                           keys to a vector of log key counts; option "tfidf" converts a sequence of
                           log keys to a vector of TF-IDF features.
        """
        normal_sequences = []
        with open(os.path.join(dirpath, "normal")) as normal_file:
            for line in normal_file.read().strip().split('\n'):
                normal_sequences.append(line)
        abnormal_sequences = []
        with open(os.path.join(dirpath, "abnormal")) as abnormal_file:
            for line in abnormal_file.read().strip().split('\n'):
                abnormal_sequences.append(line)
        if vectorizer_type is not None:
            vectorizer = {"count": CountVectorizer(), "tfidf": TfidfVectorizer()}
            X = vectorizer[vectorizer_type].fit_transform(
                normal_sequences + abnormal_sequences
            ).toarray()
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
                X, [0] * len(normal_sequences) + [1] * len(abnormal_sequences), test_size=test_size,
                random_state=4120116722
            )
        else:
            random.seed(4120116722)
            random.shuffle(normal_sequences)
            n_training_samples = int(
                (1 - test_size) * (len(normal_sequences) + len(abnormal_sequences))
            )
            self._X_train = [
                sequence.split()
                for sequence in normal_sequences[:n_training_samples]
            ]
            self._y_train = [0] * n_training_samples
            self._X_test = [
                sequence.split()
                for sequence in (normal_sequences[n_training_samples:] + abnormal_sequences)
            ]
            self._y_test = [0] * (len(normal_sequences) - n_training_samples) + \
                    [1] * len(abnormal_sequences)

    def X_train(self):
        """Return the training data."""
        return self._X_train

    def y_train(self):
        """Return the target values for the training data."""
        return self._y_train

    def X_test(self):
        """Return the test data."""
        return self._X_test

    def y_test(self):
        """Return the target values for the test data."""
        return self._y_test
