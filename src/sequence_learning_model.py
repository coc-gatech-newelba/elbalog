"""Sequence-based learning models for anomaly detection."""


import click
import dynet_config
dynet_config.set(random_seed=717328633)
import dynet as dy

from learning_data import LearningData
from learning_model import LearningModel


# python sequence_learning_model.py srn ../datasets/preproc/cpu_input_20_10 param 101 0.98 75 75 4 5

ANOMALY_FUNCTION = {
    "path": lambda next_token, top_predictions: next_token not in top_predictions,
    "param": lambda next_token, top_predictions: next_token not in top_predictions
}


class SequenceLearningModel(LearningModel):
    def __init__(self, n_tokens, input_dim, hidden_dim, top_g, is_anomaly):
        """Sequence-based learning model.

        n_tokens -- [int] number of token types.
        input_dim -- [int] input dimension.
        hidden_dim -- [int] hidden dimension.
        top_g -- [int] number of top-predicted tokens to check.
        is_anomaly -- [function(int, list of int)] return True if the next token, passed as the
                      first parameter, is an anomaly, given a list of top-predicted tokens passed as
                      the second parameter.
        """
        self._top_g = top_g
        self._pc = dy.ParameterCollection()
        self._lookup = self._pc.add_lookup_parameters((n_tokens + 1, input_dim))
        self._R = self._pc.add_parameters((n_tokens + 1, hidden_dim))
        self._bias = self._pc.add_parameters((n_tokens + 1))
        self._eos = n_tokens
        self._is_anomaly = is_anomaly

    def fit(self, X):
        """Fit the sequence-based learning model to the training data.

        X -- [array of n_samples sequences] Training data.
        """
        trainer = dy.SimpleSGDTrainer(self._pc)
        for x in X:
            dy.renew_cg()
            s0 = self._model.initial_state()
            sentence = [self._eos] + [int(token) for token in x] + [self._eos]
            s = s0
            loss = []
            for (token, next_token) in zip(sentence, sentence[1:]):
                s = s.add_input(self._lookup[token])
                probs = dy.softmax(self._R * s.output() + self._bias)
                loss.append(-dy.log(dy.pick(probs, next_token)))
            loss = dy.esum(loss)
            loss.backward()
            trainer.update()

    def predict(self, X):
        """Return predicted labels for the test data.

        X -- [array of n_samples sequences] Test data.
        """
        Y_pred = []
        for x in X:
            dy.renew_cg()
            s0 = self._model.initial_state()
            sentence = [self._eos] + [int(token) for token in x] + [self._eos]
            s = s0
            for (i, (token, next_token)) in enumerate(zip(sentence, sentence[1:])):
                s = s.add_input(self._lookup[token])
                probs = dy.softmax(self._R * s.output() + self._bias)
                probs = probs.vec_value()
                top_tokens = list(reversed(
                    sorted(range(len(probs)), key=lambda i: probs[i])[-self._top_g:]
                ))
                if next_token != self._eos and self._is_anomaly(next_token, top_tokens):
                    Y_pred.append(1)
                    break
            else:
                Y_pred.append(0)
        return Y_pred


class SRNLearningModel(SequenceLearningModel):
    def __init__(self, n_tokens, input_dim, hidden_dim, n_layers, top_g, is_anomaly):
        """Initialize a simple recurrent network.

        n_tokens -- [int] number of token types.
        input_dim -- [int] input dimension.
        hidden_dim -- [int] hidden dimension.
        n_layers -- [int] number of layers.
        top_g -- [int] number of top-predicted tokens to check.
        is_anomaly -- [function(int, list of int)] return True if the next token, passed as the
                      first parameter, is an anomaly, given a list of top-predicted tokens passed as
                      the second parameter.
        """
        super().__init__(n_tokens, input_dim, hidden_dim, top_g, is_anomaly)
        self._model = dy.SimpleRNNBuilder(n_layers, input_dim, hidden_dim, self._pc)


class LSTMLearningModel(SequenceLearningModel):
    def __init__(self, n_tokens, input_dim, hidden_dim, n_layers, top_g, is_anomaly):
        """Initialize a LSTM network.

        n_tokens -- [int] number of token types.
        input_dim -- [int] input dimension.
        hidden_dim -- [int] hidden dimension.
        n_layers -- [int] number of layers.
        top_g -- [int] number of top-predicted tokens to check.
        is_anomaly -- [function(int, list of int)] return True if the next token, passed as the
                      first parameter, is an anomaly, given a list of top-predicted tokens passed as
                      the second parameter.
        """
        super().__init__(n_tokens, input_dim, hidden_dim, top_g, is_anomaly)
        self._model = dy.LSTMBuilder(n_layers, input_dim, hidden_dim, self._pc)


class GRULearningModel(SequenceLearningModel):
    def __init__(self, n_tokens, input_dim, hidden_dim, n_layers, top_g, is_anomaly):
        """Initialize a GRU network.

        n_tokens -- [int] number of token types.
        input_dim -- [int] input dimension.
        hidden_dim -- [int] hidden dimension.
        n_layers -- [int] number of layers.
        top_g -- [int] number of top-predicted tokens to check.
        is_anomaly -- [function(int, list of int)] return True if the next token, passed as the
                      first parameter, is an anomaly, given a list of top-predicted tokens passed as
                      the second parameter.
        """
        super().__init__(n_tokens, input_dim, hidden_dim, top_g, is_anomaly)
        self._model = dy.GRUBuilder(n_layers, input_dim, hidden_dim, self._pc)


@click.group()
def main():
    pass


@main.command()
@click.argument("dirpath", metavar="<dirpath>")
@click.argument("anomaly_type", metavar="<anomaly_type>")
@click.argument("n_tokens", metavar="<n_tokens>", type=int)
@click.argument("test_size", metavar="<test_size>", type=float)
@click.argument("input_dim", metavar="<input_dim>", type=int)
@click.argument("hidden_dim", metavar="<hidden_dim>", type=int)
@click.argument("n_layers", metavar="<n_layers>", type=int)
@click.argument("top_g", metavar="<top_g>", type=int)
def srn(dirpath, anomaly_type, n_tokens, test_size, input_dim, hidden_dim, n_layers, top_g):
    "Train a SRN to classify system/application executions as normal or abnormal."
    ld = LearningData(dirpath, test_size)
    srn = SRNLearningModel(
        n_tokens, input_dim, hidden_dim, n_layers, top_g, ANOMALY_FUNCTION[anomaly_type]
    )
    srn.fit(ld.X_train())
    precision, recall, fscore = srn.precision_recall_fscore(ld.X_test(), ld.y_test())
    print("Precision = {precision}; Recall = {recall}; F-measure = {fscore}".format(
        precision=precision,
        recall=recall,
        fscore=fscore
    ))


@main.command()
@click.argument("dirpath", metavar="<dirpath>")
@click.argument("anomaly_type", metavar="<anomaly_type>")
@click.argument("n_tokens", metavar="<n_tokens>", type=int)
@click.argument("test_size", metavar="<test_size>", type=float)
@click.argument("input_dim", metavar="<input_dim>", type=int)
@click.argument("hidden_dim", metavar="<hidden_dim>", type=int)
@click.argument("n_layers", metavar="<n_layers>", type=int)
@click.argument("top_g", metavar="<top_g>", type=int)
def lstm(dirpath, anomaly_type, n_tokens, test_size, input_dim, hidden_dim, n_layers, top_g):
    "Train a LSTM network to classify system/application executions as normal or abnormal."
    ld = LearningData(dirpath, test_size)
    lstm = LSTMLearningModel(
        n_tokens, input_dim, hidden_dim, n_layers, top_g, ANOMALY_FUNCTION[anomaly_type]
    )
    lstm.fit(ld.X_train())
    precision, recall, fscore = lstm.precision_recall_fscore(ld.X_test(), ld.y_test())
    print("Precision = {precision}; Recall = {recall}; F-measure = {fscore}".format(
        precision=precision,
        recall=recall,
        fscore=fscore
    ))


@main.command()
@click.argument("dirpath", metavar="<dirpath>")
@click.argument("anomaly_type", metavar="<anomaly_type>")
@click.argument("n_tokens", metavar="<n_tokens>", type=int)
@click.argument("test_size", metavar="<test_size>", type=float)
@click.argument("input_dim", metavar="<input_dim>", type=int)
@click.argument("hidden_dim", metavar="<hidden_dim>", type=int)
@click.argument("n_layers", metavar="<n_layers>", type=int)
@click.argument("top_g", metavar="<top_g>", type=int)
def gru(dirpath, anomaly_type, n_tokens, test_size, input_dim, hidden_dim, n_layers, top_g):
    "Train a GRU network to classify system/application executions as normal or abnormal."
    ld = LearningData(dirpath, test_size)
    gru = GRULearningModel(
        n_tokens, input_dim, hidden_dim, n_layers, top_g, ANOMALY_FUNCTION[anomaly_type]
    )
    gru.fit(ld.X_train())
    precision, recall, fscore = gru.precision_recall_fscore(ld.X_test(), ld.y_test())
    print("Precision = {precision}; Recall = {recall}; F-measure = {fscore}".format(
        precision=precision,
        recall=recall,
        fscore=fscore
    ))


if __name__ == "__main__":
    main()
