import pickle
import time
import sklearn_crfsuite
from sklearn_crfsuite import metrics


def load_train_test(X_filename: str, Y_filename: str, train: float, test: float):
    train_test_limit = train / (train + test)

    with open(X_filename, 'rb') as f:
        X = pickle.load(f)
        train_X = X[:round(train_test_limit * len(X))]
        test_X = X[round(train_test_limit * len(X)):]

    with open(Y_filename, 'rb') as f:
        Y = pickle.load(f)
        train_Y = Y[:round(train_test_limit * len(Y))]
        test_Y = Y[round(train_test_limit * len(Y)):]

    return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':

    train_X, train_Y, test_X, test_Y = load_train_test(X_filename='../Data/original/vector_vector_znacilke.pickle',
                                                       Y_filename='../Data/original/vector_vector_classes.pickle',
                                                       train=70,
                                                       test=30)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.01,
        c2=0.01,
        max_iterations=10000,
        all_possible_transitions=True,
        all_possible_states=True,
        min_freq=0)

    crf.fit(train_X, train_Y)
    pred_Y = crf.predict(test_X)

    f1 = metrics.flat_f1_score(test_Y, pred_Y, average='weighted', labels=['per', 'org', 'misc', 'loc', 'deriv-per', 'notpropn'])
    print(metrics.flat_f1_score(test_Y, pred_Y, average=None, labels=['per']))
    results = metrics.flat_classification_report(test_Y, pred_Y, labels=['per', 'org', 'misc', 'loc', 'deriv-per', 'notpropn'], digits=4)
    print(results)


