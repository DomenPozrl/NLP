from sklearn.metrics import make_scorer
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


def stringify(x):
    x = [str(i) for i in x]
    x = x[:-2] + x[-1:]
    return x


def load_train_test(filename: str):
    with open(filename, 'rb') as f:
        train, test = pickle.load(f)

    train_X, train_Y = [], []
    for sentence in train:

        sentence_X = []
        sentence_Y = []
        for word in sentence:
            sentence_X.append(stringify(word[0]))
            sentence_Y.append(str(word[1]))

        train_X.append(sentence_X)
        train_Y.append(sentence_Y)

    test_X, test_Y = [], []
    for sentence in test:

        sentence_X = []
        sentence_Y = []
        for word in sentence:
            sentence_X.append(stringify(word[0]))
            sentence_Y.append(str(word[1]))

        test_X.append(sentence_X)
        test_Y.append(sentence_Y)

    return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':

    train_X, train_Y, test_X, test_Y = load_train_test('train_test.pickle')
    print(train_X[0])

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(train_X, train_Y)
    pred_Y = crf.predict(test_X)

    print(metrics.flat_f1_score(test_Y, pred_Y, average='weighted', labels=['1', '0']))
    print(metrics.flat_classification_report(test_Y, pred_Y, labels=['1', '0'], digits=3))
