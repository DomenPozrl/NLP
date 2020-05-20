import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import math


def cross_validation_split(x_filename: str, y_filename: str, k: int):
    x_folds = []
    y_folds = []

    with open(x_filename, 'rb') as f:
        x = pickle.load(f)

    with open(y_filename, 'rb') as f:
        y = pickle.load(f)

    limit = math.ceil(len(x) / k)

    for k in range(k):
        x_folds.append(x[k * limit: (k + 1) * limit])
        y_folds.append(y[k * limit: (k + 1) * limit])

    return x_folds, y_folds


def folds_2_tt(x_folds, y_folds, k):
    test_x = x_folds[k]
    test_y = y_folds[k]

    train_x = []
    train_y = []
    for i in range(len(x_folds)):
        if i != k:
            train_x += x_folds[i]
            train_y += y_folds[i]

    return test_x, test_y, train_x, train_y


def cross_validate(x_folds, y_folds, params):
    f1_per = []
    f1_org = []
    f1_misc = []
    f1_loc = []
    f1_not = []

    for i in range(len(x_folds)):
        print('\rWorking on fold {}/{} ...'.format(i + 1, len(x_folds)), end='')

        crf = sklearn_crfsuite.CRF(**params)

        test_x, test_y, train_x, train_y = folds_2_tt(x_folds, y_folds, i)

        crf.fit(train_x, train_y)
        pred_y = crf.predict(test_x)

        f1_per.append(metrics.flat_f1_score(test_y, pred_y, average=None, labels=['per']))
        f1_org.append(metrics.flat_f1_score(test_y, pred_y, average=None, labels=['org']))
        f1_misc.append(metrics.flat_f1_score(test_y, pred_y, average=None, labels=['misc']))
        f1_loc.append(metrics.flat_f1_score(test_y, pred_y, average=None, labels=['loc']))
        f1_not.append(metrics.flat_f1_score(test_y, pred_y, average=None, labels=['notpropn']))

    print()

    return f1_per, f1_org, f1_misc, f1_loc, f1_not


def class_support(y_folds):
    support = {}

    for k in y_folds:
        for sentence in k:
            for word in sentence:
                if word not in support:
                    support[word] = 0

                support[word] += 1

    return support


if __name__ == '__main__':
    x_folds, y_folds = cross_validation_split(x_filename='../../Data/original/vector_vector_znacilke.pickle',
                                              y_filename='../../Data/original/vector_vector_classes.pickle',
                                              k=5)

    support = class_support(y_folds)

    params = {'algorithm': 'lbfgs',
              'c1': 0.1, 'c2': 0.05,
              'max_iterations': 10000,
              'all_possible_transitions': True,
              'all_possible_states': True,
              'min_freq': 0}

    f1_per, f1_org, f1_misc, f1_loc, f1_not = cross_validate(x_folds, y_folds, params)

    avg_per = sum(f1_per) / len(f1_per)
    print(avg_per)
    print(f1_per)
    print()

    avg_org = sum(f1_org) / len(f1_org)
    print(avg_org)
    print(f1_org)
    print()

    avg_misc = sum(f1_misc) / len(f1_misc)
    print(avg_misc)
    print(f1_misc)
    print()

    avg_loc = sum(f1_loc) / len(f1_loc)
    print(avg_loc)
    print(f1_loc)
    print()

    avg_not = sum(f1_not) / len(f1_not)
    print(avg_not)
    print(f1_not)
    print()

    # Expected: 0.9936
    final_score = ((avg_per * support['per']) +
                   (avg_loc * support['loc']) +
                   (avg_misc * support['misc']) +
                   (avg_not * support['notpropn']) +
                   (avg_org * support['org'])) / sum(support.values())
    print('Final Score: {}'.format(final_score))

