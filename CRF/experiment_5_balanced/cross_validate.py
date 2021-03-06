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

    precision_per = []
    precision_org = []
    precision_misc = []
    precision_loc = []
    precision_not = []

    recall_per = []
    recall_org = []
    recall_misc = []
    recall_loc = []
    recall_not = []

    for i in range(len(x_folds)):
        print('\rWorking on fold {}/{} ...'.format(i + 1, len(x_folds)), end='')

        crf = sklearn_crfsuite.CRF(**params)

        test_x, test_y, train_x, train_y = folds_2_tt(x_folds, y_folds, i)
        train_x, train_y = balance(train_x, train_y)

        crf.fit(train_x, train_y)
        pred_y = crf.predict(test_x)

        f1_per.append(metrics.flat_f1_score(test_y, pred_y, average=None, labels=['per']))
        f1_org.append(metrics.flat_f1_score(test_y, pred_y, average=None, labels=['org']))
        f1_misc.append(metrics.flat_f1_score(test_y, pred_y, average=None, labels=['misc']))
        f1_loc.append(metrics.flat_f1_score(test_y, pred_y, average=None, labels=['loc']))
        f1_not.append(metrics.flat_f1_score(test_y, pred_y, average=None, labels=['notpropn']))

        precision_per.append(metrics.flat_precision_score(test_y, pred_y, average=None, labels=['per']))
        precision_org.append(metrics.flat_precision_score(test_y, pred_y, average=None, labels=['org']))
        precision_misc.append(metrics.flat_precision_score(test_y, pred_y, average=None, labels=['misc']))
        precision_loc.append(metrics.flat_precision_score(test_y, pred_y, average=None, labels=['loc']))
        precision_not.append(metrics.flat_precision_score(test_y, pred_y, average=None, labels=['notpropn']))

        recall_per.append(metrics.flat_recall_score(test_y, pred_y, average=None, labels=['per']))
        recall_org.append(metrics.flat_recall_score(test_y, pred_y, average=None, labels=['org']))
        recall_misc.append(metrics.flat_recall_score(test_y, pred_y, average=None, labels=['misc']))
        recall_loc.append(metrics.flat_recall_score(test_y, pred_y, average=None, labels=['loc']))
        recall_not.append(metrics.flat_recall_score(test_y, pred_y, average=None, labels=['notpropn']))

    print()
    avg_per_f1 = sum(f1_per) / len(f1_per)
    avg_org_f1 = sum(f1_org) / len(f1_org)
    avg_loc_f1 = sum(f1_loc) / len(f1_loc)
    avg_misc_f1 = sum(f1_misc) / len(f1_misc)
    avg_not_f1 = sum(f1_not) / len(f1_not)

    avg_per_precision = sum(precision_per) / len(precision_per)
    avg_org_precision = sum(precision_org) / len(precision_org)
    avg_loc_precision = sum(precision_loc) / len(precision_loc)
    avg_misc_precision = sum(precision_misc) / len(precision_misc)
    avg_not_precision = sum(precision_not) / len(precision_not)

    avg_per_recall = sum(recall_per) / len(recall_per)
    avg_org_recall = sum(recall_org) / len(recall_org)
    avg_loc_recall = sum(recall_loc) / len(recall_loc)
    avg_misc_recall = sum(recall_misc) / len(recall_misc)
    avg_not_recall = sum(recall_not) / len(recall_not)

    result = {'per': (avg_per_precision, avg_per_recall, avg_per_f1),
              'org': (avg_org_precision, avg_org_recall, avg_org_f1),
              'misc': (avg_misc_precision, avg_misc_recall, avg_misc_f1),
              'loc': (avg_loc_precision, avg_loc_recall, avg_loc_f1),
              'not': (avg_not_precision, avg_not_recall, avg_not_f1)}

    return result


def class_support(y_folds):
    support = {}

    for k in y_folds:
        for sentence in k:
            for word in sentence:
                if word not in support:
                    support[word] = 0

                support[word] += 1

    return support


def balance(train_x, train_y):
    balanced_x = []
    balanced_y = []

    for sentence in range(len(train_y)):
        include = False
        for word in train_y[sentence]:
            if word != 'notpropn':
                include = True
                break

        if include:
            balanced_x.append(train_x[sentence])
            balanced_y.append(train_y[sentence])

    return balanced_x, balanced_y


if __name__ == '__main__':
    x_folds, y_folds = cross_validation_split(x_filename='../Data/experiment_5_balanced/vector_vector_znacilke.pickle',
                                              y_filename='../Data/experiment_5_balanced/vector_vector_classes.pickle',
                                              k=5)

    support = class_support(y_folds)

    # Optimal hyper-parameters (based on gridsearch)
    params = {'algorithm': 'lbfgs',
              'c2': 0, 'c1': 0.01,
              'max_iterations': 10000,
              'all_possible_transitions': False,
              'all_possible_states': False,
              'min_freq': 0}

    result = cross_validate(x_folds, y_folds, params)

    for cls in result:
        print('{}: {}'.format(cls, result[cls]))

# RESULTS:
# per: (array([0.79061962]), array([0.84283167]), array([0.81482216]))
# org: (array([0.47089716]), array([0.34689024]), array([0.38873154]))
# misc: (array([0.1880303]), array([0.08595238]), array([0.11619048]))
# loc: (array([0.83962883]), array([0.83287815]), array([0.835268]))
# not: (array([0.99833686]), array([0.99880955]), array([0.99857299]))
