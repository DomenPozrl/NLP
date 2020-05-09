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


def domen(train_X, train_Y, test_X, test_Y):
    algorithms = ['lbfgs']
    min_frequencies = [0, 0.02]
    all_states = [True, False]
    all_transitions = [True, False]
    c1s = [0, 0.01, 0.05, 0.1]
    c2s = [0, 0.01, 0.05, 0.1]

    i = 1
    N = len(algorithms) * len(min_frequencies) * len(all_states) * len(all_transitions) * len(c1s) * len(c2s)
    start = time.time()

    results = []

    for algo in algorithms:
        for min_freq in min_frequencies:
            for all_state in all_states:
                for all_transition in all_transitions:
                    for c1 in c1s:
                        for c2 in c2s:
                            print(round(100 * i / N), '%')
                            print('Time elapsed: {} s'.format(round(time.time() - start)))
                            i += 1
                            params = {'algo': algo,
                                      'min_freq': min_freq,
                                      'all_state': all_state,
                                      'all_transition': all_transition,
                                      'c1': c1,
                                      'c2': c2}
                            print(params)
                            try:
                                crf = sklearn_crfsuite.CRF(
                                    algorithm=algo,
                                    c1=c1,
                                    c2=c2,
                                    max_iterations=1000,
                                    all_possible_transitions=all_transition,
                                    all_possible_states=all_state,
                                    min_freq=min_freq
                                )

                                crf.fit(train_X, train_Y)
                                pred_Y = crf.predict(test_X)

                                f1 = metrics.flat_f1_score(test_Y, pred_Y, average='weighted',
                                                           labels=['per', 'org', 'misc', 'loc', 'deriv-per',
                                                                   'notpropn'])
                                res = metrics.flat_classification_report(test_Y, pred_Y,
                                                                         labels=['per', 'org', 'misc', 'loc',
                                                                                 'deriv-per', 'notpropn'], digits=4)

                                results.append((f1, params))

                                print(res)
                                print()

                            except:
                                print('Invalid parameter combination.')
                                continue

    file = open('domen', 'wb')
    pickle.dump(results, file)
    file.close()


def jan(train_X, train_Y, test_X, test_Y):
    algorithms = ['l2sgd']
    min_frequencies = [0, 0.02]
    all_states = [True, False]
    all_transitions = [True, False]
    c2s = [0, 0.01, 0.05, 0.1]

    i = 1
    N = len(algorithms) * len(min_frequencies) * len(all_states) * len(all_transitions) * len(c2s)
    start = time.time()

    results = []

    for algo in algorithms:
        for min_freq in min_frequencies:
            for all_state in all_states:
                for all_transition in all_transitions:
                    for c2 in c2s:
                        print(round(100 * i / N), '%')
                        print('Time elapsed: {} s'.format(round(time.time() - start)))
                        i += 1
                        params = {'algo': algo,
                                  'min_freq': min_freq,
                                  'all_state': all_state,
                                  'all_transition': all_transition,
                                  'c2': c2}

                        print(params)
                        try:
                            crf = sklearn_crfsuite.CRF(
                                algorithm=algo,
                                c2=c2,
                                max_iterations=1000,
                                all_possible_transitions=all_transition,
                                all_possible_states=all_state,
                                min_freq=min_freq
                            )

                            crf.fit(train_X, train_Y)
                            pred_Y = crf.predict(test_X)

                            f1 = metrics.flat_f1_score(test_Y, pred_Y, average='weighted',
                                                       labels=['per', 'org', 'misc', 'loc', 'notpropn'])
                            res = metrics.flat_classification_report(test_Y, pred_Y,
                                                                     labels=['per', 'org', 'misc', 'loc',
                                                                             'notpropn'], digits=4)
                            results.append((f1, params))
                            print(res)
                            print()

                        except:
                            print('Invalid parameter combination.')
                            continue

    with open('jan', 'wb') as file:
        pickle.dump(results, file)


def jagos(train_X, train_Y, test_X, test_Y):
    algorithms = ['ap', 'pa', 'arow']
    min_frequencies = [0, 0.02]
    all_states = [True, False]
    all_transitions = [True, False]

    i = 1
    N = len(algorithms) * len(min_frequencies) * len(all_states) * len(all_transitions)
    start = time.time()

    results = []

    for algo in algorithms:
        for min_freq in min_frequencies:
            for all_state in all_states:
                for all_transition in all_transitions:
                    print(round(100 * i / N), '%')
                    print('Time elapsed: {} s'.format(round(time.time() - start)))
                    i += 1
                    params = {'algo': algo,
                              'min_freq': min_freq,
                              'all_state': all_state,
                              'all_transition': all_transition}

                    print(params)
                    try:
                        crf = sklearn_crfsuite.CRF(
                            algorithm=algo,
                            max_iterations=1000,
                            all_possible_transitions=all_transition,
                            all_possible_states=all_state,
                            min_freq=min_freq
                        )

                        crf.fit(train_X, train_Y)
                        pred_Y = crf.predict(test_X)

                        f1 = metrics.flat_f1_score(test_Y, pred_Y, average='weighted',
                                                   labels=['per', 'org', 'misc', 'loc', 'notpropn'])
                        res = metrics.flat_classification_report(test_Y, pred_Y,
                                                                 labels=['per', 'org', 'misc', 'loc',
                                                                         'notpropn'], digits=4)
                        results.append((f1, params))
                        print(res)
                        print()

                    except:
                        print('Invalid parameter combination.')
                        continue

    with open('jagos', 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_train_test(X_filename='../../Data/strip+-1/vector_vector_znacilke.pickle',
                                                       Y_filename='../../Data/strip+-1/vector_vector_classes.pickle',
                                                       train=70,
                                                       test=30)

    domen(train_X, train_Y, test_X, test_Y)
    jan(train_X, train_Y, test_X, test_Y)
    jagos(train_X, train_Y, test_X, test_Y)
