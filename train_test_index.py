import pickle

TRAIN, TEST = 0.6, 0.4


def split_train_test(data, train, test):
    train = train / (train + test)

    # It's probbaly bettere not to shuffle the data, since multiple sentences can come from the same source.
    # Don't shuffle at a later stage either, as it messes up the index
    limit = round(len(data) * train)

    return data[:limit], data[limit:]


if __name__ == "__main__":

    with open("znacilke_lema.pickle", "rb") as f:
        data = pickle.load(f)

    samples = []
    index = {}
    i = 0

    # Go through each word in each sentence and add it to the samples (data that goes into the classifier)
    # and add it to the index (to be able to connect different instances of the same word)
    for sentence in data:
        group = []
        samples.append(group)

        j = 0
        for word in data[sentence]:
            group.append((data[sentence][word][0], 1 if data[sentence][word][1] == 'PROPN' else 0))

            if word not in index:
                index[word] = []

            index[word].append((i, j))

            j += 1

        i += 1

    # Split the data into train and test subsets
    train, test = split_train_test(samples, TRAIN, TEST)

    with open('train_test.pickle', 'wb') as f:
        pickle.dump((train, test), f)

    with open('index.pickle', 'wb') as f:
        pickle.dump(index, f)








