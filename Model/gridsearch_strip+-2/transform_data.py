import pickle

with open('../../Data/merge_per/vector_vector_znacilke.pickle', 'rb') as f:
    data = pickle.load(f)

data_modified = []
for sentence in data:
    sentence_modified = []
    for word in sentence:
        word_modified = {}
        for key in word:
            if key[0] != '2':
                word_modified[key] = word[key]

        sentence_modified.append(word_modified)

    data_modified.append(sentence_modified)


with open('../../Data/strip+-2/vector_vector_znacilke.pickle', 'wb') as f:
    data = pickle.dump(data_modified, f)
