import pickle
import shutil


def run():
    with open('Data/experiment_3_strip1/vector_vector_znacilke.pickle', 'rb') as f:
        data = pickle.load(f)

    data_modified = []
    for sentence in data:
        sentence_modified = []
        for word in sentence:
            word_modified = {}
            for key in word:
                if key != 'trenutna_25':
                    word_modified[key] = word[key]

            sentence_modified.append(word_modified)

        data_modified.append(sentence_modified)

    with open('Data/experiment_4_morphosintactic/vector_vector_znacilke.pickle', 'wb') as f:
        pickle.dump(data_modified, f)

    shutil.copy('Data/experiment_2_strip2/vector_vector_classes.pickle',
                'Data/experiment_4_morphosintactic/vector_vector_classes.pickle')
