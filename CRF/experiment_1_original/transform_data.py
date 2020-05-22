import pickle
import shutil


def run():
    with open('../Attribute retrieval/vector_vector_classes.pickle', 'rb') as f:
        data = pickle.load(f)

    # Merge per and deriv-per classes
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == 'deriv-per':
                data[i][j] = 'per'

    with open('Data/experiment_1_original/vector_vector_classes.pickle', 'wb') as f:
        pickle.dump(data, f)

    shutil.copy('../Attribute retrieval/vector_vector_znacilke.pickle',
                'Data/experiment_1_original/vector_vector_znacilke.pickle')
