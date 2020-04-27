import pickle

with open('../../Data/original/vector_vector_classes.pickle', 'rb') as f:
    data = pickle.load(f)

for i in range(len(data)):
    for j in range(len(data[i])):
        if data[i][j] == 'deriv-per':
            data[i][j] = 'per'


with open('../../Data/merge_per/vector_vector_classes.pickle', 'wb') as f:
    pickle.dump(data, f)

