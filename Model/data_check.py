import pickle

with open('../Data/original/vector_vector_classes.pickle', 'rb') as f:
    data = pickle.load(f)

with open('../Data/original/vector_sentences.pickle', 'rb') as f:
    sentences = pickle.load(f)


for i in range(len(data)):
    for j in range(len(data[i])):
        if data[i][j] == 'deriv-per':
            print('deriv-per: ({}, {})'.format(i, j))
            print(sentences[i])
            print()
        if data[i][j] == 'misc':
            print('misc: ({}, {})'.format(i, j))
            print(sentences[i])
            print()



