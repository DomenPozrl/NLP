import pickle

with open('../Data/strip+-1/vector_vector_znacilke.pickle', 'rb') as f:
    data = pickle.load(f)

for sentence in data:
    for word in sentence:
        print(word['trenutna_25'])

with open('../Data/realistic/vector_vector_classes.pickle', 'rb') as f:
    data = pickle.load(f)

classes = set()
for sentence in data:
    for word in sentence:
        classes.add(word)

print(classes)