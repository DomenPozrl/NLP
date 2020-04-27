import pickle

with open('vector_vector_classes.pickle', 'rb') as f:
    data = pickle.load(f)

print(len(data))
for i in data[1]:
    print(i)