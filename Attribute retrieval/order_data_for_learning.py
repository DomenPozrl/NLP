import pickle

def get_dict_from_vector(vector):

	slovar = dict()
	for i in range(len(vector)):
		slovar[i] = vector[i]

	return slovar

if __name__ =="__main__":

	znacilke_filename = "znacilke_with_neigh.pickle"

	data = pickle.load(open(znacilke_filename, "rb"))



	vector_sentences = []
	vector_vector_words = []
	vector_vector_znacilke = []
	vector_vector_classes = []


	for sentence in data:

		vector_sentences.append(sentence)
		
		words = []
		znacilke = []
		classes = []
		
		for p in data[sentence]:

			head, vector = p

			words.append([head[0], head[1], head[2]])
			znacilke.append(get_dict_from_vector(vector))

			if len(head) == 4:
				classes.append(head[3])
			else:
				classes.append("notpropn")

		vector_vector_words.append(words)
		vector_vector_classes.append(classes)
		vector_vector_znacilke.append(znacilke)



	"""a = vector_vector_classes

	for x in a:
		print(x)
		print("-----------------------------------------")"""


	file = open("vector_sentences.pickle", "wb")
	pickle.dump(vector_sentences, file)
	file.close()

	file = open("vector_vector_words.pickle", "wb")
	pickle.dump(vector_vector_words, file)
	file.close()	

	file = open("vector_vector_znacilke.pickle", "wb")
	pickle.dump(vector_vector_znacilke, file)
	file.close()

	file = open("vector_vector_classes.pickle", "wb")
	pickle.dump(vector_vector_classes, file)
	file.close()	