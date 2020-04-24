import pickle


def process_single_sentence(sentence):

	final_dict = dict()

	for key in sentence:

		vector = []

		#velika zacetnica vector[0]
		if key[0].upper() == key[0] and key[0].isalnum():
			vector.append(1)
		else:
			vector.append(0)

		#le velike črke v celotni besedi vector[1]
		if key.upper() == key and key.isalnum():
			vector.append(1)
		else:
			vector.append(0)

		#mesane velike crke znotraj besede vector[2]
		if not key.upper() == key and not key.lower() == key and not key[0].upper() == key[0]:
			vector.append(1)
		else:
			vector.append(0)

		#stevilke v besedi vector[3]
		if any([True for c in key if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]]):
			vector.append(1)
		else:
			vector.append(0)

		#le stevilke v besedi vector[4]			
		if all([True if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"] else False for c in key]):
			vector.append(1)
		else:
			vector.append(0)


		#numerični izraz vector[5]
		if all([True if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"] or c in ["+", "-", "*", "/"] else False for c in key ]):
			vector.append(1)
		else:
			vector.append(0)

		#alfanumericni izraz, ki vsebuje le stevilke in crke vector[6]
		if key.isalnum():
			vector.append(1)
		else:
			vector.append(0)

		#je rimska stevilka vector[7]
		#I = 1, V = 5, X = 10, L = 50, C = 100, D = 500, M = 1000.
		if all([True if c in ["I", "V", "X", "L", "C", "D", "M"] else False for c in key ]):
			vector.append(1)
		else:
			vector.append(0)

		#vsebuje vezaj ali pomišljaj vector[8]
		if "-" in key:
			vector.append(1)
		else:
			vector.append(0)


		#le velike crke, lahko ločene s piko vector[9]
		if key.upper() == key and all([True if c.isalpha() or c == "." else False for c in key]):
			vector.append(1)
		else:
			vector.append(0) 

		#inicialka vector[10]
		if len(key) == 2 and key[0].isalpha() and key[1] == ".":
			vector.append(1)
		else:
			vector.append(0)


		#posamezne crke ne glede na velikost vector[11]
		if len(key) == 1 and key.isalpha():
			vector.append(1)
		else:
			vector.append(0)


		#posamezne velike crke vector[12]
		if len(key) == 1 and key.isalpha() and key.upper() == key:
			vector.append(1)
		else:
			vector.append(0)

		#locilo vector[13]
		if len(key) == 1 and not key.isalnum():
			vector.append(1)
		else:
			vector.append(0)


		#narekovaj vector[14]
		if len(key) == 1 and key == "'":
			vector.append(1)
		else:
			vector.append(0)

		#le male crke vector[15]
		if key.lower() == key:
			vector.append(1)
		else:
			vector.append(0)

		final_dict[key] = vector

	return final_dict

if __name__ == "__main__":

	data = pickle.load(open("parsed_data.pickle", "rb"))

	props1_processed_data = dict()
	for key in data:
		
		props1_processed_data[key] = process_single_sentence(data[key])


	file = open("props1_processed_data.pickle", "wb")
	pickle.dump(props1_processed_data, file)
	file.close()