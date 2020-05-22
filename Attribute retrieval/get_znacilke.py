import pickle



if __name__ == "__main__":
	props1 = pickle.load(open("props1_processed_data.pickle", "rb"))
	props2 = pickle.load(open("props2_processed_data.pickle", "rb"))
	props3 = pickle.load(open("props3_processed_data.pickle", "rb"))
	props4 = pickle.load(open("props4_processed_data.pickle", "rb"))


	znacilke = dict()


	#tri mozni razponi zvorcev

	#samo -1 1 odmik


	#vse mozne kombinacije -1,0,1

	#vse mozne kombinacije -2,-1,0,1,2

	for sentence in props1:

		p1 = props1[sentence]
		p2 = props2[sentence]
		p3 = props3[sentence]
		p4 = props4[sentence]


		common_p = []

		for i in range(len(p1)):

			head = p1[i][0]
			p1_tail = p1[i][1]
			p2_tail = p2[i][1]
			p3_tail = p3[i][1]
			p4_tail = p4[i][1]

			common_p.append([head, p1_tail + p2_tail + p3_tail + p4_tail])

		znacilke[sentence] = common_p


	file = open("znacilke_without_neigh.pickle", "wb")
	pickle.dump(znacilke, file)
	file.close()

	znacilke = pickle.load(open("znacilke_without_neigh.pickle", "rb"))


	#now we add the env of a word to the znacilke vector


	znacilke2 = dict()

	for sentence in znacilke:

		p = znacilke[sentence]

		combined_p_neigh = []
		
		print(sentence)
		print("+++++++++++++++++++++++++++++++++++++")

		for i in range(len(p)):

			new_vector = []

			new_vector.extend(p[i][1])

			#can we look 2 steps back?
			if i - 2 >= 0:
				new_vector.extend(p[i-2][1])
				
			#can we look 1 step back?
			if i - 1 >= 0:
				new_vector.extend(p[i-1][1])
				

			#can we look 1 step forward?
			if i + 1  <= len(p) - 1:
				new_vector.extend(p[i+1][1])
				
			
			#can we look 2 steps forward?
			if i + 2 <= len(p) - 1:
				new_vector.extend(p[i+2][1])
				

			combined_p_neigh.append([p[i][0], new_vector])
			print([p[i][0], new_vector])
		znacilke2[sentence] = combined_p_neigh
		print("-----------------------------------------")




	file = open("znacilke_with_neigh.pickle", "wb")
	pickle.dump(znacilke2, file)
	file.close()

	znacilke = pickle.load(open("znacilke_without_neigh.pickle", "rb"))
	
	for key in znacilke2:
		print(key)

		for x in znacilke2[key]:
			print(x)

		print("-----------------------")