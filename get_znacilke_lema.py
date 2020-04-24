import pickle

if __name__ == "__main__":

    props1 = pickle.load(open("props1_processed_data.pickle", "rb"))
    props2 = pickle.load(open("props2_processed_data.pickle", "rb"))
    props3 = pickle.load(open("props3_processed_data.pickle", "rb"))
    props4 = pickle.load(open("props4_processed_data.pickle", "rb"))

    data = pickle.load(open("parsed_data.pickle", "rb"))

    znacilke = dict()

    for key in data:

        pomozni_dict = dict()

        for word in data[key]:
            vektor = []

            vektor.extend(props1[key][word])
            vektor.extend(props2[key][word])
            vektor.extend(props3[key][word])
            vektor.extend(props4[key][word])

            pomozni_dict[data[key][word][0]] = (vektor, data[key][word][1])

        znacilke[key] = pomozni_dict

    file = open("znacilke_lema.pickle", "wb")
    pickle.dump(znacilke, file)
    file.close()