python3 props1.py --> generates "props1_processed_data.pickle" which contains the first group of attributes for the data using "parsed_tagged_named_enteties.pickle"
python3 props2.py --> generates "props2_processed_data.pickle" which contains the second group of attributes for the data using the files in the "extracted data" dir
python3 props3.py --> generates "props3_processed_data.pickle" which contains the third group of attributes for the data using "parsed_tagged_named_enteties.pickle"
python3 porps4.py --> generates "props4_processed_data.pickle" which contains the fourth group of attributes for the data using "parsed_tagged_named_enteties.pickle"
python3 get_znacilke.py --> generates "znacilke_without_neigh.pickle" and "znacilke_with_neigh.pickle" which contain all the attributes required for learning
python3 order_data_for_learning.py --> generates "vector_sentences.pickle", "vector_vector_words.pickle", "vector_vector_classes.pickle" and "vector_vector_znacilke.pickle" which contain the orderd data for learning

