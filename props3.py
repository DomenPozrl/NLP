import pickle

if __name__ == "__main__":
	
    data = pickle.load(open("parsed_data.pickle", "rb"))
    
    processed_data = dict()
    
    for key in data:
    
        pomozni_dict = dict()
        
        for word in data[key]:
            element = data[key][word][2]
            
            pomozni_dict[word] = [element]
        
        processed_data[key] = pomozni_dict
    
    
    file = open("props3_processed_data.pickle", "wb")
    pickle.dump(processed_data, file)
    file.close()