import pickle

if __name__ == "__main__":
	
    data = pickle.load(open("parsed_tagged_named_enteties.pickle", "rb"))
    
    processed_data = dict()
    
    for key in data:
    
        pomozni = []
        
        for word in data[key]:
            
            
            pomozni.append([word, [word[2]]])
        
        processed_data[key] = pomozni
    
    
    file = open("props3_processed_data.pickle", "wb")
    pickle.dump(processed_data, file)
    file.close()

    a = pickle.load(open("props3_processed_data.pickle", "rb"))

    for key in a:
        print(key)
        for x in a[key]:
            print(x)
        print("-----------------------")