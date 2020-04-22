import pickle


if __name__ == "__main__":
    
    data = pickle.load(open("parsed_data.pickle", "rb"))
    
    processed_data = dict()
    
    for key in data:
    
        pomozni_dict = dict()
        
        for word in data[key]:
            
            if len(word) <= 4:
                element = len(word)
            elif len(word) > 4 <= 9:
                element = 7
            else:
                element = 10
            
            pomozni_dict[word] = [element]
        
        processed_data[key] = pomozni_dict
    
    
    file = open("props4_processed_data.pickle", "wb")
    pickle.dump(processed_data, file)
    file.close()