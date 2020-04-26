import pickle


if __name__ == "__main__":
    
    data = pickle.load(open("parsed_tagged_named_enteties.pickle", "rb"))
    
    processed_data = dict()
    
    for key in data:
    
        pomozni = []
        
        for elementx in data[key]:
            word = elementx[0]

            if len(word) <= 4:
                element = len(word)
            elif len(word) > 4 and len(word) <= 9:
                element = 7
            else:
                element = 10
            
            pomozni.append([elementx, [element]])
        
        processed_data[key] = pomozni
    
    
    file = open("props4_processed_data.pickle", "wb")
    pickle.dump(processed_data, file)
    file.close()


    a = pickle.load(open("props4_processed_data.pickle", "rb"))

    for key in a:
        print(key)
        for x in a[key]:
            print(x)
        print("-----------------------")



    


    
