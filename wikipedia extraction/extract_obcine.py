import pickle

if __name__ == "__main__":
    
    file = open("obcine.txt", "r", encoding="utf-8")
    
    lines = file.readlines()
    
    imena = []
    
    for line in lines:
        x = line.strip()
        
        if len(x) > 1:
            imena.append(x)
    
    
    for ime in imena:
        print(ime)
    
    file = open("obcine.pickle", "wb")
    pickle.dump(imena, file)
    file.close()