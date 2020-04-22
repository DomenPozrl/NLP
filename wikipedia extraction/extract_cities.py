import pickle

if __name__ == "__main__":
    
    file = open("kraji.txt", "r", encoding="utf-8")
    
    lines = file.readlines()
    
    imena = []
    
    for line in lines:
        imena.append(line.split("\t")[0])
    
    file = open("kraji.pickle", "wb")
    pickle.dump(imena, file)
    file.close()