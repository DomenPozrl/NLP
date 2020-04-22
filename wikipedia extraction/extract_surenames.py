import pickle

if __name__ == "__main__":
    
    file = open("priimki.txt", "r", encoding="utf-8")
    
    lines = file.readlines()
    
    imena = []
    
    for line in lines:
        x = line.strip()
        
        if "(priimek)" in x:
            x = x.replace("(priimek)", "")
        
        if "(razločitev)" in x:
            x = x.replace("(razločitev)", "")


        if len(x) > 1:
            imena.append(x)
    

    for ime in imena:
        print(ime)
    
    file = open("priimki.pickle", "wb")
    pickle.dump(imena, file)
    file.close()