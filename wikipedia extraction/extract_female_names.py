import pickle

if __name__ == "__main__":
    
    file = open("zenska_imena.txt", "r", encoding="utf-8")
    
    lines = file.readlines()
    
    imena = []
    
    for line in lines:
        x = line.strip()
        
        if "(ime)" in x:
            x = x.replace("(ime)", "")
        
        if "(žensko ime)" in x:
            x = x.replace("(žensko ime)", "")
        
        if "(osebno ime)" in x:
            x = x.replace("(osebno ime)", "")
        
        if len(x) > 1:
            imena.append(x)
    
    imena.remove("Slovanska imena")
    
    for ime in imena:
        print(ime)
    
    file = open("zenska_imena.pickle", "wb")
    pickle.dump(imena, file)
    file.close()