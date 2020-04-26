import pickle

def get_lema(element):
    return element[1].split("-")[0]

if __name__ == "__main__":

    #ženska osebna imena https://sl.wikipedia.org/wiki/Kategorija:%C5%BDenska_osebna_imena
    #moška osebna imena https://sl.wikipedia.org/wiki/Kategorija:Mo%C5%A1ka_osebna_imena
    #občine v sloveniji https://sl.wikipedia.org/wiki/Kategorija:Ob%C4%8Dine_Slovenije
    #priimki https://sl.wikipedia.org/wiki/Kategorija:Priimki
    #naselja v sloveniji https://sl.wikipedia.org/wiki/Seznam_naselij_v_Sloveniji
    #države https://sl.wikipedia.org/wiki/Seznam_suverenih_dr%C5%BEav
    #glavna mesta držav https://sl.wikipedia.org/wiki/Kategorija:Glavna_mesta
    
    
    zenska_imena = pickle.load(open("zenska_imena.pickle", "rb"))
    moska_imena = pickle.load(open("moska_imena.pickle", "rb"))
    priimki = pickle.load(open("priimki.pickle", "rb"))
    obcine = pickle.load(open("obcine.pickle", "rb"))
    naselja = pickle.load(open("kraji.pickle", "rb"))
    glavna_mesta = pickle.load(open("glavna_mesta.pickle", "rb"))
    drzave = pickle.load(open("drzave.pickle", "rb"))
    
    #pri občinah bodi pozoren, da so notri besede "Običina" in jih dej stran
    
    data = pickle.load(open("parsed_tagged_named_enteties.pickle", "rb"))
    
    props2_processed_data = dict()
    
    for sentence in data:
        
        pomozni = []
        
        for word in data[sentence]:
            vector = []
            lema = get_lema(word)
            
            
            #a je zensko ime
            if lema in zenska_imena:
                vector.append(1)
            else:
                vector.append(0)
                
            #a je mosko ime
            if lema in moska_imena:
                vector.append(1)
            else:
                vector.append(0)
            
            
            #a je priimek
            if lema in priimki:
                vector.append(1)
            else:
                vector.append(0)
            
            #a je kraj
            if lema in naselja:
                vector.append(1)
            else:
                vector.append(0)
                
            #a je glavno mesto
            if lema in glavna_mesta:
                vector.append(1)
            else:
                vector.append(0)
            
            #a  je drzava
            if lema in drzave:
                vector.append(1)
            else:
                vector.append(0)
            
            #a je obcina
            vrednost = 0
            for obcina in obcine:
                if lema in obcina and len(lema) > 2:
                    vrednost = 1
                    break
            
            vector.append(vrednost)
            
            #dan v tednu
            if lema.lower() in ["ponedeljek", "torek", "sreda", "četrtek", "petek", "sobota", "nedelja"]:
                vector.append(1)
            else:
                vector.append(0)
            
            #mesec v letu
            if lema.lower() in ["Januar", "Februar", "Marec", "April", "Maj", "Junij", "Julij", "Avgust", "September", "Oktober", "November", "December"]:
                vector.append(1)
            else:
                vector.append(0)
            
            pomozni.append([word, vector])
            
        props2_processed_data[sentence] = pomozni
    
    file = open("props2_processed_data.pickle", "wb")
    pickle.dump(props2_processed_data, file)
    file.close()


    a = pickle.load(open("props2_processed_data.pickle", "rb"))

    for key in a:
        print(key)
        for x in a[key]:
            print(x)
        print("-----------------------")