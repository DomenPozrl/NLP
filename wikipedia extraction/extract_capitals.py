from bs4 import BeautifulSoup
import pickle

if __name__ == "__main__":
    
    soup = BeautifulSoup(open("Kategorija_Glavna mesta Afrike - Wikipedija, prosta enciklopedija.html", encoding="utf-8"), "html.parser")
    
    
    mesta = set()
    
    for mesto in soup.find_all("div", {"class": "mw-content-ltr" }):
        for el in mesto.find_all("li")[6:]:
            if "Seznam glavnih mest Afrike" not in el.text:
                x = el.text
                if "(mesto)" in x:
                    x = x.replace("(mesto)", "")
                    mesta.add(x)
                else:
                    mesta.add(x)
    
    #print(mesta)
    
    
    print("=====================================")
    print("=====================================")
    
    soup = BeautifulSoup(open("Kategorija_Glavna mesta Azije - Wikipedija, prosta enciklopedija.html", encoding="utf-8"), "html.parser")
    
    
    #mesta = set()
    
    for mesto in soup.find_all("div", {"class": "mw-content-ltr" }):
        for el in mesto.find_all("li")[18:]:
            if "Seznam glavnih mest Azije" not in el.text:
                x = el.text
                if "(mesto)" in x:
                    x = x.replace("(mesto)", "")
                    mesta.add(x)
                else:
                    mesta.add(x)
    
    #print(mesta)
    
    print("=====================================")
    print("=====================================")
    
    soup = BeautifulSoup(open("Kategorija_Glavna mesta Evrope - Wikipedija, prosta enciklopedija.html", encoding="utf-8"), "html.parser")
    
    
    #mesta = set()
    
    for mesto in soup.find_all("div", {"class": "mw-content-ltr" }):
        for el in mesto.find_all("li")[37:]:
            if "Seznam glavnih mest Evrope" not in el.text:
                
                x = el.text
                if "(mesto)" in x:
                    x = x.replace("(mesto)", "")
                    mesta.add(x)
                else:
                    mesta.add(x)
    
    #print(mesta)
    
    
    print("=====================================")
    print("=====================================")
    
    soup = BeautifulSoup(open("Kategorija_Glavna mesta Južne Amerike - Wikipedija, prosta enciklopedija.html", encoding="utf-8"), "html.parser")
    
    
    #mesta = set()
    
    for mesto in soup.find_all("div", {"class": "mw-content-ltr" }):
        for el in mesto.find_all("li")[9:]:
            if "Seznam glavnih mest Južne Amerike" not in el.text:
                
                x = el.text
                if "(mesto)" in x:
                    x = x.replace("(mesto)", "")
                    mesta.add(x)
                else:
                    mesta.add(x)
    
    #print(mesta)
    
    print("=====================================")
    print("=====================================")
    
    soup = BeautifulSoup(open("Kategorija_Glavna mesta Oceanije - Wikipedija, prosta enciklopedija.html", encoding="utf-8"), "html.parser")
    
    
    #mesta = set()
    
    for mesto in soup.find_all("div", {"class": "mw-content-ltr" }):
        for el in mesto.find_all("li")[5:]:
            if "Seznam glavnih mest Oceanije" not in el.text:
                
                x = el.text
                if "(mesto)" in x:
                    x = x.replace("(mesto)", "")
                    mesta.add(x)
                else:
                    mesta.add(x)
    
    #print(mesta)
    
    
    
    print("=====================================")
    print("=====================================")
    
    soup = BeautifulSoup(open("Kategorija_Glavna mesta Severne Amerike - Wikipedija, prosta enciklopedija.html", encoding="utf-8"), "html.parser")
    
    
    #mesta = set()
    
    for mesto in soup.find_all("div", {"class": "mw-content-ltr" }):
        for el in mesto.find_all("li")[6:]:
            if "Seznam glavnih mest Severne Amerike" not in el.text:
                x = el.text
                if "(mesto)" in x:
                    x = x.replace("(mesto)", "")
                    mesta.add(x)
                else:
                    mesta.add(x)
    
    file = open("glavna_mesta.pickle", "wb")
    pickle.dump(list(mesta), file)
    file.close()
    print(mesta)