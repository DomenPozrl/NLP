from bs4 import BeautifulSoup
import pickle

if __name__ == "__main__":
    
    soup = BeautifulSoup(open("Seznam suverenih držav - Wikipedija, prosta enciklopedija.html", encoding="utf-8"), "html.parser")
    
    
    drzave = []
    
    for drzava in soup.find_all("tr")[3:-1]:
        if len(drzava.text.split("\n")) >= 6:
            drzave.append(drzava.text.split("\n")[1])
    
    
    drzave.remove("")
    
    """for drzava in drzave:
        print(drzava)
    
    print(len(drzave))"""
    
    file = open("drzave.pickle", "wb")
    pickle.dump(drzave, file)
    file.close()