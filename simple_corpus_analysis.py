import pickle






if __name__ == "__main__":
    data = pickle.load(open("parsed_data.pickle", "rb"))
    
    print(f"Število povedi: {len(data)}")
    count_words = 0
    
    
    word_types = dict()
    for key in data:
        count_words += len(key.split(" "))
        
        for key2 in data[key]:
            tip = data[key][key2][1]
            if tip in word_types:
                word_types[tip] += 1
            else:
                word_types[tip] = 1
    
    print(f"Število besed: {count_words }")
    
    for key in word_types:
        print(f"Število {key}: {word_types[key]}")
        
        
    
    """Število povedi: 7958
    Število besed: 122114
    Število PUNCTUATION ločila: 14318
    Število DET ??: 5189
    Število NOUN samostalnik: 29748
    Število AUXILARY pomožni glagol: 8347
    Število VERB glagol: 14334
    Število PRONOUN zaimek: 5041
    Število ADP predlog: 11949
    Število SCONJUCTION priredni veznik: 4716
    Število PROPN lastno ime: 4631
    Število ADJECTIVE pridevnik: 14952
    Število CCONJUNCTION podredni veznik: 5103
    Število PARTICLE členek: 3987
    Število ADVERB prislov: 6521
    Število NUMBER številka: 1891
    Število X RESIDUAL ostalo: 330
    Število INTERJECTION medmet: 16"""