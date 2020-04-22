import pickle

def parse_file(filename):
    file = open(filename, "r", encoding="utf8")
    lines = file.readlines()
    data = dict()
    
    sample = []
    for i in range(len(lines)):
        print(i/len(lines))
        
        line = lines[i]
        if line == "\n":
            poved, vzorec = parse_sample(sample)
            data[poved] = vzorec
            sample = []
        else:
            sample.append(line)
    
    return data
    
def parse_sample(sample):
    vzorec = dict()
    poved = ""
    for line in sample:
        split_line = line.split("\t")
        #print(split_line)
        if split_line[0][0] != "#":
            vzorec[split_line[1]] = (split_line[2], split_line[3], split_line[4])
        elif "text" in split_line[0]:
            poved = split_line[0].replace("# text = ", "").strip()
            
    return poved, vzorec


if __name__ == "__main__":
    
    
    data = parse_file("sl_ssj-ud_v2.4.conllu")
    
    
    file = open('parsed_data.pickle', 'wb')

    
    pickle.dump(data, file)

    
    file.close()

