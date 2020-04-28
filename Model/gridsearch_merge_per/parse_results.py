results = {}
for filename in ['grid_search_domen_merge_per.txt', 'grid_search_jan_merge_per.txt', 'grid_search_jagos_merge_per.txt']:
    f = open(filename, 'r')

    params = None
    for line in f:
        if line[0] == '{':
            params = line
            results[line] = {}
            continue

        line = line.strip().split(' ')
        if line[0] == 'per':
            i = 0
            f1 = 0
            n = 0
            for word in line[1:]:
                if word == '':
                    continue

                if i == 2:
                    f1 = float(word)
                elif i == 3:
                    n = int(word)

                i += 1

            results[params]['per'] = (f1, n)

        elif line[0] == 'org':
            i = 0
            f1 = 0
            n = 0

            for word in line[1:]:

                if word == '':
                    continue

                if i == 2:
                    f1 = float(word)
                elif i == 3:
                    n = int(word)

                i += 1

            results[params]['org'] = (f1, n)

        elif line[0] == 'misc':
            i = 0
            f1 = 0
            n = 0
            for word in line[1:]:
                if word == '':
                    continue

                if i == 2:
                    f1 = float(word)
                elif i == 3:
                    n = int(word)

                i += 1

            results[params]['misc'] = (f1, n)

        elif line[0] == 'loc':
            i = 0
            f1 = 0
            n = 0
            for word in line[1:]:
                if word == '':
                    continue

                if i == 2:
                    f1 = float(word)
                elif i == 3:
                    n = int(word)

                i += 1

            results[params]['loc'] = (f1, n)

        elif line[0] == 'weighted':
            i = 0
            f1 = 0
            n = 0
            for word in line[2:]:
                if word == '':
                    continue

                if i == 2:
                    f1 = float(word)
                elif i == 3:
                    n = int(word)

                i += 1

            results[params]['weighted'] = (f1, n)

max = 0
max_params = None

max_weighted = 0
max_params_weighted = None

for params in results:
    f1 = ((results[params]['per'][0] * results[params]['per'][1]) +
          (results[params]['org'][0] * results[params]['org'][1]) +
           (results[params]['loc'][0] * results[params]['loc'][1]) +
           (results[params]['misc'][0] * results[params]['misc'][1])) / (results[params]['per'][1] +
                                                                                   results[params]['org'][1] +
                                                                                   results[params]['loc'][1] +
                                                                                   results[params]['misc'][1])
    if f1 > max:
        max = f1
        max_params = params

    if results[params]['weighted'][0] > max_weighted:
        max_weighted = results[params]['weighted'][0]
        max_params_weighted = params

print(max_weighted)
print(max_params_weighted)

print(max)
print(max_params)

