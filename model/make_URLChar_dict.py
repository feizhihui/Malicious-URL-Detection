# encoding=utf-8

import pickle

char_list = []
with open('../Data/URL_char.txt', "r") as file:
    for line in file.readlines():
        c = line[1]
        char_list.append(c)

char_list.sort()

print(len(char_list))
print(char_list)
char_dict = {c: i + 1 for i, c in enumerate(char_list)}

with open('../Cache/char_dict.pkl', 'wb') as file:
    pickle.dump(char_dict, file)
