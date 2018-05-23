# encoding=utf-8
import numpy as np
import glob
import collections
import pickle

train_eval_split_line = 0.9
url_char_scope = 58
sequence_lens = 150
feature_size = 9


class DataMaster(object):
    # ==============
    def __init__(self, train_mode=True):
        if train_mode:
            benigh_file = r"../Benchmark/Benign_Train.txt"
            malicious_file = r"../Benchmark/Malicious_Train.txt"
        else:
            benigh_file = r"../Benchmark/Benign_Test.txt"
            malicious_file = r"../Benchmark/Malicious_Test.txt"

        data_set = []
        data_emb = []
        data_label = []

        with open(benigh_file, "r", errors='ignore') as file:
            subset = []
            subemb = []
            print("reading", benigh_file, "~~")
            for line in file.readlines():
                line = line.split()
                url = line[0]
                emb = line[1:-1]
                if len(emb) != feature_size:
                    continue
                subset.append(url.lower())
                subemb.append(emb)
            lengths = len(subset)
            data_set.extend(subset)
            data_emb.extend(subemb)
            data_label.extend([0] * lengths)

        with open(malicious_file, "r", errors='ignore') as file:
            subset = []
            subemb = []
            print("reading", malicious_file, "~~")
            for line in file.readlines():
                line = line.split()
                url = line[0]
                emb = line[1:-1]
                if len(emb) != feature_size:
                    continue
                subemb.append(emb)
                subset.append(url.lower())
            lengths = len(subset)
            data_set.extend(subset)
            data_emb.extend(subemb)
            data_label.extend([1] * lengths)

        url_lens = []
        char_set = []
        for url in data_set:
            char_set.extend([c for c in url])
            url_lens.append(len(url))
        print("max url length:", max(url_lens))
        print("min url length:", min(url_lens))

        with open('../Cache/char_dict.pkl', 'rb') as file:
            char_dict = pickle.load(file)

        print(char_dict)

        self.train_x, self.train_e, self.train_y = \
            np.array(data_set, np.object), np.array(data_emb, np.float32), np.array(data_label, np.int32)

        self.datasize = len(self.train_y)
        self.char_dict = char_dict

    def shuffle(self):
        mark = list(range(self.datasize))
        np.random.shuffle(mark)
        self.train_x = self.train_x[mark]
        self.train_e = self.train_e[mark]
        self.train_y = self.train_y[mark]

    def mapping(self, train_batch):
        new_batch = np.zeros(shape=(len(train_batch), sequence_lens))
        batch_lens = []
        for i, url in enumerate(train_batch):
            if len(url) <= sequence_lens:
                batch_lens.append(len(url))  # endswith zeros
            else:
                batch_lens.append(sequence_lens)
                url = url[:sequence_lens]

            for j, char in enumerate(url):
                new_batch[i][j] = self.char_dict.get(char, 0)

        return new_batch, np.array(batch_lens, np.int32)


if __name__ == '__main__':
    DataMaster()
