# encoding=utf-8
import numpy as np
import glob
import collections

train_eval_split_line = 0.9
url_char_scope = 58
sequence_lens = 150


class DataMaster(object):
    # ==============
    def __init__(self, train_mode=True, test_file=""):
        train_set = dict()
        train_label = dict()
        test_set = dict()
        test_label = dict()
        for filename in glob.glob("../Data/*_norm.txt"):
            filemark = filename.split("\\")[1]
            with open(filename, "r") as file:
                subset = []
                print("reading", filename, "~~")
                for line in file.readlines():
                    line = line.split()[0]
                    subset.append(line.lower())
                lengths = len(subset)
                train_set[filemark] = subset[:int(lengths * train_eval_split_line)]
                train_label[filemark] = [0] * int(lengths * train_eval_split_line)
                test_set[filemark] = subset[int(lengths * train_eval_split_line):]
                test_label[filemark] = [0] * (lengths - int(lengths * train_eval_split_line))
                print("total size {}".format(lengths))
                print("training size:{}, test size {}".format(int(lengths * train_eval_split_line),
                                                              lengths - int(lengths * train_eval_split_line)))

        for filename in glob.glob("../Data/*_mal.txt"):
            filemark = filename.split("\\")[1]
            with open(filename, "r") as file:
                subset = []
                print("reading", filename, "~~")
                for line in file.readlines():
                    line = line.split()[0]
                    subset.append(line.lower())
                lengths = len(subset)
                train_set[filemark] = subset[:int(lengths * train_eval_split_line)]
                train_label[filemark] = [1] * int(lengths * train_eval_split_line)
                test_set[filemark] = subset[int(lengths * train_eval_split_line):]
                test_label[filemark] = [1] * (lengths - int(lengths * train_eval_split_line))
                print("total size {}".format(lengths))
                print("training size:{}, test size {}".format(int(lengths * train_eval_split_line),
                                                              lengths - int(lengths * train_eval_split_line)))

        url_lens = []
        char_set = []
        for filemark in train_set:
            for url in train_set[filemark]:
                # print(url)
                char_set.extend([c for c in url])
                url_lens.append(len(url))
        print("max url length:", max(url_lens))
        print("min url length:", min(url_lens))
        c = collections.Counter(char_set)
        char_set = c.most_common(url_char_scope)
        char_dict = dict()
        for i, (c, num) in enumerate(char_set):
            char_dict[c] = i

        if train_mode:
            self.train_x, self.train_y = self.compose_data(train_set, train_label)
        else:
            if test_file not in test_set:
                print("test all set")
                self.train_x, self.train_y = self.compose_data(test_set, test_label)
            else:
                print("test only", filemark)
                self.train_x = np.array(test_set[test_file], np.str)
                self.train_y = np.array(test_label[test_file])
        self.datasize = len(self.train_y)
        self.char_dict = char_dict

    def shuffle(self):
        mark = list(range(self.datasize))
        np.random.shuffle(mark)
        self.train_x = self.train_x[mark]
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

    def compose_data(self, train_set, train_label):
        train_x = []
        train_y = []
        for filemark in train_set:
            train_x.extend(train_set[filemark])
            train_y.extend(train_label[filemark])
        return np.array(train_x, np.str), np.array(train_y)


if __name__ == '__main__':
    DataMaster()
