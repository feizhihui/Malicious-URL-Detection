# encoding=utf-8
import numpy as np
import glob
import collections

train_eval_split_line = 0.9
url_char_scope = 58
sequence_lens = 150


class DataMaster(object):
    # ==============
    def __init__(self, train_mode=True):
        train_set = dict()
        train_label = dict()
        test_set = dict()
        test_label = dict()
        for filename in glob.glob("../Data/*_norm.txt"):
            with open(filename, "r") as file:
                subset = []
                print("reading", filename, "~~")
                for line in file.readlines():
                    line = line.split()[0]
                    subset.append(line.lower())
                lengths = len(subset)
                train_set["filename"]= subset[:int(lengths * train_eval_split_line)]
                train_label += [0] * int(lengths * train_eval_split_line)
                test_set += subset[int(lengths * train_eval_split_line):]
                test_label += [0] * (lengths - int(lengths * train_eval_split_line))
                print("total size {}".format(lengths))
                print("training size:{}, test size {}".format(int(lengths * train_eval_split_line),
                                                              lengths - int(lengths * train_eval_split_line)))

        for filename in glob.glob("../Data/*_mal.txt"):
            with open(filename, "r") as file:
                subset = []
                print("reading", filename, "~~")
                for line in file.readlines():
                    line = line.split()[0]
                    subset.append(line.lower())
                lengths = len(subset)
                train_set += subset[:int(lengths * train_eval_split_line)]
                train_label += [1] * int(lengths * train_eval_split_line)
                test_set += subset[int(lengths * train_eval_split_line):]
                test_label += [1] * (lengths - int(lengths * train_eval_split_line))
                print("total size {}".format(lengths))
                print("training size:{}, test size {}".format(int(lengths * train_eval_split_line),
                                                              lengths - int(lengths * train_eval_split_line)))

        url_lens = [len(url) for url in train_set]
        print(max(url_lens))
        print(min(url_lens))
        char_set = []
        for url in train_set:
            # print(url)
            char_set.extend([c for c in url])
        c = collections.Counter(char_set)
        char_set = c.most_common(url_char_scope)
        char_dict = dict()
        for i, (c, num) in enumerate(char_set):
            char_dict[c] = i

        if train_mode:
            self.train_x = np.array(train_set, np.str)
            self.train_y = np.array(train_label)
        else:
            self.train_x = np.array(test_set, np.str)
            self.train_y = np.array(test_label)
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


if __name__ == '__main__':
    DataMaster()
