import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def data_process(train_path, val_path, dir, is_train):
    print(is_train)
    f_train = open(train_path, 'w')
    f_val = open(val_path, 'w')
    length = []
    tag_list = set()
    if is_train:
        for i in range(1000):
            query_path = dir + str(i) + '.txt'
            ner_path = dir + str(i) + '.ann'
            query_file = open(query_path, 'r')
            ner_file = open(ner_path, 'r')
            q_list = []
            query = query_file.readline().strip('\n')
            for j in range(len(query)):
                q_list.append(query[j])
            n_list = ['O'] * len(q_list)
            for line in ner_file.readlines():
                line = line.strip('\n').split('\t')
                [tag, start, end] = line[1].split(' ')
                tag_list.add(tag)
                start = int(start)
                end = int(end)
                text = line[2]
                assert text == query[start:end]
                if end - start == 1:
                    n_list[start] = 'S-' + tag
                else:
                    n_list[start] = 'B-' + tag
                    for j in range(start+1, end):
                        n_list[j] = 'I-' + tag
                    n_list[end-1] = 'E-' + tag

            random = np.random.random_integers(0, 9)
            if random > 1:
                fw = f_train
            else:
                fw = f_val
            for j in range(len(q_list)):
                fw.write(q_list[j] + '\t' + n_list[j] + '\n')
            fw.write('\n')
            length.append(len(q_list))
    count = Counter(length)
    print(count.most_common())
    print(tag_list)
    x = []
    y = []
    for item in count.items():
        x.append(item[0])
        y.append(item[1])
    plt.plot(x, y, 'ro')
    plt.show()

train_path = './train.csv'
val_path = './val.csv'
test_path = './test.csv'
# if not os.path.exists(train_path):
data_process(train_path, val_path, './train/', True)
# if not os.path.exists(test_path):
#     data_process(test_path, './test', False)