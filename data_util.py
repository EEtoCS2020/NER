# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:55
# @Author : Scofield Phil
# @FileName: DataManager.py
# @Project: sequence-lableing-vex

import os, logging
import numpy as np
import jieba
import re, logging, datetime, csv
import pandas as pd
jieba.setLogLevel(logging.INFO)

def read_csv_(file_name, names, delimiter='t'):
    if delimiter=='t':
        sep = "\t"
    elif delimiter=='b':
        sep = " "
    else:
        sep = delimiter

    return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                       names=names)


def save_csv_(df_, file_name, names, delimiter='t'):
    if delimiter == 't':
        sep = "\t"
    elif delimiter == 'b':
        sep = " "
    else:
        sep = delimiter

    df_.to_csv(file_name, quoting=csv.QUOTE_NONE,
               columns=names, sep=sep, header=False,
               index=False)

class DataManager:
    def __init__(self, data_dir, max_sequence_length=64, embedding_dim=200):
        self.UNKNOWN = "<UNK>"
        self.PADDING = "<PAD>"
        self.delimiter='t'
        self.train_file = data_dir + 'train.csv'
        self.dev_file = data_dir + 'dev.csv'
        self.test_file = data_dir + 'test.csv'
        self.labeling_level = 'word'
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim

        self.token2id_file = data_dir + '/vocabs/t2id.txt'
        self.label2id_file = data_dir + '/vocabs/id2t.txt'

        self.token2id, self.id2token, self.label2id, self.id2label = self.loadVocab()

        self.max_token_number = len(self.token2id)
        self.max_label_number = len(self.label2id)

        jieba.load_userdict(self.token2id.keys())

    def loadVocab(self):
        return self.buildVocab(self.train_file)
        token2id = {}
        id2token = {}
        with open(self.token2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                token = row.split('\t')[0]
                token_id = int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token

        label2id = {}
        id2label = {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                label = row.split('\t')[0]
                label_id = int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label

        return token2id, id2token, label2id, id2label


    def buildVocab(self, train_path):
        df_train = read_csv_(train_path, names=["token", "label"])
        tokens = list(set(df_train["token"][df_train["token"].notnull()]))
        labels = list(set(df_train["label"][df_train["label"].notnull()]))
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        id2token[0] = self.PADDING
        id2label[0] = self.PADDING
        token2id[self.PADDING] = 0
        label2id[self.PADDING] = 0
        id2token[len(tokens) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(tokens) + 1

        self.saveVocab(id2token, id2label)

        return token2id, id2token, label2id, id2label

    def saveVocab(self, id2token, id2label):
        with open(self.token2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + "\t" + str(idx) + "\n")
        with open(self.label2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + "\t" + str(idx) + "\n")

    def getEmbedding(self, embed_file):
        emb_matrix = np.random.normal(loc=0.0, scale=0.08, size=(len(self.token2id.keys()), self.embedding_dim))
        emb_matrix[self.token2id[self.PADDING], :] = np.zeros(shape=(self.embedding_dim))

        with open(embed_file, "r", encoding="utf-8") as infile:
            for row in infile:
                row = row.rstrip()
                items = row.split()
                token = items[0]
                assert self.embedding_dim == len(
                    items[1:]), "embedding dim must be consistent with the one in `token_emb_dir'."
                emb_vec = np.array([float(val) for val in items[1:]])
                if token in self.token2id.keys():
                    emb_matrix[self.token2id[token], :] = emb_vec

        return emb_matrix


    def padding(self, sample):
        for i in range(len(sample)):
            if len(sample[i]) < self.max_sequence_length:
                sample[i] += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(sample[i]))]
        return sample

    def prepare(self, chars, tokens, labels, is_padding=True, return_psyduo_label=False):
        X = []
        y = []
        querys = []
        y_psyduo = []
        tmp_x = []
        tmp_y = []
        tmp_q = ''
        tmp_y_psyduo = []

        for record in zip(tokens, labels, chars):
            c = record[0]
            l = record[1]
            ch = record[2]
            if c == -1:  # empty line
                if len(tmp_x) <= self.max_sequence_length:
                    X.append(tmp_x)
                    y.append(tmp_y)
                    querys.append(tmp_q)
                    if return_psyduo_label: y_psyduo.append(tmp_y_psyduo)
                tmp_x = []
                tmp_y = []
                tmp_q = ''
                if return_psyduo_label: tmp_y_psyduo = []
            else:
                tmp_x.append(c)
                tmp_y.append(l)
                tmp_q += ' ' + ch
                if return_psyduo_label: tmp_y_psyduo.append(self.label2id["O"])
        if is_padding:
            X = np.array(self.padding(X))
        else:
            X = np.array(X)
        y = np.array(self.padding(y))
        if return_psyduo_label:
            y_psyduo = np.array(self.padding(y_psyduo))
            return X, y_psyduo

        return X, y, querys

    def getTrainingSet(self, train_val_ratio=0.9):
        df_train = read_csv_(self.train_file, names=["token", "label"])

        # map the token and label into id
        df_train["token_id"] = df_train.token.map(lambda x: -1 if str(x) == str(np.nan) else self.token2id[x])
        df_train["label_id"] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])

        # convert the data in maxtrix
        X, y, querys = self.prepare(df_train['token'], df_train["token_id"], df_train["label_id"], is_padding=False)
        print(len(X))
        print(len(querys))
        # shuffle the samples
        num_samples = len(X)
        indexs = np.arange(num_samples)
        np.random.shuffle(indexs)
        X = X[indexs]
        y = y[indexs]
        querys = [querys[index] for index in indexs]
        if self.dev_file != None:
            X_train = X
            y_train = y
            q_train = querys
            X_val, y_val, q_val = self.getValidingSet()
        else:
            # split the data into train and validation set
            X_train = X[:int(num_samples * train_val_ratio)]
            y_train = y[:int(num_samples * train_val_ratio)]
            q_train = querys[:int(num_samples * train_val_ratio)]
            X_val = X[int(num_samples * train_val_ratio):]
            y_val = y[int(num_samples * train_val_ratio):]
            q_val = querys[int(num_samples * train_val_ratio):]

        return X_train, y_train, q_train, X_val, y_val, q_val

    def getValidingSet(self):
        df_val = read_csv_(self.dev_file, names=["token", "label"])

        df_val["token_id"] = df_val.token.map(lambda x: self.mapFunc(x, self.token2id))
        df_val["label_id"] = df_val.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])

        X_val, y_val, querys = self.prepare(df_val['token'], df_val["token_id"], df_val["label_id"], is_padding=False)
        return X_val, y_val, querys

    def getTestingSet(self):
        df_test = read_csv_(self.test_file, names=None)

        if len(list(df_test.columns)) == 2:
            df_test.columns = ["token", "label"]
            df_test = df_test[["token"]]
        elif len(list(df_test.columns)) == 1:
            df_test.columns = ["token"]

        df_test["token_id"] = df_test.token.map(lambda x: self.mapFunc(x, self.token2id))
        df_test["token"] = df_test.token.map(lambda x: -1 if str(x) == str(np.nan) else x)

        X_test_id, y_test_psyduo_label = self.prepare(df_test["token_id"], df_test["token_id"],
                                                      return_psyduo_label=True)
        X_test_token, _ = self.prepare(df_test["token"], df_test["token"], is_padding=False)
        return X_test_id, y_test_psyduo_label, X_test_token

    def mapFunc(self, x, token2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in token2id:
            return token2id[self.UNKNOWN]
        else:
            return token2id[x]

    def prepare_single_sentence(self, sentence):
        if self.labeling_level == 'word':
            if self.check_contain_chinese(sentence):
                sentence = list(jieba.cut(sentence))
            else:
                sentence = list(sentence.split())
        elif self.labeling_level == 'char':
            sentence = list(sentence)

        gap = self.batch_size - 1

        x_ = []
        y_ = []

        for token in sentence:
            try:
                x_.append(self.token2id[token])
            except:
                x_.append(self.token2id[self.UNKNOWN])
            y_.append(self.label2id["O"])

        if len(x_) < self.max_sequence_length:
            sentence += ['x' for _ in range(self.max_sequence_length - len(sentence))]
            x_ += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(x_))]
            y_ += [self.label2id["O"] for _ in range(self.max_sequence_length - len(y_))]
        elif len(x_) > self.max_sequence_length:
            sentence = sentence[:self.max_sequence_length]
            x_ = x_[:self.max_sequence_length]
            y_ = y_[:self.max_sequence_length]

        X = [x_]
        Sentence = [sentence]
        Y = [y_]
        X += [[0 for j in range(self.max_sequence_length)] for i in range(gap)]
        Sentence += [['x' for j in range(self.max_sequence_length)] for i in range(gap)]
        Y += [[self.label2id['O'] for j in range(self.max_sequence_length)] for i in range(gap)]
        X = np.array(X)
        Sentence = np.array(Sentence)
        Y = np.array(Y)

        return X, Sentence, Y

    def check_contain_chinese(self, check_str):
        for ch in list(check_str):
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False


if __name__ == '__main__':
    dm = DataManager('./data/example_datasets1/')
    querys_train, labels_train, querys_val, labels_val = dm.getTrainingSet(0.2)