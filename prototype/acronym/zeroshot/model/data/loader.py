"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
# import pymysql
import spacy
from collections import defaultdict
from tqdm import tqdm
import pickle

from prototype.acronym.zeroshot.model.utils import constant, helper, vocab

# nlp = spacy.load("en_core_web_sm")
#
#
# db = pymysql.connect("ilcompn0", "amir1", "Welcome@12345678", "acronym")
# cursor = db.cursor(pymysql.cursors.DictCursor)
# cursor.execute(
#     'select * from text_with_sense_precision_joined limit 860000,10000;')
# data = cursor.fetchall()
#
# label_group = defaultdict(list)
# for d in tqdm(data):
#     label_group[d['long_form']].append(d)
#
# train = []
# dev = []
# test = []
#
# for lf, ds in label_group.items():
#     train.extend(ds[:int(len(ds)*0.8)])
#     dev.extend(ds[int(len(ds)*0.8):int(len(ds)*0.9)])
#     test.extend(ds[int(len(ds)*0.9):])

# number of samples: 48231283
# train:

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, label2id={}):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation

        if len(label2id) == 0:
            with open('/'.join(filename.split('/')[:-1])+'/long_forms.pkl', 'rb') as file:
                long_forms = pickle.load(file)
            labels = {}
            for l in long_forms:
                if l not in labels:
                    labels[l] = len(labels)
            # self.label2id = constant.LABEL_TO_ID
            self.label2id = labels
        else:
            self.label2id = label2id

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt, evaluation)
        print(filename+' : '+str(len(data)))

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt, evaluation):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in tqdm(data):
            # tokens = [t.text for t in nlp(d['paragraph'])]
            tokens = list(d['tokens'])
            # if len(tokens) > 200:
            #     continue
            tokens = map_to_ids(tokens, vocab.word2id)
            # acronym = get_positions(d['acronym'], d['acronym'], len(tokens))
            acronym = d['acronym_pos']
            # acronym = [1 if t == d['short_form'] else 0 for t in tokens]
            if not evaluation:
                expansion = self.label2id[d['long_form']]
            else:
                expansion = 0
            processed += [(tokens, acronym, expansion)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 3

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        acronym = get_long_tensor(batch[1], batch_size)

        exps = torch.LongTensor(batch[2])

        return (words, masks, acronym, exps, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

