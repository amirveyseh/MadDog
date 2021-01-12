import json
from tqdm import tqdm
import os
import traceback
import mmap
from prototype.acronym.zeroshot.load_models import ModelLoader

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_glove_vocab(file, wv_dim):
    vocab = {}
    with open(file, encoding='utf8') as f:
        for line in tqdm(f, total=get_num_lines(file)):
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            vocab[token] = list(map(float, elems[-wv_dim:]))
    return vocab

class ZeroShotExtractor():
    def __init__(self):
        with open(dir_path+'/diction.json') as file:
            self.diction = json.load(file)
        with open(dir_path+'/addresses.json') as file:
            self.addresses = json.load(file)
        self.model_loader = ModelLoader()


    def predict(self, sentence, ind):
        acronym_pos = [0]*len(sentence)
        acronym_pos[ind] = 1
        data = {
            'tokens': sentence,
            'acronym_pos': acronym_pos,
        }
        data_file = dir_path+'/data.json'
        with open(data_file, 'w') as file:
            json.dump([data]*2, file)
        ids = self.addresses[sentence[ind]]
        predictions = []
        probablities = []
        for id in ids:
            preds, probs = self.model_loader.predict(id, data_file, set(self.diction[sentence[ind]]))
            predictions += preds
            probablities += probs
        pairs = list(zip(predictions,probablities))
        pairs = sorted(pairs,key=lambda p:p[1],reverse=True)
        prediction = pairs[0][0]
        return prediction, pairs

    def extract(self, sentence):
        pred = {}
        for i, token in enumerate(sentence):
            if token in self.diction:
                try:
                    lf, scores = self.predict(sentence, i)
                    pred[i] = (lf, scores)
                except Exception:
                    print(traceback.format_exc())
        return pred