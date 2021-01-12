import random
import torch
import os
from tqdm import tqdm
import json
from scipy.special import softmax

from prototype.acronym.zeroshot.model.data.loader import DataLoader
from prototype.acronym.zeroshot.model.model.trainer import GCNTrainer
from prototype.acronym.zeroshot.model.utils import torch_utils
from prototype.acronym.zeroshot.model.utils.vocab import Vocab

dir_path = os.path.dirname(os.path.realpath(__file__))

class ModelLoader():
    def __init__(self):
        torch.manual_seed(1234)
        random.seed(1234)
        self.trainers = {}
        self.vocabs = {}
        self.opts = {}
        self.labels = {}

        for i in tqdm(range(483)):
            i = str(i)
            try:
                # load labels
                with open(dir_path+'/../../../saved_models100/100k_'+str(i) + '/labels.json') as file:
                    self.labels[i] = json.load(file)

                # load opt
                model_file = dir_path + '/../../../saved_models100/100k_'+str(i)+'/best_model.pt'
                print("Loading model from {}".format(model_file))
                opt = torch_utils.load_config(model_file)
                opt['cuda'] = False
                opt['cpu'] = True
                trainer = GCNTrainer(opt)
                trainer.load(model_file)

                self.opts[i] = opt
                self.trainers[i] = trainer

                # load vocab
                vocab_file = dir_path + '/../../../saved_models100/100k_'+str(i) + '/vocab.pkl'
                vocab = Vocab(vocab_file, load=True)

                self.vocabs[i] = vocab
            except Exception as e:
                print(e)
                pass

    def predict(self, id, data, valid_labels):
        if id in self.trainers:
            trainer = self.trainers[id]
            opt = self.opts[id]
            vocab = self.vocabs[id]
            label2id = self.labels[id]
            batch = DataLoader(data, opt['batch_size'], opt, vocab, evaluation=True, label2id=label2id)

            id2label = dict([(v, k) for k, v in label2id.items()])
            label_mask = []
            for k in label2id:
                if k in valid_labels:
                    label_mask += [1]
                else:
                    label_mask += [0]
            for b in batch:
                probs, _ = trainer.predict(b,label_mask)
                break

            scores = []
            predictions = []
            for i,lm in enumerate(label_mask):
                if lm == 1:
                    predictions += [id2label[i]]
                    scores += [probs[0][i]]

            scores = softmax(scores).tolist()

            return predictions, scores
        else:
            return '', 0



