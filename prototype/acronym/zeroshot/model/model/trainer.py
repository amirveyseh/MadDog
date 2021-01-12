"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from prototype.acronym.zeroshot.model.model.gcn import GCNClassifier
from prototype.acronym.zeroshot.model.utils import constant, torch_utils

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        opt = checkpoint['config']
        opt['cuda'] = False
        opt['cpu'] = True
        self.opt = opt

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:3]]
        labels = Variable(batch[3].cuda())
    else:
        inputs = [Variable(b) for b in batch[:3]]
        labels = Variable(batch[3])
    return inputs, labels

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, labels = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, label_mask, unsort=True):
        inputs, labels = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[-1]
        # forward
        self.model.eval()
        logits = self.model(inputs)
        label_mask = torch.Tensor([label_mask,label_mask])
        loss = self.criterion(logits, labels)
        probs = F.softmax(F.softmax(logits, 1)*label_mask,1)
        probs = probs.data.cpu().numpy().tolist()
        # predictions = np.argmax((F.softmax(logits, 1)*label_mask).data.cpu().numpy(), axis=1).tolist()
        if unsort:
            # _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
            #         predictions, probs)))]
            _, probs = [list(t) for t in zip(*sorted(zip(orig_idx, probs)))]
        # return predictions, probs, loss.item()
        return probs, loss.item()
