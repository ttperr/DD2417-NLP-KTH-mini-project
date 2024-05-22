# First run this cell
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import math
import random

class CharDataset(Dataset) :

    # class variables
    char_to_id = {}
    id_to_char = {}
    PADDING_SYMBOL = '<PAD>'
    BOQ = '<BOQ>'

    char_to_id[PADDING_SYMBOL] = 0
    char_to_id[BOQ] = 1
    id_to_char[0] = PADDING_SYMBOL
    id_to_char[1] = BOQ

    def __init__(self, dataset, max_len) :
        self.datapoints = []
        self.labels = []

        for seq in dataset:
            words = seq.split(" ")

            lbl_words = " ".join(words[-2:])
            lbl_chars = [c for c in lbl_words if c != '?'] + [self.BOQ]

            feat_words = " ".join(words[:-2])
            feat_chars = [c for c in feat_words] + ['?' if lbl_words[0][0] == '?' else '']

            # Updating vocabulary
            lbl_ids = []
            feat_ids = []

            for c in lbl_chars:
                if c not in self.char_to_id.keys():
                    self.char_to_id[c] = len(self.id_to_char)
                    self.id_to_char[len(self.id_to_char)] = c
                lbl_ids.append(self.char_to_id[c])

            for c in feat_chars:
                if c not in self.char_to_id.keys():
                    self.char_to_id[c] = len(self.id_to_char)
                    self.id_to_char[len(self.id_to_char)] = c
                feat_ids.append(self.char_to_id[c])
                        
            # Building features and labels
            for i in range(len(lbl_ids)) :
                ctxt = feat_ids[-max_len + i:] + lbl_ids[:i]
                self.datapoints.append( [self.char_to_id[self.PADDING_SYMBOL]] * (max_len - len(ctxt)) + ctxt)
                self.labels.append(lbl_ids[i])

                """# Checking that the dimension is correct
                if len([self.char_to_id[self.PADDING_SYMBOL]] * (max_len - len(ctxt)) + ctxt )!= max_len :
                    print(seq[::-1])"""

        # shuffling the dataset
        combined = list(zip(self.datapoints, self.labels))
        random.shuffle(combined)
        self.datapoints, self.labels = zip(*combined)

    def __len__(self) :
        return len(self.datapoints)

    def __getitem__(self, idx) :
        idx = idx % len(self.datapoints)
        return torch.tensor(self.datapoints[idx]), torch.tensor(self.labels[idx], dtype=torch.long)
    
