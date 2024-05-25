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

    def __init__(self, dataset, max_len, seq_ctxt = True) :
        self.datapoints = []
        self.labels = []
        self.seq_ctxt = seq_ctxt

        for seq in dataset:
            # Will be used for context
            if len(seq.split("?")) == 1 : words_a = seq.split("?")[0]
            else : words_a, words_q = seq.split("?")[0], seq.split("?")[1]

            words = seq.split(" ")

            lbl_words = " ".join(words[-2:])
            lbl_chars = (list(lbl_words) if lbl_words[0][0] != '?' else list(lbl_words)[1:]) + [self.BOQ]

            feat_words = " ".join(words[:-2])
            feat_chars = list(feat_words) + ['?' if lbl_words[0][0] == '?' else '']

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
            
            words_a_id = [self.char_to_id[c] for c in words_a]
            

            # Building features and labels. Contexts using beginning of answer and end of question
            
            ### First sequential version of the context
            if self.seq_ctxt : 
                for i in range(len(lbl_ids)) :
                    ctxt = feat_ids[-max_len + i:] + lbl_ids[:i] if len(lbl_ids[:i]) < max_len else lbl_ids[-max_len:i]
                    self.datapoints.append( [self.char_to_id[self.PADDING_SYMBOL]] * (max_len - len(ctxt)) + ctxt )
                    self.labels.append(lbl_ids[i])

            ### Second version of the context
            else :
                len_q = len(words_q) + 1 # counting the <BOS> token
                for i in range(len(lbl_ids)) :
                    len_q_ctxt = len_q - len(lbl_ids) + i

                    # Sequential context, when short question or short answer
                    if len_q_ctxt <= max_len//2 or len(words_a) < max_len//8 :
                        ctxt = feat_ids[-max_len + i:] + lbl_ids[:i] if len(lbl_ids[:i]) < max_len else lbl_ids[-max_len:i]
                        self.datapoints.append( [self.char_to_id[self.PADDING_SYMBOL]] * (max_len - len(ctxt)) + ctxt )

                    # Keeping the beginning of the question
                    else :
                        ctxt = words_a_id[-max_len//2:] + feat_ids[-max_len//2 + i:] + lbl_ids[:i] if len(lbl_ids[:i]) < max_len//2 else words_a_id[-max_len//2:] + lbl_ids[-max_len//2:i]
                        self.datapoints.append( ctxt + [self.char_to_id[self.PADDING_SYMBOL]] * (max_len - len(ctxt)) )
                    
                    self.labels.append(lbl_ids[i])

                    # Checking that the dimension is correct
                    if len(ctxt + [self.char_to_id[self.PADDING_SYMBOL]] * (max_len - len(ctxt)))!= max_len :
                        print("FEAT LEN", len(ctxt + [self.char_to_id[self.PADDING_SYMBOL]] * (max_len - len(ctxt))))

        # Shuffling the dataset
        combined = list(zip(self.datapoints, self.labels))
        random.shuffle(combined)
        self.datapoints, self.labels = zip(*combined)

    def __len__(self) :
        return len(self.datapoints)

    def __getitem__(self, idx) :
        idx = idx % len(self.datapoints)
        return torch.tensor(self.datapoints[idx]), torch.tensor(self.labels[idx], dtype=torch.long)
    
