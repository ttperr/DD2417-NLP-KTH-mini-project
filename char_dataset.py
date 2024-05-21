# First run this cell
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import math

class CharDataset(Dataset) :

    # class variables
    char_to_id = {}
    id_to_char = {}
    PADDING_SYMBOL = '<PAD>'
    char_to_id[PADDING_SYMBOL] = 0

    def __init__(self, file_path, n) :
        self.datapoints = []
        self.labels = []
        chars = []
        try :
            # First read the dataset to find all the unique characters
            with open(file_path,'r',encoding='utf-8') as f :
                contents = f.read()
            for char in contents:
                if char not in self.char_to_id:
                    self.char_to_id[char] = len(self.id_to_char)
                    self.id_to_char[len(self.id_to_char)] = char
                chars.append( self.char_to_id[char] )
            # Then go through all the chars and chunk them up into datapoints
            k = 0
            while k < len(chars)-n:
                for i in range(1, n+1):
                    self.datapoints.append([c for c in chars[k:i+k]+[0]*(n-i)])
                    self.labels.append(chars[i+k])
                k += n
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def __len__(self) :
        return len(self.datapoints)

    def __getitem__(self, idx) :
        idx = idx % len(self.datapoints)
        return torch.tensor(self.datapoints[idx]), torch.tensor(self.labels[idx], dtype=torch.long)