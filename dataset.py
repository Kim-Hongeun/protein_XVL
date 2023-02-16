from torch.utils.data import Dataset
import random
import pandas as pd

class KIBADataset(Dataset):
    def __init__(self, shuffle=False):
        self.data = pd.read_csv('./data/KIBA.csv').to_numpy()

        if shuffle:
            random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = 'Q' + self.data[:,2][index]
        protSeq = self.data[:,4][index]
        return smiles, protSeq

class DAVISDataset(Dataset):
    def __init__(self, shuffle=False):
        self.data = pd.read_csv('./data/DAVIS.csv').to_numpy()

        if shuffle:
            random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = 'Q' + self.data[:,2][index]
        protSeq = self.data[:,4][index]
        return smiles, protSeq

