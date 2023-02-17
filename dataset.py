from torch.utils.data import Dataset
import random
import pandas as pd

class KIBADataset(Dataset):
    def __init__(self, data_path, shuffle=False):
        self.data = pd.read_csv(data_path).to_numpy()

        if shuffle:
            random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = 'Q' + self.data[:,2][index]
        proteinSeq = self.data[:,4][index]
        KIBA_score = self.data[:,5][index]
        return smiles, proteinSeq, KIBA_score

class DAVISDataset(Dataset):
    def __init__(self, data_path, shuffle=False):
        self.data = pd.read_csv(data_path).to_numpy()

        if shuffle:
            random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = 'Q' + self.data[:,2][index]
        proteinSeq = self.data[:,4][index]
        pKd = self.data[:,5][index]
        return smiles, proteinSeq, pKd

