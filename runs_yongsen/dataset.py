from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from DeconT.preprocessing import load_process_SHAC


class Textdataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels[:, None]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class SHAC(Dataset):
    def __init__(self, root='/edata/xiruod/SocialDeterminants_SHAC_n2c2_2022/n2c2_sdoh_challenge'):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        shac = load_process_SHAC(replaceNA='all', base_dir=root)
        self.train, self.val, self.test = shac[shac.set ==
                                               'train'], shac[shac.set == 'dev'], shac[shac.set == 'test']
