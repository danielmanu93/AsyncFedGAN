from PIL import Image
import torch
import os
import random

from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from molecular_dataset import data_iid, data_noniid
from molecular_dataset import MolecularDataset
# from molecular_dataset import *

class Molecular(data.Dataset):
    """Dataset class for the Molecular dataset"""
    
    def __init__(self, data_dir):
        self.data = MolecularDataset()
        self.data.load(data_dir)

    def __getitem__(self, index):
        """Return one molecule and its corresponding attribute label"""

        return index, self.data.data[index], self.data.smiles[index],\
               self.data.data_S[index], self.data.data_A[index],\
               self.data.data_X[index], self.data.data_D[index],\
               self.data.data_F[index], self.data.data_Le[index],\
               self.data.data_Lv[index]

    def __len__(self):
        """Return the number of molecules"""
        return len(str(self.data))
        print(len(str(self.data)))
    
# def get_loader(image_dir, batch_size, mode, num_workers=1):
def get_loader(args):
    """Build and return a data loader."""

    # image_dir = '/Users/daniel/Desktop/PhD materials/Fed-GNN-GAN/fedgan/data_smiles/esol_smiles.pkl'
    # image_dir = '/Users/daniel/Desktop/PhD materials/Fed-GNN-GAN/fedgan/data_smiles/esol.dataset'
    num_workers = 1
    dataset = Molecular(args.mol_data_dir)

    train_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    test_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    if args.data_iid:
            # Sample IID user data from dataset
        user_groups = data_iid(train_loader, args.num_users)
    else:
        # Chose equal splits for every user
        user_groups = data_noniid(train_loader, args.num_users)
            
    return train_loader, test_loader, user_groups
