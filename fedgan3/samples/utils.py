import math
import numpy as np
import pickle
import gzip
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from molecular_dataset import *
import copy
import torch
import tensorflow as tf
from itertools import chain, zip_longest
# tf.compat.v1.disable_eager_execution()

from sklearn.metrics import classification_report as classification_report
from sklearn.metrics import confusion_matrix
import trainer 
from rdkit.Chem.rdMolDescriptors import CalcMolFormula



class MolecularMetrics(object):

    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda(x):
        return x is not None and Chem.MolToSmiles(x) != ''

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ''
        return x is not None and '*' not in s and '.' not in s and s != ''

    @staticmethod
    def valid_scores(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda_special, mols)), dtype=np.float32)

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean()

    @staticmethod
    def novel_scores(mols, data):
        return np.array(
            list(map(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols)))

    @staticmethod
    def novel_filter(mols, data):
        return list(filter(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols))

    @staticmethod
    def novel_total_score(mols, data):
        return MolecularMetrics.novel_scores(MolecularMetrics.valid_filter(mols), data).mean()

    @staticmethod
    def unique_scores(mols):
        smiles = list(map(lambda x: Chem.MolToSmiles(x) if MolecularMetrics.valid_lambda(x) else '', mols))
        return np.clip(
            0.75 + np.array(list(map(lambda x: 1 / smiles.count(x) if x != '' else 0, smiles)), dtype=np.float32), 0, 1)

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: Chem.MolToSmiles(x), v))
        return 0 if len(v) == 0 else len(s) / len(v)
    
    # @staticmethod
    # def novel_and_unique_total_score(mols, data):
    #     return ((MolecularMetrics.unique_scores(mols) == 1).astype(float) * MolecularMetrics.novel_scores(mols,
    #                                                                                                       data)).sum()
    
    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        return np.array(list(map(lambda x: 0 if x is None else x, [
            MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None for mol in
            mols])))

    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        scores = [MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
                  for mol in mols]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def diversity_scores(mols, data):
        rand_mols = np.random.choice(data.data, 100)
        #get a harsed Morgan fingerprint of molecules
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

        scores = np.array(
            list(map(lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0, mols)))
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        #compute similarity scores between ref fps and target fps
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True) 
        score = np.mean(dist)
        return score

    @staticmethod
    def reconstruction_scores(data, batch_size=10, sample=False):  
            m0, _, _, a, x, _, f, _, _ = data.next_validation_batch()

            n, e = trainer.nodes_hard, trainer.edges_hard

            m1 = [data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) for n_, e_ in zip(n, e)]

            return np.mean([float(Chem.MolToSmiles(m0_) == Chem.MolToSmiles(m1_)) if m1_ is not None else 0
                    for m0_, m1_ in zip(m0, m1)])

    @staticmethod
    def classification_report(data, batch_size=10):
        _, _, _, a, x, _, f, _, _ = data.next_validation_batch()
        
        n, e = trainer.nodes_hat, trainer.edges_hat
        # print(n, e)
        # print(n.shape, e.shape)

        # m1 = [data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) for n_, e_ in zip(n, e)]
        # m1 = [data.matrices2mol(n_.data.cpu().numpy(), strict=True) for n_, in zip(n, e)]
        # print(m1)

        e, n = torch.max(e, -1)[1], torch.max(n, -1)[1]
        e = e.detach().cpu().numpy()
        n = n.detach().cpu().numpy()
        # print(type(e))

        y_true = a.flatten()
        # print(y_true)
        # print(y_true.shape)
        y_pred = e.flatten()
        # print(y_pred.type)
        # print(y_pred.shape)
        target_names = [str(Chem.rdchem.BondType.values[int(e)]) for e in data.bond_decoder_m.values()]
        # print(type(target_names))

        print('######## Classification Report ########\n')
        print(classification_report(y_true, y_pred, labels=list(range(len(target_names))), target_names=target_names))

        print('######## Confusion Matrix ########\n')
        print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

        y_true = x.flatten()
        y_pred = n.flatten()
        target_names = [Chem.Atom(e).GetSymbol() for e in data.atom_decoder_m.values()]
        # print(type(target_names))

        print('######## Classification Report ########\n')
        print(classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                    target_names=target_names))

        print('\n######## Confusion Matrix ########\n')
        print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

    @staticmethod
    def reconstructions(data, batch_dim=100, sample=False):
            m0, _, _, a, x, _, f, _, _ = data.next_train_batch(batch_dim)
                                    
            n, e = trainer.nodes_hard, trainer.edges_hard

            m1 = np.array([e if e is not None else Chem.RWMol() for e in [data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) 
                                                                        for n_, e_ in zip(n, e)]])
            m1 = m1[:100]
            
            # mols = np.vstack((m0, m1)).T.flatten()
            mols = m1
            # mols = [mol for mol in chain(*zip_longest(m0, m1)) if mol is not None]
            return mols

    # @staticmethod
    # def similarity_scores(mols, data):
    #     rand_mols = np.random.choice(data.data, 100)
    #     fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]
    #     # print(fps)
    #     gen_fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) if mol is not None else 0 for mol in mols]
    #     # print(gen_fps)
    #     #compute similarity scores between generated fps and target fps
    #     scores = [DataStructs.BulkTanimotoSimilarity(i, fps) for i in gen_fps] 
    #     # similarity = np.mean(scores)
    #     return scores

    @staticmethod
    def similarity_scores(mols, data):
        rand_mols = np.random.choice(data.data, 100)
        mol = next((mol for mol in mols if mol is not None), np.random.choice(mols))
        # print(mol)
        #get a harsed Morgan fingerprint of molecules
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

        if mol is None:
            scores = 0
        else:
            scores = MolecularMetrics.__compute_similarity(mol, fps)
        # scores = np.array(list(map(lambda x: MolecularMetrics.__compute_similarity(x, fps) if x is not None else 0, mol)))
        # scores = [MolecularMetrics.__compute_similarity(x, fps) if x is not None else 0 for x in mols]
        # print(scores)
        return np.mean(scores)

    @staticmethod
    def __compute_similarity(mol, fps):
        gen_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        #compute similarity scores between ref fps and target fps
        score = DataStructs.BulkTanimotoSimilarity(gen_fps, fps) 
        return score

def mols2grid_image(mols, molsPerRow):
    mols = [m if m is not None else Chem.RWMol() for m in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(50,50))

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def samples(data, model, session, embeddings, sample=False):
    n, e = session.run([model.nodes_hard, model.edges_hard], feed_dict={
        model.embeddings: embeddings, model.training: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    return mols

def report(data):

    rep = MolecularMetrics.classification_report(data)
    
    return rep 

def reconstructed_mols(data, sample=False):

    mols = MolecularMetrics.reconstructions(data, sample=True)

    return mols


def all_scores(mols, data, sample=False, norm=False, reconstruction=False):
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'QED score': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
        'logP score': MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
        'diversity score': MolecularMetrics.diversity_scores(mols, data)}.items()}


    m1 = {'similarity score': MolecularMetrics.similarity_scores(mols, data),
        'valid score': MolecularMetrics.valid_total_score(mols) * 100,
          'unique score': MolecularMetrics.unique_total_score(mols) * 100,
          'novel score': MolecularMetrics.novel_total_score(mols, data) * 100}

    return m0, m1
