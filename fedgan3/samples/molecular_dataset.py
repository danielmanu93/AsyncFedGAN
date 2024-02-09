import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from datetime import datetime
import time
import itertools

class MolecularDataset():

    def load(self, filename, subset=1):

        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        self.train_idx = np.random.choice(self.train_idx, int(len(self.train_idx) * subset), replace=False)
        self.validation_idx = np.random.choice(self.validation_idx, int(len(self.validation_idx) * subset),
                                               replace=False)
        self.test_idx = np.random.choice(self.test_idx, int(len(self.test_idx) * subset), replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def generate(self, filename, add_h=False, filters=lambda x: True, size=None, validation=0.1, test=0.1):
        self.log('Extracting {}..'.format(filename))


        if filename.endswith('.pkl'):
            self.data = [Chem.MolFromSmiles(line) for line in pickle.load(open(filename, 'rb'))]
        
        self.data = list(map(Chem.AddHs, self.data)) if add_h else self.data   #add H's to molecule and map
        self.data = list(filter(filters, self.data))
        self.data = self.data[:size]
        

        self.log('Extracted {} out of {} molecules {}adding Hydrogen!'.format(len(self.data),
                                                                              len(Chem.SDMolSupplier(filename)),
                                                                              '' if add_h else 'not '))
        
        self._generate_encoders_decoders()
        self._generate_AX()

        # contains all molecules stored as rdkit.Chem objects
        self.data = np.array(self.data)
        
        # contains all molecules stored as SMILES strings
        self.smiles = np.array(self.smiles)
        
        # (N, L) matrix where N is the length of the dataset and each L-dim vector contains the 
        # indices corresponding to a SMILE sequences with padding w.r.t the max length of the longest 
        # SMILES sequence in the dataset
        self.data_S = np.stack(self.data_S)
        
        # (N, 9, 9) tensor where N is the length of the dataset and each 9x9 matrix contains the 
        # indices of the positions of the ones in the one-hot representation of the adjacency tensor
        self.data_A = np.stack(self.data_A)
        
        # (N, 9) matrix where N is the length of the dataset and each 9-dim vector contains the 
        # indices of the positions of the ones in the one-hot representation of the annotation matrix
        self.data_X = np.stack(self.data_X)
        
        # (N, 9) matrix where N is the length of the dataset and each 9-dim vector contains the 
        # diagonal of the correspondent adjacency matrix
        self.data_D = np.stack(self.data_D)
        
        # a (N, F) matrix where N is the length of the dataset and each F vector contains features 
        # of the correspondent molecule
        self.data_F = np.stack(self.data_F)
        
        # (N, 9) matrix where N is the length of the dataset and each 9-dim vector contains the
        # eigenvalues of the correspondent Laplacian matrix
        self.data_Le = np.stack(self.data_Le)
        
        # (N, 9, 9) matrix where N is the length of the dataset and each 9x9 matrix contains the 
        # eigenvectors of the correspondent Laplacian matrix
        self.data_Lv = np.stack(self.data_Lv)

        self.vertexes = self.data_F.shape[-2]
        self.features = self.data_F.shape[-1]

        self._generate_train_validation_test(validation, test)
    
    def _generate_encoders_decoders(self):
        self.log('Creating atoms encoder and decoder..')
        atom_labels = sorted(set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0]))
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        self.log('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
            self.atom_num_types))

        self.log('Creating bonds encoder and decoder..')
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()
                                                                    for mol in self.data
                                                                    for bond in mol.GetBonds())))

        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        self.log('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
            self.bond_num_types - 1))

        self.log('Creating SMILES encoder and decoder..')
        smiles_labels = ['E'] + list(set(c for mol in self.data for c in Chem.MolToSmiles(mol)))
        self.smiles_encoder_m = {l: i for i, l in enumerate(smiles_labels)}
        self.smiles_decoder_m = {i: l for i, l in enumerate(smiles_labels)}
        self.smiles_num_types = len(smiles_labels)
        self.log('Created SMILES encoder and decoder with {} types and 1 PAD symbol!'.format(
            self.smiles_num_types - 1))
        
    def _generate_AX(self):
        self.log('Creating features and adjacency matrices..')

        pb = tqdm(len(self.data))

        data = []
        smiles = []
        data_S = []
        data_A = []
        data_X = []
        data_D = []
        data_F = []
        data_Le = []
        data_Lv = []

        max_length = max(mol.GetNumAtoms() for mol in self.data)
        max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in self.data)  #max len of smiles

        for i, mol in enumerate(self.data):
            A = self._genA(mol, connected=True, max_length=max_length)
            D = np.count_nonzero(A, -1)
            if A is not None:
                data.append(mol)
                smiles.append(Chem.MolToSmiles(mol))
                data_S.append(self._genS(mol, max_length=max_length_s))
                data_A.append(A)
                data_X.append(self._genX(mol, max_length=max_length))
                data_D.append(D)
                data_F.append(self._genF(mol, max_length=max_length))

                L = np.diag(D) - A   #Diag mat(non-zero counts in adjacency tensor) - adjacency tensor
                Le, Lv = np.linalg.eigh(L) #get eig values and vectors

                data_Le.append(Le)
                data_Lv.append(Lv)

            pb.update(i + 1)

        self.log(date=False)
        self.log('Created {} features and adjacency matrices  out of {} molecules!'.format(len(data),
                                                                                           len(self.data)))

        self.data = data
        self.smiles = smiles
        self.data_S = data_S
        self.data_A = data_A
        self.data_X = data_X
        self.data_D = data_D
        self.data_F = data_F
        self.data_Le = data_Le
        self.data_Lv = data_Lv
        self.__len = len(self.data)


    def _genA(self, mol, connected=True, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        #get atom index of end atom in the bond
        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] + [0] * (
                    max_length - mol.GetNumAtoms()), dtype=np.int32)

    def _genS(self, mol, max_length=None):

        max_length = max_length if max_length is not None else len(Chem.MolToSmiles(mol))

        return np.array([self.smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)] + [self.smiles_encoder_m['E']] * (
                    max_length - len(Chem.MolToSmiles(mol))), dtype=np.int32)

    def _genF(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()
        
        features = np.array([[*[a.GetDegree() == i for i in range(5)], # T = 5 ---> get no. of bonded neighbors in the graph
                              *[a.GetExplicitValence() == i for i in range(10)], # N = 9 ---> get explicit valence (including Hs) of this atom
                              *[int(a.GetHybridization()) == i for i in range(1, 7)], #get the hybridization 
                              *[a.GetImplicitValence() == i for i in range(10)],  # get implicit valence of this atom
                              a.GetIsAromatic(),
                              a.GetNoImplicit(),
                              *[a.GetNumExplicitHs() == i for i in range(5)], # Y = 5 ---> get no. of explicit Hs
                              *[a.GetNumImplicitHs() == i for i in range(5)], # Y = 5 ---> get no. of implicit Hs the atom is bonded to
                              *[a.GetNumRadicalElectrons() == i for i in range(5)], #get no. of radical electron of the atom
                              a.IsInRing(),
                              *[a.IsInRingSize(i) for i in range(2, 10)]] for a in mol.GetAtoms()], dtype=np.int32)

        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))
    
    def matrices2mol(self, node_labels, edge_labels, strict=False):
        mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label])) #add atoms and returns the bonded atom

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end), self.bond_decoder_m[edge_labels[start, end]])

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def seq2mol(self, seq, strict=False):
        mol = Chem.MolFromSmiles(''.join([self.smiles_decoder_m[e] for e in seq if e != 0]))

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def _generate_train_validation_test(self, validation, test):

        # self.log('Creating train, validation and test sets..')

        validation = int(validation * len(self))
        test = int(test * len(self))
        train = len(self) - validation - test

        self.all_idx = np.random.permutation(len(self))
        self.train_idx = self.all_idx[0:train]
        self.validation_idx = self.all_idx[train:train + validation]
        self.test_idx = self.all_idx[train + validation:]

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        self.train_count = train
        self.validation_count = validation
        self.test_count = test

        # self.log('Created train ({} items), validation ({} items) and test ({} items) sets!'.format(
        #     train, validation, test))

    def _next_batch(self, counter, count, idx, batch_size):
        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            output = [obj[idx[counter:counter + batch_size]]
                      for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                  self.data_D, self.data_F, self.data_Le, self.data_Lv)]

            counter += batch_size
        else:
            output = [obj[idx] for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                           self.data_D, self.data_F, self.data_Le, self.data_Lv)]

        return [counter] + output
    
    def next_train_batch(self, batch_size=None):
        out = self._next_batch(counter=self.train_counter, count=self.train_count,
                               idx=self.train_idx, batch_size=batch_size)
        self.train_counter = out[0]

        return out[1:]

    def next_validation_batch(self, batch_size=None):
        out = self._next_batch(counter=self.validation_counter, count=self.validation_count,
                               idx=self.validation_idx, batch_size=batch_size)
        self.validation_counter = out[0]

        return out[1:]

    def next_test_batch(self, batch_size=None):
        out = self._next_batch(counter=self.test_counter, count=self.test_count,
                               idx=self.test_idx, batch_size=batch_size)
        self.test_counter = out[0]

        return out[1:]

    @staticmethod
    def log(msg='', date=True):
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' ' + str(msg) if date else str(msg))

    def __len__(self):
        return self.__len

def data_iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                            replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def data_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from dataset
    :param dataset:
    :param num_users:
    :return:
    """
    beta = 0.1
    mol_data = MolecularDataset()
    mol_data.generate('C:\\Users\\danie\\OneDrive\\Desktop\\fedgan\\data_smiles\\qm9_smiles.pkl', validation=0.1, test=0.1)
    data = mol_data.data
    atom_labels = sorted(set([atom.GetAtomicNum() for mol in data for atom in mol.GetAtoms()] + [0]))
    atom_labels = np.array(atom_labels)
    # print(atom_labels)
    all_idxs = mol_data.all_idx
    # print(len(all_idxs))
    all_labels = []
    for idx in range(len(all_idxs)):
        for label in atom_labels:
            all_labels.append(label)
    # print(all_labels)
    all_labels = np.array(all_labels)
    train_idxs = mol_data.train_idx
    # print(len(train_idxs))
    # print(train_idxs)
    train_idxs = train_idxs.astype(int)
    train_atom_labels = all_labels[train_idxs]
    # print(train_atom_labels)
    # 903 training mols -->  180 mols/shard X 5 shards
    num_shards, num_mols = 3, 35702
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_mols)
    labels = np.array(train_atom_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    # print(idxs_labels)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # print(idxs_labels)
    idxs = idxs_labels[0, :]
    # print(idxs)

    # label_new = np.unique(labels)
    # print(label_new)
    # label_cat = len(np.unique(labels))
    # print(label_cat)
    # dirichlet_params = np.repeat(beta, num_users)
    # print(dirichlet_params)
    # proportions = np.random.dirichlet(dirichlet_params, size=num_users)
    # # print(proportions)
    # proportions = proportions / proportions.sum()
    # # print(proportions)
    # proportions = np.cumsum(proportions)
    # # print(proportions)
    # props = [round(i, 2) for i in proportions]
    # # print(props)
    # # props = [prop for prop in props if prop not in (0.0, 1.0)]
    # # print(props)
    # new_props = np.unique(props)
    # print(new_props)

    # clients_distrib = []
    # for vals in itertools.combinations(props, num_users):
    #         if 1 <= sum(vals) <= 2:
    #             clients_distrib.append(vals)
    
    # print(clients_distrib)
    # clients_distrib = np.unique(clients_distrib, axis=0)   
    # print(clients_distrib)

    # add = 0
    # new_prop = []
    # for row in range(len(clients_distrib)):
    #     for col in range(len(clients_distrib[0])):
    #         add += clients_distrib[row][col]

    #     if add <= 2:
    #         new_prop.append(clients_distrib[row])

    #     add = 0

    # new_prop = np.append(new_prop, 1)
    # new_prop = np.insert(new_prop, 0, 0) 
    # new_prop = np.unique(new_prop)
    # print(new_prop)
    # print(new_prop[0])

    # print(data[train_idxs][label_new[0]])
    # proportions = proportions * len(data[train_idxs][label_new[0]])

    # divide and assign 1 shards/client
    
    # for j in range(num_users):
    #     rand_set = set(np.random.choice(new_props[1:], 1, replace=False))
    #     print(rand_set)
    #     ind = np.where(new_props == list(rand_set))
    #     print(ind)
    #     # next_prop = new_props[ind[0][0] + 1]
    #     prev = new_props[ind[0][0]-1]
    #     for rand in rand_set:
    #         print(idxs[int(prev*num_mols)])
    #         print(idxs[int(rand*num_mols)])
    #         # print(idxs[int(next_prop*num_mols)])
    #         dict_users[j] = np.concatenate((dict_users[j], idxs[int(prev*num_mols):int(rand*num_mols)]), axis=0)
    #         # dict_users[j] = np.concatenate((dict_users[j], idxs[int(rand*num_mols):int(next_prop*num_mols)]), axis=0)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_mols:(rand+1)*num_mols]), axis=0)
    return dict_users


if __name__ == '__main__':

    data = MolecularDataset()
    data.generate('C:\\Users\\danie\\OneDrive\\Desktop\\fedgan\\data_smiles\\qm9_smiles.pkl', validation=0.1, test=0.1)
    data.save('C:\\Users\\danie\\OneDrive\\Desktop\\fedgan\\data_smiles\\qm9.dataset')