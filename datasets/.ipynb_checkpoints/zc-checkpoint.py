from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pickle
from IPython import embed

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
from torch_geometric.nn import radius_graph

from torch.utils.data import Dataset


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

# atomrefs = {
#     6: [0., 0., 0., 0., 0.],
#     7: [
#         -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
#         -2713.48485589
#     ],
#     8: [
#         -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
#         -2713.44632457
#     ],
#     9: [
#         -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
#         -2713.42063702
#     ],
#     10: [
#         -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
#         -2713.88796536
#     ],
#     11: [0., 0., 0., 0., 0.],
# }

# atomrefs = torch.tensor([-376.6024, -23895.4004])÷
# H, C, N, O, F
# atomrefs = torch.tensor([-25038.3340, -17794.9316, -3831.4836, -3111.9292, -223.1435, 0.])
# atomrefs = torch.tensor([-24899, -17621, -3794.4, -3081.9, -220.95, -89.642, -513.99, -151.80, -28.369, -7.0687])
# ensor([[-2.4911e+04],
#         [-1.7658e+04],
#         [-3.8025e+03],
#         [-3.0884e+03],
#         [-2.2142e+02],
#         [-8.9834e+01],
#         [-5.1506e+02],
#         [-1.5213e+02],
#         [-2.8430e+01],
#         [-7.0837e+00],
#         [-1.9415e+03]])

# base_energy = -1937.4

atomrefs = torch.tensor([-3.77565496e+02, -2.38962932e+04, -3.43331046e+04, -4.71717522e+04,
        -6.26078485e+04, -2.14190975e+05, -2.49792706e+05, -2.88701309e+05,
        -1.61513413e+06, -1.86873723e+05])
base_energy = 42.405408008780796

targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0',
           'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']


# for pre-processing target based on atom ref


class ZC(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root, split, radius=2.0, update_atomrefs=True):
        assert split in ["train", "valid", "test"]
        self.split = split
        self.root = osp.abspath(root)
        self.radius = radius
        self.update_atomrefs = update_atomrefs
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # with open('/cpfs01/projects-HDD/cfff-4405968bce88_HDD/anjunyi/ZC_training/240304_ZC_1100_balance_samples.pkl', 'rb') as f:
        #     suppl = pickle.load(f)

        # slices = np.load('/cpfs01/projects-HDD/cfff-4405968bce88_HDD/anjunyi/ZC_training/splits.npz')
        # indices = slices['idx_'+split]

        # self.data = list(map(suppl.__getitem__, indices))

    def mean(self, target: int) -> float:
        # y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        # y = self.data.energy
        return float(self.data.energy.mean())


    def std(self, target: int) -> float:
        # y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(self.data.energy.std())


    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None


    @property
    def raw_file_names(self) -> List[str]:
        return ['240304_balance_samples.pkl']


    @property
    def processed_file_names(self) -> str:
        #return "_".join([self.split, str(np.round(self.radius, 2)), self.feature_type]) + '.pt'
        return "all_balance_".join(self.split) + '.pt'


    # def download(self):
        # try:
        #     import rdkit  # noqa
        #     file_path = download_url(self.raw_url, self.raw_dir)
        #     extract_zip(file_path, self.raw_dir)
        #     os.unlink(file_path)

        #     file_path = download_url(self.raw_url2, self.raw_dir)
        #     os.rename(osp.join(self.raw_dir, '3195404'),
        #               osp.join(self.raw_dir, 'uncharacterized.txt'))
        # except ImportError:
        #     path = download_url(self.processed_url, self.raw_dir)
        #     extract_zip(path, self.raw_dir)
        #     os.unlink(path)


    def process(self):
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem.rdchem import HybridizationType
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            assert False, "Install rdkit-pypi"
            
        # H He Li p B Ca[7]
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7, "Br": 8, "I": 9}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # with open(self.raw_paths[1], 'r') as f:
        #     target = f.read().split('\n')[1:-1]
        #     target = [[float(x) for x in line.split(',')[1:20]]
        #               for line in target]
        #     target = torch.tensor(target, dtype=torch.float)
        #     target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
        #     target = target * conversion.view(1, -1)
        with open(self.raw_paths[0], 'rb') as f:
            suppl = pickle.load(f)

        data_list = []

        Nmols = len(suppl)
        Ntrain = int(0.7 * Nmols)
        Ntest = Nmols - Ntrain

        np.random.seed(0)
        data_perm = np.random.permutation(Nmols)

        train, valid = np.split(data_perm, [Ntrain])
        indices = {"train": train, "valid": valid}

        np.savez(os.path.join(self.root, 'splits.npz'), idx_train=train, idx_valid=valid)

        # Add a second index to align with cormorant splits.
        j = 0
        
        for i, mol in enumerate(tqdm(suppl)):
            if j not in indices[self.split]:
                j += 1
                continue
            j += 1

            N = len(mol['elements'])
            mol_name = mol['mol_name']
            inchi_key = mol['inchi_key']
            position = torch.tensor(mol['position'], dtype=torch.float)
            formal_charge = torch.tensor(mol['formal_charge'], dtype=torch.float)
            energy = torch.tensor(mol['energy'], dtype=torch.float)
            # force = torch.tensor(mol['force'], dtype=torch.float).reshape(-1, 3)
            bond_list = torch.tensor(mol['edge_list'], dtype=torch.long)
            bond_types = torch.tensor(mol['edge_types'], dtype=torch.long)

            # edge_index = radius_graph(position, r=self.radius, loop=False)
            edge_index = None
        
            
            z = torch.tensor(mol['elements'], dtype=torch.long)
            # k = torch.tensor([1, 6, 7, 8, 9, 15, 16, 17, 35, 53])
            # types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            if self.update_atomrefs:
                node_atom = z.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4, -1, -1, -1, -1, -1, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9])[z]

                atomrefs_value = atomrefs[node_atom]
                atomrefs_value = torch.sum(atomrefs_value, dim=0, keepdim=True)
                energy = energy - atomrefs_value - base_energy          

            data = Data(position=position, z=z, 
                name=mol_name, inchi_key=inchi_key, index=i,
                formal_charge=formal_charge, energy=energy,
                bond_list=bond_list, bond_types=bond_types)
            data_list.append(data)
            
        torch.save(self.collate(data_list), self.processed_paths[0])


def get_cormorant_features(one_hot, charges, charge_power, charge_scale):
    """ Create input features as described in section 7.3 of https://arxiv.org/pdf/1906.04015.pdf """
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


if __name__ == "__main__":
    
    from torch_geometric.loader import DataLoader
    import matplotlib.pyplot as plt
    
    #dataset = QM9("temp", "valid", feature_type="one_hot")
    #print("length", len(dataset))
    #dataloader = DataLoader(dataset, batch_size=4)
    
    '''
    _target = 1
    
    dataset = QM9("test_atom_ref/with_atomrefs", "test", feature_type="one_hot", update_atomrefs=True)
    mean = dataset.mean(_target)
    _, std = dataset.calc_stats(_target)
    
    dataset_original = QM9("test_atom_ref/without_atomrefs", "test", feature_type="one_hot", update_atomrefs=False)
    
    for i in range(12):
        mean = dataset.mean(i)
        std = dataset.std(i)
        
        mean_original = dataset_original.mean(i)
        std_original = dataset_original.std(i)
        
        print('Target: {}, mean diff = {}, std diff = {}'.format(i, 
            mean - mean_original, std - std_original))
    '''

    dataset = ZC("test_torchmd_net_splits", "train", feature_type="one_hot", update_atomrefs=True, torchmd_net_split=True)
    dataset.process()


# b = suppl
# m = torch.ones(len(b), 11)
# n = torch.zeros(len(b), 1)
# k = torch.tensor([1, 6, 7, 8, 9, 15, 16, 17, 35, 53])


# for i in range(len(b)):
#     for p in range(10):
#         j = k[p]
#         m[i][p] = (torch.tensor(b[i]['elements']) == j).sum()
#         n[i][0] = torch.tensor(b[i]['energy'])
# res = torch.linalg.lstsq(m, n)
