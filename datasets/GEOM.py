from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
from torch_geometric.nn import radius_graph


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0',
           'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']


# for pre-processing target based on atom ref
atomrefs_tensor = torch.zeros(5, 19)
atomrefs_tensor[:, 7]  = torch.tensor(atomrefs[7])
atomrefs_tensor[:, 8]  = torch.tensor(atomrefs[8])
atomrefs_tensor[:, 9]  = torch.tensor(atomrefs[9])
atomrefs_tensor[:, 10] = torch.tensor(atomrefs[10])

class GEOM(InMemoryDataset):

    def __init__(self, root, split, feature_type="one_hot", update_atomrefs=True, torchmd_net_split=True):
        assert feature_type in ["one_hot", "cormorant", "gilmer"], "Please use valid features"
        assert split in ["train", "valid", "test"]
        self.split = split
        self.feature_type = feature_type
        self.root = osp.abspath(root)
        self.update_atomrefs = update_atomrefs
        self.torchmd_net_split = torchmd_net_split
        
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def calc_stats(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        y = y[:, target]
        mean = float(torch.mean(y))
        mad = float(torch.mean(torch.abs(y - mean)))
        #ys = np.array([data.y.item() for data in self])
        #mean = np.mean(ys)
        #mad = np.mean(np.abs(ys - mean))
        return mean, mad

    def mean_std(self) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y.mean()), float(y.std())

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        return ['drugs_crude.msgpack']

    @property
    def processed_file_names(self) -> str:
        return "_".join([self.split, self.feature_type]) + '.pt'

    def process(self):
        try:
            import msgpack
            import rdkit
            from rdkit import Chem
            from rdkit.Chem.rdchem import HybridizationType
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            assert False, "Install rdkit-pypi"
            
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


        suppl = msgpack.Unpacker(open(self.raw_paths[0], "rb"))
        data_list = []

        # Nmols = 0
        # for _ in tqdm(suppl):
        #     Nmols += 1
        # print('Number of mols:', Nmols)
        # Ntrain = int(0.7*Nmols)
        # Nvalid = Nmols - Ntrain

        # np.random.seed(0)
        # data_perm = np.random.permutation(Nmols)

        # train, valid = np.split(data_perm, [Ntrain])
        # indices = {"train": train, "valid": valid}

        # np.savez(os.path.join(self.root, 'splits.npz'), idx_train=train, idx_valid=valid)

        # Add a second index to align with cormorant splits.
        # j = 0
        for i, mol in enumerate(suppl):
            for sub_mol in mol.values():
                for conf in sub_mol['conformers']:
                    xyz = torch.tensor(data=conf['xyz'])
                    y = torch.tensor(data=conf['totalenergy'], dtype=torch.float)
                    pos = xyz[:, 1:]

                    z = torch.tensor(data=xyz[:, 0]).to(dtype=torch.long)            

                    data = Data(pos=pos, z=z, y=y)
                    data_list.append(data)
        
        Nmols = len(data_list)
        Ntrain = int(0.7*Nmols)
        Nvalid = Nmols - Ntrain

        train_name = "_".join(['train', self.feature_type]) + '.pt'
        val_name = "_".join(['valid', self.feature_type]) + '.pt'
        

        torch.save(self.collate(data_list[:Ntrain]), train_name)
        torch.save(self.collate(data_list[Ntrain:]), val_name)


def get_cormorant_features(one_hot, charges, charge_power, charge_scale):
    """ Create input features as described in section 7.3 of https://arxiv.org/pdf/1906.04015.pdf """
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars