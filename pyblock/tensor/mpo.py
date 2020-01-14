
from ..symmetry.symmetry import ParticleN, SU2, LineCoupling
from ..symmetry.symmetry import point_group
from ..hamiltonian.block import BlockSymmetry
from ..tensor.tensor import TensorNetwork, Tensor
from .mps import MPS
from fractions import Fraction
import copy

class MPO(TensorNetwork):
    pass

class BlockMPO(MPO):

    def __init__(self, hamiltonian):

        self.hamil = hamiltonian

        self.PG = point_group(self.hamil.point_group)

        # assuming SU2 representation
        self.empty = ParticleN(0) * SU2(0) * self.PG(0)
        self.spatial = [self.PG.IrrepNames[ir] for ir in self.hamil.spatial_syms]
        self.site_basis = [{
            ParticleN(0) * SU2(0) * self.PG(0): 1,
            ParticleN(1) * SU2(Fraction(1, 2)) * self.PG(sp): 1,
            ParticleN(2) * SU2(0) * self.PG(0): 1
        } for sp in self.spatial]
        self.target = ParticleN(self.hamil.n_electrons) \
            * SU2(self.hamil.target_s) * self.PG(self.hamil.target_spatial_sym)

        self.n_sites = self.hamil.n_sites

        # virtual tensor representation
        mpo_tensors = []
        for i in range(self.n_sites):
            t = Tensor(blocks=None, tags={i})
            t.contractor = self
            mpo_tensors.append(t)
        
        super().__init__(tensors=mpo_tensors)

        self.site_info = {'_LEFT': {}, '_RIGHT': {}}
        self.rot_mat = {'_LEFT': {}, '_RIGHT': {}}
        self.mps0 = None
    
    def copy(self):
        cp = copy.copy(self)
        cp.tensors = [t.copy() for t in self.tensors]
        return cp

    def rand_state(self, bond_dim=-1):
        lcp = LineCoupling(
            self.n_sites, self.site_basis, self.target, self.empty, both_dir=True)
        
        if bond_dim != -1:
            lcp.set_bond_dim(bond_dim)
        
        mps = MPS.from_line_coupling(lcp)
        mps.randomize()
        return mps

    def hf_state(self, occ):
        pass
    
    def contract(self, tn, tags):
        dir, i = tags
        ket = tn[{i, '_KET'}]
        if dir == '_LEFT':
            if i == 0:
                self.site_info[dir][i] = BlockSymmetry.initial_state_info(i)
                self.mps0 = ket
            else:
                cur = self.site_info[dir][i - 1]
                if i == 1:
                    tensor0 = self.mps0
                else:
                    tensor0 = None
                rot, cur = BlockSymmetry.to_rotation_matrix(cur, ket, i, tensor0)
                self.site_info[dir][i] = cur
                self.rot_mat[dir][i] = rot
            extra_tag = (0, i)
        elif dir == '_RIGHT':
            if i == self.n_sites - 1:
                self.site_info[dir][i] = BlockSymmetry.initial_state_info(i)
                self.mps0 = ket
            else:
                cur = self.site_info[dir][i + 1]
                if i == self.n_sites - 2:
                    tensor0 = self.mps0
                else:
                    tensor0 = None
                rot, cur = BlockSymmetry.to_rotation_matrix(cur, ket, i, tensor0)
                self.site_info[dir][i] = cur
                self.rot_mat[dir][i] = rot
            extra_tag = (i, self.n_sites - 1)
        print('rot mat: ', extra_tag)
        return tn[dir]
