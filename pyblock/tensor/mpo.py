
from ..symmetry.symmetry import ParticleN, SU2, LineCoupling
from ..symmetry.symmetry import point_group
from ..hamiltonian.block import BlockSymmetry
from ..hamiltonian.qc import QCHamiltonian
from ..tensor.tensor import TensorNetwork, Tensor
from .mps import MPS
from fractions import Fraction
import copy
import numpy as np


class TensorSet:
    def __init__(self):
        self.tensors = {}


class MPO(TensorNetwork):
    pass


class QCMPO(MPO):

    def __init__(self, hamiltonian):

        self.hamil = hamiltonian

        mpo_tensor_sets = []
        for i in range(self.n_sites):

            ts_set = TensorSet()
            mpo_tensor_sets.append(ts_set)

            ts_set[('I', (), 0)] = self.hamil.operator_identity(
                i).set_contractor(self)
            ts_set[('Cre', (i, ), Fraction(1, 2))] = self.hamil.operator_cre(
                i).set_contractor(self)
            ts_set[('Des', (i,), Fraction(1, 2))] = self.hamil.operator_des(
                i).set_contractor(self)
            
            ts_set.update(self.build_one_site_s_tensor(i, ts_set))
            ts_set.update(self.build_one_site_r_tensor(i, i, i, ts_set))

        super().__init__(tensors=[])

    def build_one_site_s_tensor(self, j, ts_set_ref):
        ts_set = {}
        for i in range(self.hamil.n_sites):
            ts_set[('S', (i,), Fraction(1, 2))] = ts_set_ref[(
                'Des', (j,), Fraction(1, 2))] * self.hamil.t[i, j]
        return ts_set
    
    def build_one_site_r_tensor(self, j, k, l, ts_set_ref):
        ts_set = {}
        for i in range(self.hamil.n_sites):
            ck = ts_set_ref[('Cre', (k,), Fraction(1, 2))]
            dl = ts_set_ref[('Des', (l,), Fraction(1, 2))]
            dj = ts_set_ref[('Des', (j,), Fraction(1, 2))]
            ck_dl = Tensor.contract(dl, ck, [1], [1], self.hamil.empty)
            ck_dl = Tensor.partial_trace(ck_dl, [1], [3])
            ck_dl_dj = Tensor.contract(dj, ck_dl, [1], [1], dj.blocks[0].q_labels[1])
            ck_dl_dj = Tensor.partial_trace(ck_dl_dj, [1], [3]).set_contractor(self)
            ts_set[('R', (i,), Fraction(1, 2))] = ck_dl_dj * (self.hamil.v[i, j, k, l] * np.sqrt(2))
        return ts_set

    def build_one_site_a_tensor(self, i, k, ts_set_ref):
        ts_set = {}
        ci = ts_set_ref[('Cre', (i,), Fraction(1, 2))]
        ck = ts_set_ref[('Cre', (k,), Fraction(1, 2))]
        target_0, target_1 = ci.blocks[0].q_labels[1] + ck.blocks[0].q_labels[1]
        ci_ck_0 = Tensor.contract(ck, ci, [1], [1], target_0)
        ci_ck_0 = Tensor.partial_trace(ci_ck_0, [1], [3]).set_contractor(self)
        ci_ck_1 = Tensor.contract(ck, ci, [1], [1], target_1)
        ci_ck_1 = Tensor.partial_trace(ci_ck_1, [1], [3]).set_contractor(self)
        ts_set[('A', (i, k), 0)] = ci_ck_0
        ts_set[('A', (i, k), 1)] = ci_ck_1
        return ts_set

class BlockMPO(MPO):

    def __init__(self, hamiltonian):

        self.hamil = hamiltonian

        self.PG = point_group(self.hamil.point_group)

        # assuming SU2 representation
        self.empty = ParticleN(0) * SU2(0) * self.PG(0)
        self.spatial = [self.PG.IrrepNames[ir]
                        for ir in self.hamil.spatial_syms]
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
        self.system = None
        self.new_system = None

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
        if isinstance(tags, tuple):
            dir, i = tags
        else:
            dir = tags
        if dir == '_LEFT':
            ket = tn[{i, '_KET'}]
            if i == 0:
                self.site_info[dir][i] = BlockSymmetry.initial_state_info(i)
                self.mps0 = ket
                if self.new_system is None:
                    self.system = self.hamil.make_starting_block(forward=True)
                else:
                    self.new_system = None
            else:
                cur = self.site_info[dir][i - 1]
                if i == 1:
                    tensor0 = self.mps0
                else:
                    tensor0 = None
                rot, cur = BlockSymmetry.to_rotation_matrix(
                    cur, ket, i, tensor0)
                self.site_info[dir][i] = cur
                self.rot_mat[dir][i] = rot
                if self.new_system is None:
                    self.system = self.hamil.enlarge_block(
                        forward=True, system=self.system, rot_mat=rot)
                else:
                    self.system = self.hamil.block_rotation(
                        new_system=self.new_system, system=self.system, rot_mat=rot)
                    self.new_system = None
            extra_tag = (0, i)
        elif dir == '_RIGHT':
            if i == self.n_sites - 1:
                self.site_info[dir][i] = BlockSymmetry.initial_state_info(i)
                self.mps0 = ket
                if self.new_system is None:
                    self.system = self.hamil.make_starting_block(forward=False)
            else:
                cur = self.site_info[dir][i + 1]
                if i == self.n_sites - 2:
                    tensor0 = self.mps0
                else:
                    tensor0 = None
                rot, cur = BlockSymmetry.to_rotation_matrix(
                    cur, ket, i, tensor0)
                self.site_info[dir][i] = cur
                self.rot_mat[dir][i] = rot
                if self.new_system is None:
                    self.system = self.hamil.enlarge_block(
                        forward=False, system=self.system, rot_mat=rot)
                else:
                    self.system = self.hamil.block_rotation(
                        new_system=self.new_system, system=self.system, rot_mat=rot)
                    self.new_system = None
            extra_tag = (i, self.n_sites - 1)
        elif dir == '_HAM':
            s, sd, self.new_system, env, big = self.hamil.make_big_block(
                self.system)
            self.new_system_extra = s, sd, env, big
            return Tensor(blocks=None, tags={'_HAM'}, contractor=self)
        if dir == '_LEFT' or dir == '_RIGHT':
            print('rot mat: ', extra_tag)
            tags = set.union(ts.tags for ts in tn.select(
                tags=i, which='all').tensors)
            for tag in tags:
                tn[dir].tags.add(tag)
            return tn[dir]
