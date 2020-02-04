#
#    pyblock: Spin-adapted quantum chemistry DMRG in MPO language (based on Block C++ code)
#    Copyright (C) 2019-2020 Huanchen Zhai
#
#    Block 1.5.3: density matrix renormalization group (DMRG) algorithm for quantum chemistry
#    Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012
#    Copyright (C) 2012 Garnet K.-L. Chan
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Matrix Product Operator.
"""

from ..symmetry.symmetry import ParticleN, SU2, LineCoupling
from ..symmetry.symmetry import point_group
from ..hamiltonian.block import BlockSymmetry, BlockEvaluation, MPSInfo
from ..hamiltonian.block import BlockWavefunction, BlockMultiplyH
from ..davidson import davidson
from .tensor import TensorNetwork, Tensor
from .operator import OpElement, OpNames
from .mps import MPS
from fractions import Fraction
import copy
import numpy as np

class MPOError(Exception):
    pass

class MPO(TensorNetwork):
    pass

class OperatorTensor(Tensor):
    """
    MPO tensor.
    
    Attributes:
        mat : numpy.ndarray(dtype=OpExpression)
            2-D array of Symbolic operator expressions.
        ops : dict(OpElement -> StackSparseMatrix)
            Numeric representation of operator symbols.
            When the object is the super block MPO, :attr:`ops` is a pair of dicts representing
            operator symbols for left and right blocks, respectively.
        left_op_names : numpy.ndarray(dtype=OpExpression)
            1-D array of operator symbols, for naming the result Symbolically from left-block blocking.
            When the object is the super block MPO, :attr:`left_op_names` is the left block StateInfo.
        right_op_names : numpy.ndarray(dtype=OpExpression)
            1-D array of operator symbols, for naming the result Symbolically from right-block blocking.
            When the object is the super block MPO, :attr:`right_op_names` is the right block StateInfo.
    """
    def __init__(self, mat, ops, tags=None, contractor=None, lop=None, rop=None):
        self.mat = mat
        self.ops = ops
        self.left_op_names = lop
        self.right_op_names = rop
        super().__init__([], tags=tags, contractor=contractor)
    
    def __repr__(self):
        if isinstance(self.ops, dict):
            return repr(self.mat) + "\n" + "\n".join([repr(k) + " :: \n" + repr(v) for k, v in self.ops.items()])
        elif isinstance(self.ops, tuple) and len(self.ops) == 2:
            mat = repr(self.mat)
            l = "\n[ LEFT]" + "\n".join([repr(k) + " :: \n" + repr(v) for k, v in self.ops[0].items()])
            r = "\n[RIGHT]" + "\n".join([repr(k) + " :: \n" + repr(v) for k, v in self.ops[1].items()])
            return mat + l + r
        else:
            assert False
    
    def copy(self):
        """Return shallow copy of this object."""
        assert isinstance(self.ops, dict)
        return OperatorTensor(mat=self.mat.copy(), ops=self.ops.copy(),
            lop=self.left_op_names.copy(), rop=self.right_op_names.copy(),
            tags=self.tags.copy(), contractor=self.contractor)

class BlockMPO(MPO):
    """
    MPO chain for interfacing block code.
    
    Attributes:
        hamil : BlockHamiltonian
            Hamiltonian.
        PG : type(PointGroup)
            The class for PointGroup (type object).
        empty : DirectProdGroup
            Vaccum state.
        target : DirectProdGroup
            Target state.
        spatial : list(str)
            Name of irreducible representation of point group for each site.
        site_basis : list(dict(DirectProdGroup -> int))
            Site basis.
        creation_q_labels : list(DirectProdGroup)
            The quantum label of creation operator for each site.
        n_sites : int
            Number of sites/orbitals.
        lcp : LineCoupling
            LineCoupling object.
        info : MPSInfo
            MPSInfo object.
        site_operators : list(dict(OpElement -> StackSparseMatrix))
            Numeric representation of operator symbols for each site.
        tensors : OperatorTensor
            MPO tensor for each site.
    """

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
        
        self.creation_q_labels = [ParticleN(1) * SU2(Fraction(1, 2)) * self.PG(sp)
                                 for sp in self.spatial]

        self.n_sites = self.hamil.n_sites
        
        self.lcp = None
        self.info = None
        
        super().__init__()
    
    def init_site_operators(self):
        """Generate :attr:`site_operators`."""
        self.site_operators = []
        for i in range(self.n_sites):
            self.site_operators.append(self.hamil.get_site_operators(i))
    
    def init_mpo_tensors(self):
        """Generate :attr:`tensors`."""
        self.tensors = []
        op_h = OpElement(OpNames.H, ())
        op_i = OpElement(OpNames.I, ())
        op_c = OpElement(OpNames.C, ())
        op_d = OpElement(OpNames.D, ())
        for i in range(self.n_sites):
            if i == 0:
                mat = np.array([[op_h, op_c, op_d, op_i]], dtype=object)
            else:
                if i == self.n_sites - 1:
                    mat = np.zeros((2 + 2 * i, 1), dtype=object)
                else:
                    mat = np.zeros((2 + 2 * i, 4 + 2 * i), dtype=object)
                    mat[1 + 2 * i, 2 * i + 1] = op_c
                    mat[1 + 2 * i, 2 * i + 2] = op_d
                    mat[1 + 2 * i, 2 * i + 3] = op_i
                    for j in range(1, 2 * i + 1):
                        mat[j, j] = op_i
                mat[0, 0] = op_i
                mat[1 + 2 * i, 0] = op_h
                for j in range(0, i):
                    mat[1 + j * 2, 0] = OpElement(OpNames.S, (j, ))
                    mat[2 + j * 2, 0] = -OpElement(OpNames.SD, (j, ))
            lop = np.zeros((mat.shape[0], ), dtype=object)
            rop = np.zeros((mat.shape[1], ), dtype=object)
            lop[-1] = OpElement(OpNames.H, (), q_label=self.empty)
            if i != 0:
                lop[0] = OpElement(OpNames.I, (), q_label=self.empty)
                for j in range(0, i):
                    lop[1 + j * 2] = OpElement(OpNames.S, (j, ), q_label=-self.creation_q_labels[j])
                    lop[2 + j * 2] = -OpElement(OpNames.SD, (j, ), q_label=self.creation_q_labels[j])
            rop[0] = OpElement(OpNames.H, (), q_label=self.empty)
            if i != self.n_sites - 1:
                for j in range(i + 1):
                    rop[1 + j * 2] = OpElement(OpNames.C, (j, ), q_label=self.creation_q_labels[j])
                    rop[2 + j * 2] = OpElement(OpNames.D, (j, ), q_label=-self.creation_q_labels[j])
                rop[-1] = OpElement(OpNames.I, (), q_label=self.empty)
            self.tensors.append(OperatorTensor(mat=mat, tags={i}, lop=lop, rop=rop,
                ops=self.site_operators[i], contractor=self))

    def copy(self):
        """Return shallow copy of this object."""
        cp = copy.copy(self)
        cp.tensors = [t.copy() for t in self.tensors]
        return cp
    
    def get_line_coupling(self, bond_dim=-1):
        """Generate LineCoupling using default scheme, with given maximal bond dimension.
        The LineCoupling object defines the quantum numbers in MPS and the bond dimension for each quantum number."""
        lcp = LineCoupling(
            self.n_sites, self.site_basis, self.target, self.empty, both_dir=True)

        if bond_dim != -1:
            lcp.set_bond_dim(bond_dim)
        
        return lcp
    
    def set_line_coupling(self, lcp):
        """Apply the given LineCoupling to generate :attr:`info`."""
        self.lcp = lcp
        self.info = MPSInfo.from_line_coupling(lcp)
        self.info.init_state_info()

    def rand_state(self):
        """Return a MPS filled with random reduced matrices.
        A LineCoupling must be set before calling this method."""
        assert self.lcp is not None
        mps = MPS.from_line_coupling(self.lcp)
        mps.randomize()
        return mps
    
    def identity_state(self):
        """Return a MPS filled with identity matrices (whenever possible).
        A LineCoupling must be set before calling this method."""
        assert self.lcp is not None
        mps = MPS.from_line_coupling(self.lcp)
        mps.build_identity()
        return mps

    def hf_state(self, occ):
        """Return a MPS corresponding to the determinant with given occupation number.
        Not implemented."""
        pass
    
    def _tag_site(self, tensor):
        tags = tensor.tags
        for tag in tags:
            if isinstance(tag, int):
                return tag
        if '_LEFT' in tags:
            return -1
        elif '_RIGHT' in tags:
            return self.n_sites
        else:
            assert False
            return None
    
    def exact_eigs(self, hmpo, mps):
        """
        Exact diagonalization.
        
        Args:
            hmpo : OperatorTensor
                Super block MPO.
            mps : MPS
                MPS including tensors in dot blocks.
        
        Returns:
            energy : float
                Ground state energy.
            v : class:`Tensor`
                In two-dot scheme, the rank-2 tensor representing two-dot object.
                Both left and right rank indices are fused. No CG factor are generated.
                One-dot scheme is not implemented.
        """
        if len(mps) == 2:
            mps_tensors = sorted(mps.tensors, key=self._tag_site)
            wfn = self.info.get_wavefunction(self._tag_site(mps_tensors[0]), mps_tensors)
            b = [BlockWavefunction(wfn)]
            
            from block.symmetry import state_tensor_product_target
            st = state_tensor_product_target(hmpo.left_op_names, hmpo.right_op_names)
            st.collect_quanta()
            direct = BlockEvaluation.expr_eval(hmpo.mat[0, 0], hmpo.ops[0], hmpo.ops[1], st, self.empty)
            evs = []
            for k, v in direct.non_zero_blocks:
                p, pp = np.linalg.eigh(v.ref)
                assert np.allclose(v.ref - v.ref.T, 0.0)
                ppp = sorted(zip(p, pp.T), key=lambda x : x[0])
                evs.append((ppp[0][0], ppp[0][1], k))
            evs.sort()
            wfn.clear()
            ig = 0
            for j in st.old_to_new_state[evs[0][2][0]]:
                il = st.left_unmap_quanta[j]
                ir = st.right_unmap_quanta[j]
                gn = (st.left_state_info.n_states[il], st.right_state_info.n_states[ir])
                gnx = gn[0] * gn[1]
                wfn.operator_element(il, ir).ref[:, :] = evs[0][1][ig:ig + gnx].reshape(gn)
                ig += gnx
            assert ig == len(evs[0][1])
            
            a = BlockMultiplyH(hmpo)
            b = BlockWavefunction(wfn)
            br = b.clear_copy()
            a.apply(b, br)
            br += (-evs[0][0]) * b
            assert np.isclose(br.dot(br), 0.0)
            
            v = self.info.from_wavefunction_fused(self._tag_site(mps_tensors[0]), wfn)
            return evs[0][0] + self.hamil.e, v
        else:
            assert False
    
    def eigs(self, hmpo, mps):
        """
        Davidson diagonalization.
        
        Args:
            hmpo : OperatorTensor
                Super block MPO.
            mps : MPS
                MPS including tensors in dot blocks.
        
        Returns:
            energy : float
                Ground state energy.
            v : class:`Tensor`
                In two-dot scheme, the rank-2 tensor representing two-dot object.
                Both left and right rank indices are fused. No CG factor are generated.
                One-dot scheme is not implemented.
        """
        if len(mps) == 2:
            mps_tensors = sorted(mps.tensors, key=self._tag_site)
            wfn = self.info.get_wavefunction(self._tag_site(mps_tensors[0]), mps_tensors)
            b = [BlockWavefunction(wfn)]
            a = BlockMultiplyH(hmpo)
            es, vs = davidson(a, b, 1)
            
            if len(es) == 0:
                raise MPOError('Davidson not converged!!')
            
            e = es[0]
            v = self.info.from_wavefunction_fused(self._tag_site(mps_tensors[0]), vs[0].data)
            return e + self.hamil.e, v
        else:
            assert False
    
    def update_local_mps_info(self, i, l_fused):
        """Update :attr:`info` for site i using the left tensor from SVD."""
        block_basis = [(b.q_labels[1], b.reduced.shape[1]) for b in l_fused.blocks]
        self.info.update_local_block_basis(i, block_basis)
    
    def unfuse_left(self, i, tensor):
        __doc__ = self.info.unfuse_left.__doc__
        return self.info.unfuse_left(i, tensor)
    
    def unfuse_right(self, i, tensor):
        __doc__ = self.info.unfuse_right.__doc__
        return self.info.unfuse_right(i, tensor)
    
    def contract(self, tn, tags):
        """
        Tensor network contraction.
        
        Args:
            tn : TensorNetwork
                Part of tensor network to be contracted.
            tags : (str, int) or (str, )
                Tags of the tensor network to be contracted.
                If tags = ('_LEFT', i), the contraction is corresponding to
                blocking and renormalizing left block at site i.
                If tags = ('_RIGHT', i), the contraction is corresponding to
                blocking and renormalizing right block at site i.
                If tags = ('_HAM'), the contraction is corresponding to
                blocking both left and right block and forming the super block hamiltonian.
        
        Returns:
            mpo : OperatorTensor
                The contracted MPO tensor.
        """
        assert isinstance(tags, tuple)
        if len(tags) == 2:
            dir, i = tags
        else:
            dir = tags[0]
        if dir == '_LEFT':
            ket = tn[{i, '_KET'}]
            ham = tn[{i, '_HAM'}]
            if i == 0:
                ham_rot = BlockEvaluation.left_rotate(ham, ket, self.info, i)
                ham_rot.tags.add(dir)
                return ham_rot
            else:
                ham_prev = tn[{i - 1, '_HAM', dir}]
                ham_ctr = BlockEvaluation.left_contract(ham_prev, ham, self.info, i)
                ham_rot = BlockEvaluation.left_rotate(ham_ctr, ket, self.info, i)
                ham_rot.tags.add(dir)
                return ham_rot
        elif dir == '_RIGHT':
            ket = tn[{i, '_KET'}]
            ham = tn[{i, '_HAM'}]
            if i == self.n_sites - 1:
                ham_rot = BlockEvaluation.right_rotate(ham, ket, self.info, i)
                ham_rot.tags.add(dir)
                return ham_rot
            else:
                ham_prev = tn[{i + 1, '_HAM', dir}]
                ham_ctr = BlockEvaluation.right_contract(ham_prev, ham, self.info, i)
                ham_rot = BlockEvaluation.right_rotate(ham_ctr, ket, self.info, i)
                ham_rot.tags.add(dir)
                return ham_rot
        elif dir == '_HAM':
            ts = sorted(tn.tensors, key=self._tag_site)
            if len(tn) == 4:
                if self._tag_site(ts[0]) != -1:
                    ham_left = BlockEvaluation.left_contract(ts[0], ts[1], self.info, self._tag_site(ts[1]))
                else:
                    ham_left = ts[1]
                if self._tag_site(ts[3]) != self.n_sites:
                    ham_right = BlockEvaluation.right_contract(ts[3], ts[2], self.info, self._tag_site(ts[2]))
                else:
                    ham_right = ts[2]
                return BlockEvaluation.left_right_contract(ham_left, ham_right, self.info, self._tag_site(ts[1]))
            else:
                assert False
        else:
            assert False
