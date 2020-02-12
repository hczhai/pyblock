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
Specialized MPS/MPO operations for DMRG.
"""

from block.symmetry import state_tensor_product_target
from block.operator import Wavefunction
from block.rev import tensor_scale, tensor_dot_product, tensor_scale_add_no_trans
from block.rev import tensor_precondition

from ..tensor.tensor import Tensor, SubTensor
from ..davidson import davidson
from .core import BlockHamiltonian, BlockEvaluation
import numpy as np


class DMRGContractor:
    def __init__(self, mps_info, mpo_info):
        self.mps_info = mps_info
        self.mpo_info = mpo_info
        self.n_sites = mpo_info.n_sites
        self.mem_ptr = 0
    
    def pre_sweep(self):
        """Operations performed at the beginning of each DMRG sweep."""
        self.mem_ptr = BlockHamiltonian.get_current_memory()
    
    def post_sweep(self):
        """Operations performed at the end of each DMRG sweep."""
        BlockHamiltonian.set_current_memory(self.mem_ptr)
    
    def _tag_site(self, tensor):
        tags = tensor.tags
        itag = None
        for tag in tags:
            if isinstance(tag, int):
                if itag is None or itag > tag:
                    itag = tag
        if itag is not None:
            return itag
        if '_LEFT' in tags:
            return -1
        elif '_RIGHT' in tags:
            return self.n_sites
        else:
            assert False
            return None
    
    def eigs(self, opt, mpst):
        """
        Davidson diagonalization.
        
        Args:
            opt : OperatorTensor
                Super block contracted operator tensor.
            mpst : Tensor
                Contracted MPS tensor in dot blocks.
        
        Returns:
            energy : float
                Ground state energy.
            v : class:`Tensor`
                In two-dot scheme, the rank-2 tensor representing two-dot object.
                Both left and right rank indices are fused.
                One-dot scheme is not implemented.
            ndav : int
                Number of Davidson iterations.
        """
        if len(mpst.tags - {'_KET'}) == 2:
            
            mpst = mpst.copy()
            i = self._tag_site(mpst)
            
            l = self.mps_info.left_block_basis[i - 1] if i > 0 else None
            r = self.mps_info.basis[i]
            mpst.fuse_index(0, self.mps_info.lcp.tensor_product(l, r), target=self.mps_info.lcp.target)
            
            l = self.mps_info.basis[i + 1]
            r = self.mps_info.right_block_basis[i + 2] if i + 1 < self.n_sites - 1 else None
            mpst.fuse_index(1, self.mps_info.lcp.tensor_product(l, r), target=self.mps_info.lcp.target)
            
            wfn = self.mps_info.get_wavefunction_fused(i, mpst, dot=2)
            
            st_l = self.mps_info.left_state_info_no_trunc[i]
            st_r = self.mps_info.right_state_info_no_trunc[i + 1]
            st = state_tensor_product_target(st_l, st_r)
            a = BlockMultiplyH(opt, st)
            b = [BlockWavefunction(wfn)]
            
            es, vs, ndav = davidson(a, b, 1)
            
            if len(es) == 0:
                raise MPOError('Davidson not converged!!')
            
            e = es[0]
            v = self.mps_info.from_wavefunction_fused(i, vs[0].data)
            return e + self.mpo_info.hamil.e, v, ndav
        else:
            assert False
    
    def update_local_left_mps_info(self, i, l_fused):
        """Update :attr:`info` for site i using the left tensor from SVD."""
        block_basis = [(b.q_labels[1], b.reduced.shape[1]) for b in l_fused.blocks]
        self.mps_info.update_local_left_block_basis(i, block_basis)
        
    def update_local_right_mps_info(self, i, r_fused):
        """Update :attr:`info` for site i using the right tensor from SVD."""
        block_basis = [(b.q_labels[0], b.reduced.shape[0]) for b in r_fused.blocks]
        self.mps_info.update_local_right_block_basis(i, block_basis)
    
    def unfuse_left(self, i, tensor):
        l = [(self.mps_info.lcp.empty, 1)] if i == 0 else self.mps_info.left_block_basis[i - 1]
        r = self.mps_info.basis[i]
        tensor = tensor.copy()
        tensor.unfuse_index(0, l, r)
        return tensor.set_tags({i}).set_contractor(self)
    
    def unfuse_right(self, i, tensor):
        l = self.mps_info.basis[i]
        r = [(self.mps_info.lcp.empty, 1)] if i == self.n_sites - 1 else self.mps_info.right_block_basis[i + 1]
        tensor = tensor.copy()
        tensor.unfuse_index(1, l, r)
        return tensor.set_tags({i}).set_contractor(self)
    
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
        elif len(tags) == 3:
            dir, i, j = tags
        else:
            dir = tags[0]
        if dir == '_LEFT':
            ket = tn[{i, '_KET'}]
            ham = tn[{i, '_HAM'}]
            if i == 0:
                ham_rot = BlockEvaluation.left_rotate(ham, ket, self.mps_info, i)
                ham_rot.tags.add(dir)
                return ham_rot
            else:
                ham_prev = tn[{i - 1, '_HAM', dir}]
                ham_ctr = BlockEvaluation.left_contract(ham_prev, ham, self.mps_info, self.mpo_info, i)
                ham_rot = BlockEvaluation.left_rotate(ham_ctr, ket, self.mps_info, i)
                ham_rot.tags.add(dir)
                return ham_rot
        elif dir == '_RIGHT':
            ket = tn[{i, '_KET'}]
            ham = tn[{i, '_HAM'}]
            if i == self.n_sites - 1:
                ham_rot = BlockEvaluation.right_rotate(ham, ket, self.mps_info, i)
                ham_rot.tags.add(dir)
                return ham_rot
            else:
                ham_prev = tn[{i + 1, '_HAM', dir}]
                ham_ctr = BlockEvaluation.right_contract(ham_prev, ham, self.mps_info, self.mpo_info, i)
                ham_rot = BlockEvaluation.right_rotate(ham_ctr, ket, self.mps_info, i)
                ham_rot.tags.add(dir)
                return ham_rot
        elif dir == '_HAM':
            ts = sorted(tn.tensors, key=self._tag_site)
            if len(tn) == 4:
                if self._tag_site(ts[0]) != -1:
                    ham_left = BlockEvaluation.left_contract(ts[0], ts[1], self.mps_info, self.mpo_info, self._tag_site(ts[1]))
                else:
                    ham_left = ts[1]
                if self._tag_site(ts[3]) != self.n_sites:
                    ham_right = BlockEvaluation.right_contract(ts[3], ts[2], self.mps_info, self.mpo_info, self._tag_site(ts[2]))
                else:
                    ham_right = ts[2]
                return BlockEvaluation.left_right_contract(ham_left, ham_right)
            else:
                print(tn.tags)
                assert False
        elif dir == '_KET':
            ts = sorted(tn.tensors, key=self._tag_site)
            map_idx_b = {}
            for block in ts[1].blocks:
                subg = block.q_labels[0]
                if subg not in map_idx_b:
                    map_idx_b[subg] = []
                map_idx_b[subg].append(block)
            # note that for future fusing indices same q_labels must be kept un-summed
            blocks = []
            for block_a in ts[0].blocks:
                subg = block_a.q_labels[2]
                if subg in map_idx_b:
                    outga = block_a.q_labels[0:2]
                    for block_b in map_idx_b[subg]:
                        outg = outga + block_b.q_labels[1:]
                        mat = np.tensordot(block_a.reduced, block_b.reduced, axes=([2], [0]))
                        blocks.append(SubTensor(q_labels=outg, reduced=mat))
            return Tensor(blocks=blocks).set_tags({dir, i, j})
        else:
            assert False

class BlockWavefunction:
    """A wrapper of Wavefunction (block code) for Davidson algorithm."""
    def __init__(self, wave, factor=1.0):
        self.data = wave
        self.factor = factor
    
    def __rmul__(self, factor):
        return BlockWavefunction(self.data, self.factor * factor)
    
    def __imul__(self, factor):
        tensor_scale(self.factor * factor, self.data)
        self.factor = 1.0
        return self
    
    def __iadd__(self, other):
        assert self.factor == 1.0
        tensor_scale_add_no_trans(other.factor, other.data, self.data)
        return self
    
    def copy(self):
        """Return a deep copy of this object."""
        mat = self.data.__class__()
        mat.deep_copy(self.data)
        return BlockWavefunction(mat, self.factor)
    
    def clear_copy(self):
        """Return a deep copy of this object, but all the matrix elements are set to zero."""
        mat = self.data.__class__()
        mat.deep_clear_copy(self.data)
        return BlockWavefunction(mat, 1.0)
    
    def copy_data(self, other):
        """Fill the matrix elements in this object with data
        from another :class:`BlockWavefunction` object."""
        self.data.copy_data(other.data)
        self.factor = other.factor
    
    def dot(self, other):
        """Return dot product of two :class:`BlockWavefunction`."""
        return tensor_dot_product(self.data, other.data) * self.factor * other.factor
    
    def precondition(self, ld, diag):
        """
        Apply precondition on this object.
        
        Args:
            ld : float
                Eigenvalue.
            diag : DiagonalMatrix
                Diagonal elements of Hamiltonian.
        """
        tensor_precondition(self.data, ld, diag)
    
    def normalize(self):
        """Normalization."""
        self.factor = 1.0
        tensor_scale(1 / np.sqrt(self.dot(self)), self.data)
    
    def deallocate(self):
        """Deallocate the memory associated with this object."""
        assert self.data is not None
        self.data.deallocate()
        self.data = None
    
    def __repr__(self):
        return repr(self.factor) + " * " + repr(self.data)

class BlockMultiplyH:
    """
    A wrapper of Block.MultiplyH (block code) for Davidson algorithm.
    
    Attributes:
        opt : OperatorTensor
            The (symbolic) super block Hamiltonian.
        st : StateInfo
            StateInfo of super block.
        diag_mat : DiagonalMatrix
            Diagonal elements of super block Hamiltonian, in flatten form with no quantum labels.
    """
    def __init__(self, opt, st):
        self.opt = opt
        self.st = st
        self.diag_mat = BlockEvaluation.expr_diagonal_eval(opt.mat[0, 0], opt.ops[0], opt.ops[1], st)
    
    def diag(self):
        """Returns Diagonal elements (for preconditioning)."""
        return self.diag_mat
    
    def apply(self, other, result):
        """
        Perform :math:`\\hat{H}|\\psi\\rangle`.
        
        Args:
            other : BlockWavefunction
                Input vector/wavefunction.
            result : BlockWavefunction
                Output vector/wavefunction.
        """
        assert isinstance(result, BlockWavefunction)
        result.factor = 1.0
        BlockEvaluation.expr_multiply_eval(self.opt.mat[0, 0], self.opt.ops[0], self.opt.ops[1],
            other.data, result.data, self.st)
