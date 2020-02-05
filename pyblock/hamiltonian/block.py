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
Translation of low-level objects in Block code.
"""

from block import VectorInt, VectorVectorInt, VectorMatrix, Matrix
from block import save_rotation_matrix, DiagonalMatrix
from block.io import Global, read_input, Input, AlgorithmTypes
from block.io import init_stack_memory, release_stack_memory, AlgorithmTypes
from block.io import get_current_stack_memory, set_current_stack_memory
from block.dmrg import MPS_init, MPS, get_dot_with_sys
from block.symmetry import VectorStateInfo, get_commute_parity
from block.symmetry import state_tensor_product, SpinQuantum, VectorSpinQuantum
from block.symmetry import StateInfo, SpinSpace, IrrepSpace, state_tensor_product_target
from block.operator import Wavefunction, OpTypes, StackSparseMatrix
from block.block import Block, StorageTypes, VectorBlock
from block.block import init_starting_block, init_new_system_block
from block.rev import tensor_scale, tensor_trace, tensor_rotate, tensor_product
from block.rev import tensor_scale_add, tensor_scale_add_no_trans, tensor_precondition
from block.rev import tensor_trace_diagonal, tensor_product_diagonal, tensor_dot_product
from block.rev import tensor_trace_multiply, tensor_product_multiply

from ..symmetry.symmetry import ParticleN, SU2, SZ, PointGroup, point_group
from ..symmetry.symmetry import LineCoupling, DirectProdGroup
from ..tensor.operator import OpElement, OpNames, OpString, OpSum
from ..tensor.tensor import Tensor, SubTensor
from .qc import read_fcidump
from fractions import Fraction
import numpy as np
import bisect


class BlockError(Exception):
    pass


# collection of "StateInfo" at each site
# this class also defines how to translate collection of python q numbers
# to StateInfo in block
class MPSInfo:
    """
    Basis information in MPS.
    
    Attributes:
        n_sites : int
            Number of sites.
        empty : DirectProdGroup
            Vaccum state.
        target : DirectProdGroup
            Target state.
        basis : [[(DirectProdGroup, int)]]
            For each site, a list of composite quantum numbers
            and corresponding dimension of reduced matrices,
            which defines the single-site basis.
        left_block_basis : [[(DirectProdGroup, int)]]
            For each left block, a list of composite quantum numbers
            and corresponding dimension of reduced matrices,
            which defines the renormalized basis.
        right_block_basis : [[(DirectProdGroup, int)]]
            For each right block, a list of composite quantum numbers
            and corresponding dimension of reduced matrices,
            which defines the renormalized basis.
        left_state_info : StateInfo
            Left block renormalized/truncated basis, translated to StateInfo (block code).
        right_state_info : StateInfo
            Right block renormalized/truncated basis, translated to StateInfo (block code).
        left_state_info_no_trunc : StateInfo
            Left block basis just before truncation.
        right_state_info_no_trunc : StateInfo
            Right block basis just before truncation.
    """

    def __init__(self, empty, n_sites, basis, target):
        self.n_sites = n_sites
        self.empty = empty
        self.target = target
        self.basis = basis
        self.left_block_basis = []
        self.right_block_basis = []
        self.left_state_info = None
        self.right_state_info = None
        self.left_state_info_no_trunc = None
        self.right_state_info_no_trunc = None
    
    @staticmethod
    def from_line_coupling(lcp):
        """
        Construct :class:`MPSInfo` from :class:`LineCoupling`.
        
        The constructed :class:`MPSInfo` will include all renormalized block basis in python objects,
        but no StateInfo objects will be generated. :func:`init_state_info` can be used to
        generate these StateInfo objects.
        
        Args:
            lcp : :class:`LineCoupling`
        
        Returns:
            info : :class:`MPSInfo`
        """
        info = MPSInfo(lcp.empty, lcp.l, [], lcp.target)
        for i, post in enumerate(lcp.dims):
            info.basis.append(sorted(lcp.basis[i].items(), key=lambda x: x[0]))
            info.left_block_basis.append([])
            for k, v in sorted(post.items(), key=lambda x: x[0]):
                info.left_block_basis[-1].append((k, v))
        lcp_right = LineCoupling(lcp.l, lcp.basis[::-1], lcp.target, lcp.empty, lcp.both_dir)
        if lcp.bond_dim != -1:
            lcp_right.set_bond_dim(lcp.bond_dim)
        for i, post in enumerate(lcp_right.dims[::-1]):
            info.right_block_basis.append([])
            for k, v in sorted(post.items(), key=lambda x: x[0]):
                info.right_block_basis[-1].append((k, v))
        return info
    
    def init_state_info(self):
        """Generate StateInfo objects."""
        self.left_state_info = [None] * self.n_sites
        self.right_state_info = [None] * self.n_sites
        self.left_state_info_no_trunc = [None] * self.n_sites
        self.right_state_info_no_trunc = [None] * self.n_sites
        
        left = None
        for i in range(self.n_sites):
            left = self.update_local_left_state_info(i, left=left)
        
        right = None
        for i in range(self.n_sites - 1, -1, -1):
            right = self.update_local_right_state_info(i, right=right)
    
    def update_local_left_state_info(self, i, left=None):
        """
        Update StateInfo objects for left block ending at site i.
        
        Args:
            i : int
                Last site in the left block.
        
        Kwargs:
            left : StateInfo
                The (optional) StateInfo object for previous left block.
                Defaults to None.
        
        Returns:
            right : StateInfo
                The StateInfo object for current left block.
        """
        right, no_truc = self.get_left_state_info(i, left=left)
        self.left_state_info[i] = right
        self.left_state_info_no_trunc[i] = no_truc
        return right
    
    def update_local_right_state_info(self, i, right=None):
        """
        Update StateInfo objects for right block starting at site i.
        
        Args:
            i : int
                First site in the right block.
        
        Kwargs:
            right : StateInfo
                The (optional) StateInfo object for previous right block.
                Defaults to None.
        
        Returns:
            left : StateInfo
                The StateInfo object for current right block.
        """
        left, no_truc = self.get_right_state_info(i, right=right)
        self.right_state_info[i] = left
        self.right_state_info_no_trunc[i] = no_truc
        return left
    
    # basis is a list of pairs: [(q_label, n_states)]
    def update_local_block_basis(self, i, block_basis):
        """
        Update renormalized basis at site i and associated StateInfo objects.
        
        This will update for left block with sites [0..i] and [0..i+1]
        and right block with sites [i+1..:attr:`n_sites`-1] and  [i..:attr:`n_sites`-1].
        
        Args:
            i : int
                Center site, for determining assocated left and right blocks.
            block_basis : [(DirectProdGroup, int)]
                Renormalized basis for left block with sites [0, i].
        """
        
        self.left_block_basis[i] = []
        for k, v in sorted(block_basis, key=lambda x: x[0]):
            self.left_block_basis[i].append((k, v))
        
        right_block_basis = [(self.target + (-k), v) for k, v in block_basis]
        
        self.right_block_basis[i + 1] = []
        for k, v in sorted(right_block_basis, key=lambda x: x[0]):
            assert isinstance(k, DirectProdGroup)
            self.right_block_basis[i + 1].append((k, v))
        
        left = self.update_local_left_state_info(i)
        self.update_local_left_state_info(i + 1, left=left)
        right = self.update_local_right_state_info(i + 1)
        self.update_local_right_state_info(i, right=right)
    
    def get_left_state_info(self, i, left=None):
        """Construct StateInfo for left block [0..i] (used internally)"""
        if left is None:
            if i == 0:
                left = BlockSymmetry.to_state_info([(self.empty, 1)])
            else:
                left = BlockSymmetry.to_state_info(self.left_block_basis[i - 1])
        middle = BlockSymmetry.to_state_info(self.basis[i])
        right = BlockSymmetry.to_state_info(self.left_block_basis[i])
        right.set_left_state_info(left)
        right.set_right_state_info(middle)
        uncollected = state_tensor_product(left, middle)
        right.left_unmap_quanta = uncollected.left_unmap_quanta
        right.right_unmap_quanta = uncollected.right_unmap_quanta
        right.set_uncollected_state_info(uncollected)
        right.old_to_new_state = VectorVectorInt([VectorInt() for i in range(len(right.quanta))])
        for ii, q in enumerate(uncollected.quanta):
            j = bisect.bisect_left(right.quanta, q)
            if j != len(right.quanta) and right.quanta[j] == q:
                right.old_to_new_state[j].append(ii)
        collected = uncollected.copy()
        collected.collect_quanta()
        return right, collected
    
    def get_right_state_info(self, i, right=None):
        """Construct StateInfo for right block [i..:attr:`n_sites`-1] (used internally)"""
        if right is None:
            if i == self.n_sites - 1:
                right = BlockSymmetry.to_state_info([(self.empty, 1)])
            else:
                right = BlockSymmetry.to_state_info(self.right_block_basis[i + 1])
        middle = BlockSymmetry.to_state_info(self.basis[i])
        left = BlockSymmetry.to_state_info(self.right_block_basis[i])
        left.set_left_state_info(middle)
        left.set_right_state_info(right)
        uncollected = state_tensor_product(middle, right)
        left.left_unmap_quanta = uncollected.left_unmap_quanta
        left.right_unmap_quanta = uncollected.right_unmap_quanta
        left.set_uncollected_state_info(uncollected)
        left.old_to_new_state = VectorVectorInt([VectorInt() for i in range(len(left.quanta))])
        for ii, q in enumerate(uncollected.quanta):
            j = bisect.bisect_left(left.quanta, q)
            if j != len(left.quanta) and left.quanta[j] == q:
                left.old_to_new_state[j].append(ii)
        collected = uncollected.copy()
        collected.collect_quanta()
        return left, collected
    
    def get_wavefunction(self, i, tensors):
        """
        Construct Wavefunction (block code) from :class:`Tensor`.
        
        Args:
            i : int
                Site index of first/left dot.
            tensors : [Tensor]
                For two-dot scheme, a list of two tensors in dot blocks.
                For one-dot scheme, a list of one tensor (not implemented yet).
        
        Returns:
            wfn : Wavefunction
        """
        if len(tensors) == 2:
            st_l = self.left_state_info_no_trunc[i]
            st_r = self.right_state_info_no_trunc[i + 1]
            
            target_state_info = BlockSymmetry.to_state_info([(self.target, 1)])

            wfn = Wavefunction()
            wfn.initialize(target_state_info.quanta, st_l, st_r, False)
            wfn.clear()
            
            rot_l = self.get_left_rotation_matrix(i, tensors[0])
            rot_r = self.get_right_rotation_matrix(i + 1, tensors[1])
            
            for (il, ir), mat in wfn.non_zero_blocks:
                if rot_l[il].cols != 0 and rot_r[ir].cols != 0:
                    mat.ref[:, :] = rot_l[il].ref @ rot_r[ir].ref.T
            
            return wfn
        else:
            assert False
    
    def unfuse_left(self, i, tensor):
        """
        Unfuse left index of rank-2 tensor in one site of MPS.
        
        Args:
            i : int
                Site index of the tensor in MPS.
            tensor : class:`Tensor`
                Rank-2 tensor with left index fused.
        
        Returns:
            tensor : class:`Tensor`
                Rank-3 tensor.
        """
        collected = self.left_state_info_no_trunc[i]
        l = self.left_state_info[i].left_state_info
        r = self.left_state_info[i].right_state_info
        lr = self.left_state_info[i].uncollected_state_info
        
        otn = collected.old_to_new_state
        lr_idl = collected.left_unmap_quanta
        lr_idr = collected.right_unmap_quanta
        
        map_tensor = {block.q_labels[1]: block for block in tensor.blocks}
        
        blocks = []
        for k, js in enumerate(otn):
            q = collected.quanta[k]
            q_fused = BlockSymmetry.from_spin_quantum(q)
            if q_fused not in map_tensor:
                continue
            reduced_collected = map_tensor[q_fused].reduced
            idx_rot = 0
            for j in js:
                assert lr.quanta[j] == q
                sqs = [l.quanta[lr_idl[j]], r.quanta[lr_idr[j]], q]
                q_labels = tuple(BlockSymmetry.from_spin_quantum(sq) for sq in sqs)
                red_shape = (l.n_states[lr_idl[j]], r.n_states[lr_idr[j]], -1)
                rot_l_sh = red_shape[0] * red_shape[1]
                reduced = np.array(reduced_collected[idx_rot:idx_rot + rot_l_sh, :])
                idx_rot += rot_l_sh
                blocks.append(SubTensor(q_labels, reduced.reshape(red_shape)))
            assert idx_rot == reduced_collected.shape[0]
        
        t = Tensor(blocks)
        t.build_rank3_cg()
        t.sort()
        return t
    
    def unfuse_right(self, i, tensor):
        """
        Unfuse right index of rank-2 tensor in one site of MPS.
        
        Args:
            i : int
                Site index of the tensor in MPS.
            tensor : class:`Tensor`
                Rank-2 tensor with right index fused.
        
        Returns:
            tensor : class:`Tensor`
                Rank-3 tensor.
        """
        collected = self.right_state_info_no_trunc[i]
        l = self.right_state_info[i].left_state_info
        r = self.right_state_info[i].right_state_info
        lr = self.right_state_info[i].uncollected_state_info
        
        otn = collected.old_to_new_state
        lr_idl = collected.left_unmap_quanta
        lr_idr = collected.right_unmap_quanta
        
        map_tensor = {block.q_labels[0]: block for block in tensor.blocks}
        
        blocks = []
        for k, js in enumerate(otn):
            q = collected.quanta[k]
            q_fused = self.target + (-BlockSymmetry.from_spin_quantum(q))
            if q_fused not in map_tensor:
                continue
            reduced_collected = map_tensor[q_fused].reduced
            idx_rot = 0
            for j in js:
                assert lr.quanta[j] == q
                sqs = [q, l.quanta[lr_idl[j]], r.quanta[lr_idr[j]]]
                q_labels = tuple(BlockSymmetry.from_spin_quantum(sq) for sq in sqs)
                q_labels = (self.target + (-q_labels[0]), q_labels[1], self.target + (-q_labels[2]))
                assert isinstance(q_labels[0], DirectProdGroup)
                assert isinstance(q_labels[2], DirectProdGroup)
                red_shape = (-1, l.n_states[lr_idl[j]], r.n_states[lr_idr[j]])
                rot_l_sh = red_shape[1] * red_shape[2]
                reduced = np.array(reduced_collected[:, idx_rot:idx_rot + rot_l_sh])
                idx_rot += rot_l_sh
                blocks.append(SubTensor(q_labels, reduced.reshape(red_shape)))
            assert idx_rot == reduced_collected.shape[1]
        
        t = Tensor(blocks)
        t.build_rank3_cg()
        t.sort()
        return t
    
    # no cgs will be generated for twodot
    # expressed in 2-index tensor, can be used for svd
    def from_wavefunction_fused(self, i, wfn):
        """
        Construct rank-2 :class:`Tensor` with fused indices from Wavefunction (block code).
        
        
        Args:
            i : int
                Site index of first/left dot.
            wfn : Wavefunction
                Wavefunction.
        
        Returns:
            tensor : class:`Tensor`
                In two-dot scheme, the rank-2 tensor representing two-dot object.
                Both left and right rank indices are fused. No CG factor are generated.
                One-dot scheme is not implemented.
        """
        if not wfn.onedot:
            st_l = self.left_state_info_no_trunc[i]
            st_r = self.right_state_info_no_trunc[i + 1]
            
            blocks = []
            for (il, ir), mat in wfn.non_zero_blocks:
                q_labels = (BlockSymmetry.from_spin_quantum(st_l.quanta[il]),
                    self.target + (-BlockSymmetry.from_spin_quantum(st_r.quanta[ir])))
                reduced = mat.ref.copy()
                assert reduced.shape[0] == st_l.n_states[il]
                assert reduced.shape[1] == st_r.n_states[ir]
                blocks.append(SubTensor(q_labels, reduced))
            return Tensor(blocks)
        else:
            assert False
    
    # no cgs will be generated for twodot
    # expressed in direct product 4-index tensor
    # may not be useful
    def from_wavefunction(self, i, wfn):
        """
        Construct rank-4 :class:`Tensor` with unfused indices from Wavefunction (block code).
        
        
        Args:
            i : int
                Site index of first/left dot.
            wfn : Wavefunction
                Wavefunction.
        
        Returns:
            tensor : class:`Tensor`
                In two-dot scheme, the rank-4 tensor representing two-dot object.
                Both left and right rank indices are unfused. No CG factors are generated.
                One-dot scheme is not implemented.
        """
        if not wfn.onedot:
            st_l = self.left_state_info_no_trunc[i]
            st_r = self.right_state_info_no_trunc[i + 1]
            
            otn_l = st_l.old_to_new_state
            ll = st_l.left_state_info
            lr = st_l.right_state_info
            l_idl = st_l.left_unmap_quanta
            l_idr = st_l.right_unmap_quanta
            
            otn_r = st_r.old_to_new_state
            rl = st_r.left_state_info
            rr = st_r.right_state_info
            r_idl = st_r.left_unmap_quanta
            r_idr = st_r.right_unmap_quanta
            
            blocks = []
            for (il, ir), mat in wfn.non_zero_blocks:
                idx_l = 0
                for kl in otn_l[il]:
                    idx_r = 0
                    for kr in otn_r[ir]:
                        sqs = [ll.quanta[l_idl[kl]], lr.quanta[l_idr[kl]], rl.quanta[r_idl[kr]], rr.quanta[r_idr[kr]]]
                        q_labels = tuple(BlockSymmetry.from_spin_quantum(sq) for sq in sqs[:-1])
                        q_labels = q_labels + (self.target + (-BlockSymmetry.from_spin_quantum(sqs[-1])), )
                        red_shape = (ll.n_states[l_idl[kl]], lr.n_states[l_idr[kl]],
                                     rl.n_states[r_idl[kr]], rr.n_states[r_idr[kr]])
                        l_sh = red_shape[0] * red_shape[1]
                        r_sh = red_shape[2] * red_shape[3]
                        reduced = np.array(mat.ref[idx_l:idx_l + l_sh, idx_r:idx_r + r_sh])
                        idx_r += r_sh
                        blocks.append(SubTensor(q_labels, reduced.reshape(red_shape)))
                    assert idx_r == mat.cols
                    idx_l += l_sh
                assert idx_l == mat.rows
            return Tensor(blocks)
        else:
            assert False
    
    def from_left_rotation_matrix(self, i, rot):
        """
        Translate rotation matrix (block code) in left block to rank-3 :class:`Tensor`.
        
        Args:
            i : int
                Site index.
            rot : VectorMatrix
                Rotation matrix, defining the transformation
                from untruncated (but collected) basis to truncated basis.
        
        Returns:
            tensor : class:`Tensor`
        """
        collected = self.left_state_info_no_trunc[i]
        l = self.left_state_info[i].left_state_info
        r = self.left_state_info[i].right_state_info
        lr = self.left_state_info[i].uncollected_state_info
        
        otn = collected.old_to_new_state
        lr_idl = collected.left_unmap_quanta
        lr_idr = collected.right_unmap_quanta
        
        blocks = []
        for k, js in enumerate(otn):
            if rot[k].cols == 0:
                continue
            idx_rot = 0
            for j in js:
                q = lr.quanta[j]
                sqs = [l.quanta[lr_idl[j]], r.quanta[lr_idr[j]], q]
                q_labels = tuple(BlockSymmetry.from_spin_quantum(sq) for sq in sqs)
                red_shape = (l.n_states[lr_idl[j]], r.n_states[lr_idr[j]], -1)
                rot_l_sh = red_shape[0] * red_shape[1]
                reduced = np.array(rot[k].ref[idx_rot:idx_rot + rot_l_sh, :])
                idx_rot += rot_l_sh
                blocks.append(SubTensor(q_labels, reduced.reshape(red_shape)))
            assert idx_rot == rot[k].rows
        t = Tensor(blocks)
        t.build_rank3_cg()
        t.sort()
        return t
    
    def from_right_rotation_matrix(self, i, rot):
        """
        Translate rotation matrix (block code) in right block to rank-3 :class:`Tensor`.
        
        Args:
            i : int
                Site index.
            rot : VectorMatrix
                Rotation matrix, defining the transformation
                from untruncated (but collected) basis to truncated basis.
        
        Returns:
            tensor : class:`Tensor`
        """
        collected = self.right_state_info_no_trunc[i]
        l = self.right_state_info[i].left_state_info
        r = self.right_state_info[i].right_state_info
        lr = self.right_state_info[i].uncollected_state_info
        
        otn = collected.old_to_new_state
        lr_idl = collected.left_unmap_quanta
        lr_idr = collected.right_unmap_quanta
        
        blocks = []
        for k, js in enumerate(otn):
            if rot[k].cols == 0:
                continue
            idx_rot = 0
            for j in js:
                q = lr.quanta[j]
                sqs = [q, l.quanta[lr_idl[j]], r.quanta[lr_idr[j]]]
                q_labels = tuple(BlockSymmetry.from_spin_quantum(sq) for sq in sqs)
                q_labels = (self.target + (-q_labels[0]), q_labels[1], self.target + (-q_labels[2]))
                assert isinstance(q_labels[0], DirectProdGroup)
                assert isinstance(q_labels[2], DirectProdGroup)
                red_shape = (-1, l.n_states[lr_idl[j]], r.n_states[lr_idr[j]])
                rot_l_sh = red_shape[1] * red_shape[2]
                reduced = np.array(rot[k].ref[idx_rot:idx_rot + rot_l_sh, :].T)
                idx_rot += rot_l_sh
                blocks.append(SubTensor(q_labels, reduced.reshape(red_shape)))
            assert idx_rot == rot[k].rows
        t = Tensor(blocks)
        t.build_rank3_cg()
        t.sort()
        return t
    
    def get_left_rotation_matrix(self, i, tensor):
        """
        Translate rank-3 :class:`Tensor` to rotation matrix (block code) in left block.
        
        Args:
            i : int
                Site index.
            tensor : class:`Tensor`
                MPS tensor.
        
        Returns:
            rot : VectorMatrix
                Rotation matrix, defining the transformation
                from untruncated (but collected) basis to truncated basis.
        """
        collected = self.left_state_info_no_trunc[i]
        l = self.left_state_info[i].left_state_info
        r = self.left_state_info[i].right_state_info
        lr = self.left_state_info[i].uncollected_state_info
        
        otn = collected.old_to_new_state
        lr_idl = collected.left_unmap_quanta
        lr_idr = collected.right_unmap_quanta
        
        map_tensor = {block.q_labels: block for block in tensor.blocks}
        
        rot = []
        for js in otn:
            red = []
            for j in js:
                q = lr.quanta[j]
                sqs = [l.quanta[lr_idl[j]], r.quanta[lr_idr[j]], q]
                q_labels = tuple(BlockSymmetry.from_spin_quantum(sq) for sq in sqs)
                if q_labels in map_tensor:
                    block = map_tensor[q_labels]
                    a = block.reduced.shape[0] * block.reduced.shape[1]
                    b = block.reduced.shape[2]
                    reduced = block.reduced.reshape((a, b))
                    red.append(reduced)
            if len(red) == 0:
                rot.append(Matrix())
            else:
                rot.append(Matrix(np.ascontiguousarray(np.concatenate(red, axis=0))))
        
        return VectorMatrix(rot)
    
    def get_right_rotation_matrix(self, i, tensor):
        """
        Translate rank-3 :class:`Tensor` to rotation matrix (block code) in right block.
        
        Args:
            i : int
                Site index.
            tensor : class:`Tensor`
                MPS tensor.
        
        Returns:
            rot : VectorMatrix
                Rotation matrix, defining the transformation
                from untruncated (but collected) basis to truncated basis.
        """
        collected = self.right_state_info_no_trunc[i]
        l = self.right_state_info[i].left_state_info
        r = self.right_state_info[i].right_state_info
        lr = self.right_state_info[i].uncollected_state_info
        
        otn = collected.old_to_new_state
        lr_idl = collected.left_unmap_quanta
        lr_idr = collected.right_unmap_quanta
        
        map_tensor = {block.q_labels: block for block in tensor.blocks}
        
        rot = []
        for js in otn:
            red = []
            for j in js:
                q = lr.quanta[j]
                sqs = [q, l.quanta[lr_idl[j]], r.quanta[lr_idr[j]]]
                q_labels = tuple(BlockSymmetry.from_spin_quantum(sq) for sq in sqs)
                q_labels = (self.target + (-q_labels[0]), q_labels[1], self.target + (-q_labels[2]))
                assert isinstance(q_labels[0], DirectProdGroup)
                assert isinstance(q_labels[2], DirectProdGroup)
                if q_labels in map_tensor:
                    block = map_tensor[q_labels]
                    a = block.reduced.shape[1] * block.reduced.shape[2]
                    b = block.reduced.shape[0]
                    reduced = block.reduced.reshape((b, a)).T
                    red.append(reduced)
            if len(red) == 0:
                rot.append(Matrix())
            else:
                rot.append(Matrix(np.ascontiguousarray(np.concatenate(red, axis=0))))
        
        return VectorMatrix(rot)

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
        mpo : numpy.ndarray([[OpExpression]], dtype=object)
            A 1x1 numpy array containing symbolic expression of the (super block) Hamiltonian.
        st : StateInfo
            StateInfo of super block.
        diag_mat : DiagonalMatrix
            Diagonal elements of (super block) Hamiltonian, in flatten form with no quantum labels.
    """
    def __init__(self, mpo):
        self.mpo = mpo
        self.st = state_tensor_product_target(mpo.left_op_names, mpo.right_op_names)
        self.diag_mat = BlockEvaluation.expr_diagonal_eval(mpo.mat[0, 0], mpo.ops[0], mpo.ops[1], self.st)
    
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
        BlockEvaluation.expr_multiply_eval(self.mpo.mat[0, 0], self.mpo.ops[0], self.mpo.ops[1],
            other.data, result.data, self.st)

class BlockEvaluation:
    """Explicit evaluation of symbolic expression for operators."""
    @classmethod
    def tensor_rotate(self, mpo, old_st, new_st, rmat):
        """
        Transform basis of MPO using rotation matrix.
        
        Args:
            mpo : BlockMPO
                One-site MPO in (untruncated) old basis.
            old_st : StateInfo
                Old (untruncated) basis.
            new_st : StateInfo
                New (truncated) basis.
            rmat : VectorMatrix
                Rotation matrix.
        
        Returns:
            new_mpo : BlockMPO
                One-site MPO in (truncated) new basis.
        """
        new_ops = {}
        for k, v in mpo.ops.items():
            nmat = StackSparseMatrix()
            nmat.delta_quantum = v.delta_quantum
            nmat.fermion = v.fermion
            nmat.allocate(new_st)
            nmat.initialized = True
            state_info = VectorStateInfo([old_st, new_st])
            assert v.rows == v.cols and v.rows == len(old_st.quanta)
            assert nmat.rows == nmat.cols and nmat.rows == len(new_st.quanta)
            assert len(rmat) == len(old_st.quanta)
            assert len([r for r in rmat if r.cols != 0]) == len(new_st.quanta)
            tensor_rotate(v, nmat, state_info, rmat)
            new_ops[k] = nmat
        return mpo.__class__(mat=mpo.mat, ops=new_ops, tags=mpo.tags,
                             lop=mpo.left_op_names,
                             rop=mpo.right_op_names,
                             contractor=mpo.contractor)
    
    @classmethod
    def left_rotate(self, mpo, mps, info, i):
        """Perform rotation <MPS|MPO|MPS> for left block.
        
        Args:
            mpo : BlockMPO
                One-site MPO in (untruncated) old basis.
            mps : MPS
                One-site MPS.
            info : MPSInfo
                MPSInfo object.
            i : int
                Site index.
        
        Returns:
            new_mpo : BlockMPO
                One-site MPO in (truncated) new basis.
        """
        old_st = info.left_state_info_no_trunc[i]
        new_st = info.left_state_info[i]
        rmat = info.get_left_rotation_matrix(i, mps)
        return self.tensor_rotate(mpo, old_st, new_st, rmat)
    
    @classmethod
    def right_rotate(self, mpo, mps, info, i):
        """Perform rotation <MPS|MPO|MPS> for right block.
        
        Args:
            mpo : BlockMPO
                One-site MPO in (untruncated) old basis.
            mps : MPS
                One-site MPS.
            info : MPSInfo
                MPSInfo object.
            i : int
                Site index.
        
        Returns:
            new_mpo : BlockMPO
                One-site MPO in (truncated) new basis.
        """
        old_st = info.right_state_info_no_trunc[i]
        new_st = info.right_state_info[i]
        rmat = info.get_right_rotation_matrix(i, mps)
        return self.tensor_rotate(mpo, old_st, new_st, rmat)
    
    @classmethod
    def left_contract(self, mpol, mpo, info, i):
        """Perform blocking MPO x MPO for left block.
        
        Args:
            mpol: BlockMPO
                MPO at previous left block.
            mpo : BlockMPO
                MPO at dot block.
            info : MPSInfo
                MPSInfo object.
            i : int
                Site index.
        
        Returns:
            new_mpo : BlockMPO
                MPO in untruncated basis in current left block.
        """
        new_mat = mpol.mat @ mpo.mat
        st = info.left_state_info_no_trunc[i]
        new_ops = {}
        for j in range(new_mat.shape[1]):
            ql = mpo.right_op_names[j].q_label
            if mpo.right_op_names[j].sign == -1:
                new_ops[-mpo.right_op_names[j]] = self.expr_eval(-new_mat[0, j], mpol.ops, mpo.ops, st, ql)
            else:
                new_ops[mpo.right_op_names[j]] = self.expr_eval(new_mat[0, j], mpol.ops, mpo.ops, st, ql)
        return mpo.__class__(mat=mpo.right_op_names.reshape(new_mat.shape),
                             ops=new_ops, tags=mpo.tags,
                             lop=mpol.left_op_names,
                             rop=mpo.right_op_names,
                             contractor=mpo.contractor)
    
    @classmethod
    def right_contract(self, mpor, mpo, info, i):
        """Perform blocking MPO x MPO for right block.
        
        Args:
            mpor: BlockMPO
                MPO at previous right block.
            mpo : BlockMPO
                MPO at dot block.
            info : MPSInfo
                MPSInfo object.
            i : int
                Site index.
        
        Returns:
            new_mpo : BlockMPO
                MPO in untruncated basis in current right block.
        """
        new_mat = mpo.mat @ mpor.mat
        st = info.right_state_info_no_trunc[i]
        new_ops = {}
        for j in range(new_mat.shape[0]):
            ql = mpo.left_op_names[j].q_label
            if mpo.left_op_names[j].sign == -1:
                new_ops[-mpo.left_op_names[j]] = self.expr_eval(-new_mat[j, 0], mpo.ops, mpor.ops, st, ql)
            else:
                new_ops[mpo.left_op_names[j]] = self.expr_eval(new_mat[j, 0], mpo.ops, mpor.ops, st, ql)
        return mpo.__class__(mat=mpo.left_op_names.reshape(new_mat.shape),
                             ops=new_ops, tags=mpo.tags,
                             lop=mpo.left_op_names,
                             rop=mpor.right_op_names,
                             contractor=mpo.contractor)
    
    @classmethod
    def left_right_contract(self, mpol, mpor, info, i):
        """Symbolically construct the super block MPO.
        
        Args:
            mpol: BlockMPO
                MPO at (enlarged) left block.
            mpor : BlockMPO
                MPO at (enlarged) right block.
            info : MPSInfo
                MPSInfo object.
            i : int
                Site index of first/left dot block.
        
        Returns:
            mpo : BlockMPO
                MPO for super block.
                This method does not evaluate the super block MPO experssion.
        """
        new_mat = mpol.mat @ mpor.mat
        st_l = info.left_state_info_no_trunc[i]
        st_r = info.right_state_info_no_trunc[i + 1]
        assert new_mat.shape == (1, 1)
        return mpol.__class__(mat=new_mat, ops=(mpol.ops, mpor.ops),
                              tags={'_HAM'},
                              lop=st_l, rop=st_r, contractor=mpol.contractor)
    
    @classmethod
    def expr_diagonal_eval(self, expr, a, b, st):
        """
        Evaluate the diagonal elements of the result of a symbolic operator expression.
        The diagonal elements are required for perconditioning in Davidson algorithm.
        
        Args:
            expr : OpString or OpSum
                The operator expression to evaluate.
            a : dict(OpElement -> StackSparseMatrix)
                A map from operator symbol in left block to its matrix representation.
            b : dict(OpElement -> StackSparseMatrix)
                A map from operator symbol in right block to its matrix representation.
            st : StateInfo
                StateInfo in which the result of the operator expression is represented.
        
        Returns:
            diag : DiagonalMatrix
        """
        diag = DiagonalMatrix()
        diag.resize(st.n_total_states)
        diag.ref[:] = 0.0
        state_info = VectorStateInfo([st.left_state_info, st.right_state_info, st])
        if isinstance(expr, OpString):
            assert len(expr.ops) == 2
            if expr.ops[0] == OpElement(OpNames.I, ()):
                tensor_trace_diagonal(b[expr.ops[1]], diag, state_info, False, float(expr.sign))
            elif expr.ops[1] == OpElement(OpNames.I, ()):
                tensor_trace_diagonal(a[expr.ops[0]], diag, state_info, True, float(expr.sign))
            else:
                aq, bq = a[expr.ops[0]].delta_quantum[0], b[expr.ops[1]].delta_quantum[0]
                op_q = (aq + bq)[0]
                if expr.sign == -1:
                    scale = get_commute_parity(aq, bq, op_q)
                else:
                    scale = 1.0
                tensor_product_diagonal(a[expr.ops[0]], b[expr.ops[1]], diag, state_info, scale)
            return diag
        elif isinstance(expr, OpSum):
            diag = self.expr_diagonal_eval(expr.strings[0], a, b, st)
            for x in expr.strings[1:]:
                diag = diag + self.expr_diagonal_eval(x, a, b, st)
            return diag
        else:
            assert False
    
    @classmethod
    def expr_multiply_eval(self, expr, a, b, c, nwave, st):
        """
        Evaluate the result of a symbolic operator expression applied on a wavefunction.
        
        Args:
            expr : OpString or OpSum
                The operator expression.
            a : dict(OpElement -> StackSparseMatrix)
                A map from operator symbol in left block to its matrix representation.
            b : dict(OpElement -> StackSparseMatrix)
                A map from operator symbol in right block to its matrix representation.
            c : Wavefunction
                The input wavefuction.
            nwave : Wavefunction
                The output wavefuction.
            st : StateInfo
                StateInfo in which the wavefuction is represented.
        """
        if isinstance(expr, OpString):
            assert len(expr.ops) == 2
            if expr.ops[0] == OpElement(OpNames.I, ()):
                tensor_trace_multiply(b[expr.ops[1]], c, nwave, st, False, float(expr.sign))
            elif expr.ops[1] == OpElement(OpNames.I, ()):
                tensor_trace_multiply(a[expr.ops[0]], c, nwave, st, True, float(expr.sign))
            else:
                # TODO here we did not consider S != 0 products
                aq, bq = a[expr.ops[0]].delta_quantum[0], b[expr.ops[1]].delta_quantum[0]
                op_q = (aq + bq)[0]
                # with SU(2), it is not simply fermonic sign
                # for example, when exchange two S=1/2 fermonic operators, scale = 1 not -1
                if expr.sign == -1:
                    scale = get_commute_parity(aq, bq, op_q)
                else:
                    scale = 1.0
                tensor_product_multiply(a[expr.ops[0]], b[expr.ops[1]], c, nwave, st, op_q, scale)
        elif isinstance(expr, OpSum):
            self.expr_multiply_eval(expr.strings[0], a, b, c, nwave, st)
            twave = Wavefunction()
            twave.initialize_from(nwave)
            for x in expr.strings[1:]:
                twave.clear()
                self.expr_multiply_eval(x, a, b, c, twave, st)
                tensor_scale_add(1.0, twave, nwave, st)
            twave.deallocate()
        else:
            assert False
    
    @classmethod
    def expr_eval(self, expr, a, b, st, q_label):
        """
        Evaluate the result of a symbolic operator expression.
        The operator expression is usually a sum of direct products of
        operators in left and right blocks.
        
        Args:
            expr : OpString or OpSum
                The operator expression to evaluate.
            a : dict(OpElement -> StackSparseMatrix)
                A map from operator symbol in left block to its matrix representation.
            b : dict(OpElement -> StackSparseMatrix)
                A map from operator symbol in right block to its matrix representation.
            st : StateInfo
                StateInfo in which the result of the operator expression is represented.
            q_label : DirectProdGroup
                Quantum label of the result operator
                (indicating how it changes the state quantum labels).
        
        Returns:
            nmat : StackSparseMatrix
        """
        state_info = VectorStateInfo([st.left_state_info, st.right_state_info, st])
        if isinstance(expr, OpString):
            assert len(expr.ops) == 2
            nmat = StackSparseMatrix()
            if expr.ops[0] == OpElement(OpNames.I, ()):
                nmat.delta_quantum = b[expr.ops[1]].delta_quantum
                nmat.fermion = b[expr.ops[1]].fermion
                nmat.allocate(st)
                nmat.initialized = True
                assert q_label == BlockSymmetry.from_spin_quantum(nmat.delta_quantum[0])
                assert b[expr.ops[1]].rows == b[expr.ops[1]].cols
                assert b[expr.ops[1]].rows == len(state_info[1].quanta)
                assert nmat.rows == nmat.cols
                assert nmat.rows == len(state_info[2].quanta)
                tensor_trace(b[expr.ops[1]], nmat, state_info, False, float(expr.sign))
            elif expr.ops[1] == OpElement(OpNames.I, ()):
                nmat.delta_quantum = a[expr.ops[0]].delta_quantum
                nmat.fermion = a[expr.ops[0]].fermion
                nmat.allocate(st)
                nmat.initialized = True
                assert q_label == BlockSymmetry.from_spin_quantum(nmat.delta_quantum[0])
                assert a[expr.ops[0]].rows == a[expr.ops[0]].cols
                assert a[expr.ops[0]].rows == len(state_info[0].quanta)
                assert nmat.rows == nmat.cols
                assert nmat.rows == len(state_info[2].quanta)
                tensor_trace(a[expr.ops[0]], nmat, state_info, True, float(expr.sign))
            else:
                # TODO here we did not consider S != 0 products
                aq, bq = a[expr.ops[0]].delta_quantum[0], b[expr.ops[1]].delta_quantum[0]
                nmat.delta_quantum = VectorSpinQuantum([(aq + bq)[0]])
                nmat.fermion = a[expr.ops[0]].fermion ^ b[expr.ops[1]].fermion
                assert q_label == BlockSymmetry.from_spin_quantum(nmat.delta_quantum[0])
                if expr.sign == -1:
                    scale = get_commute_parity(aq, bq, nmat.delta_quantum[0])
                else:
                    scale = 1.0
                nmat.allocate(st)
                nmat.initialized = True
                tensor_product(a[expr.ops[0]], b[expr.ops[1]], nmat, state_info, scale)
            return nmat
        elif isinstance(expr, OpSum):
            nmat = self.expr_eval(expr.strings[0], a, b, st, q_label)
            assert nmat.conjugacy == 'n'
            for x in expr.strings[1:]:
                t = self.expr_eval(x, a, b, st, q_label)
                tensor_scale_add(1.0, t, nmat, st)
                t.deallocate()
            return nmat
        else:
            assert False
    
    @classmethod
    def eigen_values(self, mat):
        """Return all eigenvalues of a StackSparseMatrix."""
        mat = mpo.ops[op]
        evs = []
        for k, v in mat.non_zero_blocks:
            p, pp = np.linalg.eigh(v.ref)
            ppp = sorted(zip(p, pp.T), key=lambda x : x[0])
            evs.append(ppp[0][0])
        evs.sort()
        return np.array(evs)


class BlockSymmetry:
    """Including functions for translating quantum label related objects."""
    @classmethod
    def to_spin_quantum(self, dpg):
        """Translate from :class:`DirectProdGroup` to SpinQuantum (block code)."""
        if Global.dmrginp.is_spin_adapted:
            subg = [ParticleN, SU2, PointGroup]
            if dpg.ng != 3 or not all(isinstance(ir, c) for c, ir in zip(subg, dpg.irs)):
                raise BlockError('Representation not supported by block code.')
        else:
            subg = [ParticleN, SZ, PointGroup]
            if dpg.ng != 3 or not all(isinstance(ir, c) for c, ir in zip(subg, dpg.irs)):
                raise BlockError('Representation not supported by block code.')
        return SpinQuantum(dpg.irs[0].ir, SpinSpace(dpg.irs[1].ir), IrrepSpace(dpg.irs[2].ir))

    # translate SpinQuantum (block code) to a DirectProdGroup object
    @classmethod
    def from_spin_quantum(self, sq):
        """Translate from SpinQuantum (block code) to :class:`DirectProdGroup`."""
        PG = point_group(Global.point_group)
        if Global.dmrginp.is_spin_adapted:
            return ParticleN(sq.n) * SU2(sq.s.irrep) * PG(sq.symm.irrep)
        else:
            return ParticleN(sq.n) * SZ(sq.s.irrep) * PG(sq.symm.irrep)

    # translate a [(DirectProdGroup, int)] to StateInfo (block code)
    @classmethod
    def to_state_info(self, states):
        """Translate from [(:class:`DirectProdGroup`, int)] to StateInfo (block code)."""
        qs = VectorSpinQuantum()
        ns = VectorInt()
        for k, v in states:
            qs.append(self.to_spin_quantum(k))
            ns.append(v)
        return StateInfo(qs, ns)

    @classmethod
    def initial_state_info(self, i=0):
        """Return StateInfo for site basis at site i."""
        return MPS.site_blocks[i].ket_state_info


class BlockHamiltonian:
    """
    Initialization of block code.
    
    Attributes:
        output_level : int
            Output level of block code.
        point_group : str
            Point group of molecule.
        n_sites : int
            Number of sites/orbitals.
        n_electrons : int
            Number of electrons.
        target_s : Fraction
            SU(2) quantum number of target state.
        spatial_syms : [int]
            point group irrep number at each site.
        target_spatial_sym : int
            point group irrep number of target state.
        dot : int
            two-dot (2) or one-dot (1) scheme.
        t : TInt
            One-electron integrals.
        v : VInt
            Two-electron integrals.
        e : float
            Const term in energy.
    """

    def __init__(self, *args, **kwargs):

        Global.dmrginp.output_level = -1
        self.output_level = -1

        if 'output_level' in kwargs:
            Global.dmrginp.output_level = kwargs['output_level']
            self.output_level = kwargs['output_level']
            del kwargs['output_level']

        file_input = False

        if len(args) == 1 and isinstance(args[0], str):
            read_input(args[0])
            self.output_level = Global.dmrginp.output_level
            file_input = True
            if len(kwargs) != 0:
                raise BlockError(
                    'Cannot accept additional arguments if initialized by input file name.')
        elif len(args) != 0:
            raise BlockError(
                'Unknown argument for initialize Block Hamiltonian.')

        if not file_input:
            input = {'noreorder': '', 'maxM': '500', 'maxiter': '30', 'sym': 'c1',
                     'hf_occ': 'integral', 'schedule': 'default'}

            if 'orbitals' in kwargs or 'fcidump' in kwargs:
                fd_name = kwargs['orbitals' if 'orbitals' in kwargs else 'fcidump']
                opts, (t, v, e) = read_fcidump(fd_name)

                input['nelec'] = opts['nelec']
                input['spin'] = opts['ms2']
                input['irrep'] = opts['isym']
                
                self.t = t
                self.v = v
                self.e = e

            for k, v in kwargs.items():
                if k in ['orbitals', 'fcidump']:
                    input['orbitals'] = v
                elif k in ['sym', 'point_group']:
                    input['sym'] = v
                elif k == 'nelec':
                    input['nelec'] = str(v)
                elif k == 'spin':
                    input['spin'] = str(v)
                elif k == 'dot':
                    if v == 2:
                        input['twodot'] = ''
                    elif v == 1:
                        input['onedot'] = ''
                elif k == 'irrep':
                    input['irrep'] = str(v)
                elif k == 'output_level':
                    input['outputlevel'] = str(v)
                elif k == 'hf_occ':
                    input['hf_occ'] = str(v)
                elif k == 'max_iter':
                    input['maxiter'] = str(v)
                elif k == 'max_m':
                    input['maxM'] = str(v)
                elif k == 'spin_adapted':
                    if v == False:
                        input['nonspinadapted'] = ''

            Global.dmrginp = Input.read_input_contents(
                '\n'.join([k + ' ' + v for k, v in input.items()]))

            read_input("")

        Global.dmrginp.output_level = self.output_level

        self.point_group = Global.point_group
        self.n_sites = Global.dmrginp.slater_size // 2
        self.n_electrons = Global.dmrginp.n_electrons
        self.target_s = Fraction(Global.dmrginp.molecule_quantum.s.irrep, 2)
        self.spatial_syms = Global.dmrginp.spin_orbs_symmetry[::2]
        self.target_spatial_sym = Global.dmrginp.molecule_quantum.symm.irrep
        if Global.dmrginp.algorithm_type == AlgorithmTypes.TwoDotToOneDot:
            raise BlockError('Currently two dot to one dot is not supported.')
        self.dot = 1 if Global.dmrginp.algorithm_type == AlgorithmTypes.OneDot else 2

        init_stack_memory()
        
        MPS.site_blocks = VectorBlock([])
        MPS_init(True)
    
    def get_site_operators(self, i):
        """Return operator representations dict(OpElement -> StackSparseMatrix) at site i."""
        ops = {}
        block = Block(i, i, 0, False)
        
        mat = StackSparseMatrix()
        mat.deep_copy(block.ops[OpTypes.Hamiltonian].local_element_linear(0)[0])
        assert not mat.fermion
        ops[OpElement(OpNames.H, ())] = mat
        
        mat = StackSparseMatrix()
        mat.deep_copy(block.ops[OpTypes.Overlap].local_element_linear(0)[0])
        assert not mat.fermion
        ops[OpElement(OpNames.I, ())] = mat
        
        mat = StackSparseMatrix()
        mat.deep_copy(block.ops[OpTypes.Cre].local_element_linear(0)[0])
        assert mat.fermion
        ops[OpElement(OpNames.C, ())] = mat
        
        mat = StackSparseMatrix()
        mat.deep_copy(block.ops[OpTypes.Des].local_element_linear(0)[0])
        assert mat.fermion
        ops[OpElement(OpNames.D, ())] = mat
        
        for j in range(0, i):
            
            mat = StackSparseMatrix()
            mat.deep_copy(block.ops[OpTypes.Des].local_element_linear(0)[0])
            assert mat.fermion
            tensor_scale(self.t[j, i] * np.sqrt(2), mat)
            ops[OpElement(OpNames.S, (j, ))] = mat
            ql = mat.delta_quantum[0]
            ql.symm = IrrepSpace(self.spatial_syms[j])
            mat.delta_quantum = VectorSpinQuantum([ql])
            
            mat = StackSparseMatrix()
            mat.deep_copy(block.ops[OpTypes.Cre].local_element_linear(0)[0])
            assert mat.fermion
            tensor_scale(self.t[j, i] * np.sqrt(2), mat)
            ops[OpElement(OpNames.SD, (j, ))] = mat
            ql = mat.delta_quantum[0]
            ql.symm = IrrepSpace(self.spatial_syms[j])
            mat.delta_quantum = VectorSpinQuantum([ql])
            
            if self.spatial_syms[j] != self.spatial_syms[i]:
                assert np.isclose(self.t[j, i], 0.0)
        
        # TODO :: need to store Block to deallocate it later
        return ops
    
    @staticmethod
    def get_current_memory():
        """Return current stack memory position."""
        return get_current_stack_memory()
    
    @staticmethod
    def set_current_memory(m):
        """Reset current stack memory to given position."""
        return set_current_stack_memory(m)
    
    @staticmethod
    def release_memory():
        release_stack_memory()
    
    @staticmethod
    def block_operator_summary(block):
        """Return a summary str of operators included in one Block (block code)."""
        r = []
        for k, v in block.ops.items():
            r.append(repr(k) + " :: local_elements")
            g = []
            for vv in v.local_indices:
                g.append("%r -> %d" % (tuple(vv), len(v.local_element(*vv))))
            r.append(', '.join(g))
        return "\n".join(r)
