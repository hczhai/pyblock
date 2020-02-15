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
Matrix Product State for quantum chemistry calculations.
"""

from block import VectorInt, VectorVectorInt, VectorMatrix, Matrix
from block.symmetry import state_tensor_product
from block.operator import Wavefunction

from ..symmetry.symmetry import DirectProdGroup
from ..tensor.tensor import Tensor, SubTensor, TensorNetwork
from .core import BlockSymmetry
import numpy as np
import bisect
import collections

class LineCoupling:
    def __init__(self, n_sites, basis, empty, target):
        self.n_sites = n_sites
        self.basis = basis
        self.empty = empty
        self.target = target
        self.bond_dim = -1
        assert n_sites != 0
        self.left_dims = self._fill_from_left()
        self.right_dims = self._fill_from_right()
        if self.target is not None:
            self._filter()
        self.left_dims_fci = [d.copy() for d in self.left_dims]
        self.right_dims_fci = [d.copy() for d in self.right_dims]
    
    def _post_check_left(self):
        for d in range(0, self.n_sites):
            if d == 0:
                dd = self.tensor_product(None, self.basis[d])
            else:
                dd = self.tensor_product(self.left_dims[d - 1], self.basis[d])
            for k, v in self.left_dims[d].items():
                if k in dd and self.left_dims[d][k] > dd[k]:
                    self.left_dims[d][k] = dd[k]
    
    def _post_check_right(self):
        for d in range(self.n_sites - 1, -1, -1):
            if d == self.n_sites - 1:
                dd = self.tensor_product(self.basis[d], None)
            else:
                dd = self.tensor_product(self.basis[d], self.right_dims[d + 1])
            for k, v in self.right_dims[d].items():
                if k in dd and self.right_dims[d][k] > dd[k]:
                    self.right_dims[d][k] = dd[k]
    
    def _fill_from_left(self):
        dim_l = [None] * self.n_sites
        for d in range(0, self.n_sites):
            if d == 0:
                dim_l[d] = self.tensor_product(None, self.basis[d])
            else:
                dim_l[d] = self.tensor_product(dim_l[d - 1], self.basis[d])
        return dim_l
    
    def _fill_from_right(self):
        dim_r = [None] * self.n_sites
        for d in range(self.n_sites - 1, -1, -1):
            if d == self.n_sites - 1:
                dim_r[d] = self.tensor_product(self.basis[d], None)
            else:
                dim_r[d] = self.tensor_product(self.basis[d], dim_r[d + 1])
        return dim_r
    
    def tensor_product(self, p, q):
        r = collections.Counter()
        if p is not None and not isinstance(p, list):
            p = p.items()
        if q is not None and not isinstance(q, list):
            q = q.items()
        for pk, pv in ([(self.empty, 1)] if p is None else p):
            for rks, rv in ((pk + qk, pv * qv) for qk, qv in ([(self.empty, 1)] if q is None else q)):
                for rk in rks if isinstance(rks, list) else [rks]:
                    if self.target is None or rk <= self.target:
                        r[rk] += rv
        return r
    
    def _filter(self):
        for d in range(-1, self.n_sites):
            ld = self.left_dims[d] if d >= 0 else collections.Counter({self.empty: 1})
            rd = self.right_dims[d + 1] if d + 1 < self.n_sites else collections.Counter({self.empty: 1})
            for k, v in ld.copy().items():
                rk = self.target - k
                x = sum((rd[r] if r in rd else 0) for r in (rk if isinstance(rk, list) else [rk]))
                if x == 0:
                    del ld[k]
                else:
                    ld[k] = min(ld[k], x)
            for k, v in rd.copy().items():
                lk = self.target - k
                x = sum((ld[l] if l in ld else 0) for l in (lk if isinstance(lk, list) else [lk]))
                if x == 0:
                    del rd[k]
                else:
                    rd[k] = min(rd[k], x)
    
    def set_bond_dimension(self, m):
        """
        Truncate the renormalized basis, using the given bond dimension.
        Note that the ceiling is used for rounding for each quantum number,
        so the actual bond dimension is often larger than the given value.
        """
        self.bond_dim = m
        if m == -1:
            return
        for i in range(0, self.n_sites):
            x = sum(self.left_dims[i].values())
            if x > m:
                for k, v in self.left_dims[i].items():
                    self.left_dims[i][k] = int(np.ceil(v * m / x))
        for i in range(self.n_sites - 1, -1, -1):
            x = sum(self.right_dims[i].values())
            if x > m:
                for k, v in self.right_dims[i].items():
                    self.right_dims[i][k] = int(np.ceil(v * m / x))
        self._post_check_left()
        self._post_check_right()

class MPSInfo:
    def __init__(self, lcp):
        self.lcp = lcp
        self.n_sites = self.lcp.n_sites
        self.basis = [sorted(bs.items(), key=lambda x: x[0]) for bs in self.lcp.basis]
        self.left_block_basis = [sorted(ld.items(), key=lambda x: x[0]) for ld in self.lcp.left_dims]
        self.right_block_basis = [sorted(rd.items(), key=lambda x: x[0]) for rd in self.lcp.right_dims]
        self._init_state_info()
    
    def _init_state_info(self):
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
    def update_local_left_block_basis(self, i, left_block_basis):
        """
        Update renormalized basis at site i and associated StateInfo objects.
        
        This will update for left block with sites [0..i] and [0..i+1]
        
        Args:
            i : int
                Center site, for determining assocated left and right blocks.
            left_block_basis : [(DirectProdGroup, int)]
                Renormalized basis for left block with sites [0..i].
        """
        
        self.left_block_basis[i] = []
        for k, v in sorted(left_block_basis, key=lambda x: x[0]):
            self.left_block_basis[i].append((k, v))
        
        left = self.update_local_left_state_info(i)
        self.update_local_left_state_info(i + 1, left=left)
        
    # basis is a list of pairs: [(q_label, n_states)]
    def update_local_right_block_basis(self, i, right_block_basis):
        """
        Update renormalized basis at site i and associated StateInfo objects.
        
        This will update for right block with sites [i+1..:attr:`n_sites`-1] and  [i..:attr:`n_sites`-1].
        
        Args:
            i : int
                Center site, for determining assocated left and right blocks.
            left_block_basis : [(DirectProdGroup, int)]
                Renormalized basis for left block with sites [0..i].
            right_block_basis : [(DirectProdGroup, int)]
                Renormalized basis for right block with sites [i+1..:attr:`n_sites`-1].
        """
        
        self.right_block_basis[i + 1] = []
        for k, v in sorted(right_block_basis, key=lambda x: x[0]):
            self.right_block_basis[i + 1].append((k, v))
        
        right = self.update_local_right_state_info(i + 1)
        self.update_local_right_state_info(i, right=right)
    
    def get_left_state_info(self, i, left=None):
        """Construct StateInfo for left block [0..i] (used internally)"""
        if left is None:
            if i == 0:
                left = BlockSymmetry.to_state_info([(self.lcp.empty, 1)])
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
                right = BlockSymmetry.to_state_info([(self.lcp.empty, 1)])
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
    
    def get_wavefunction_fused(self, i, tensor, dot=2):
        """
        Construct Wavefunction (block code) from rank-2 :class:`Tensor`.
        
        Args:
            i : int
                Site index of first/left dot.
            tensors : [Tensor]
                Rank-2 :class:`Tensor` with fused indices.
            dot : int
                One-dot or two-dot (default) scheme.
        
        Returns:
            wfn : Wavefunction
        """
        if dot == 2:
            st_l = self.left_state_info_no_trunc[i]
            st_r = self.right_state_info_no_trunc[i + 1]
            
            target_state_info = BlockSymmetry.to_state_info([(self.lcp.target, 1)])

            wfn = Wavefunction()
            wfn.initialize(target_state_info.quanta, st_l, st_r, False)
            wfn.clear()
            
            map_tensor = {block.q_labels: block for block in tensor.blocks}
            
            for (il, ir), mat in wfn.non_zero_blocks:
                q_labels = (BlockSymmetry.from_spin_quantum(st_l.quanta[il]),
                    BlockSymmetry.from_spin_quantum(st_r.quanta[ir]))
                if q_labels in map_tensor:
                    mat.ref[:, :] = map_tensor[q_labels].reduced
            
            return wfn
        else:
            assert False
    
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
                    BlockSymmetry.from_spin_quantum(st_r.quanta[ir]))
                reduced = mat.ref.copy()
                assert reduced.shape[0] == st_l.n_states[il]
                assert reduced.shape[1] == st_r.n_states[ir]
                blocks.append(SubTensor(q_labels, reduced))
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
                q_labels = (q_labels[0], q_labels[1], q_labels[2])
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
                q_labels = (q_labels[0], q_labels[1], q_labels[2])
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
        
                    
class MPS(TensorNetwork):
    """Matrix Product State."""
    def __init__(self, lcp, center, dot=2):
        self.lcp = lcp
        self.n_sites = lcp.n_sites
        self.center = center
        self.dot = dot
        tensors = self._init_mps_tensors(lcp)
        super().__init__(tensors)
    
    def _init_mps_tensors(self, lcp):
        tensors = []
        l = collections.Counter({lcp.empty: 1})
        for i in range(0, self.center):
            tensors.append(Tensor.rank3_init_left(l, lcp.basis[i], lcp.left_dims[i]))
            tensors[-1].tags = {i}
            l = lcp.left_dims[i]
        r = collections.Counter({lcp.empty: 1})
        for i in range(lcp.n_sites - 1, self.center + self.dot - 1, -1):
            tensors.append(Tensor.rank3_init_right(r, lcp.basis[i], lcp.right_dims[i]))
            tensors[-1].tags = {i}
            r = lcp.right_dims[i]
        if self.dot == 1:
            ld = lcp.tensor_product(l, lcp.basis[self.center])
            rd = r
        elif self.dot == 2:
            ld = lcp.tensor_product(l, lcp.basis[self.center])
            rd = lcp.tensor_product(lcp.basis[self.center + 1], r)
        if self.dot != 0:
            tensors.append(Tensor.rank2_init_target(ld, rd, lcp.target))
            tensors[-1].tags = set(range(self.center, self.center + self.dot))
        return tensors
        
    def randomize(self):
        """Fill MPS reduced matrices with random numbers in [0, 1)."""
        for ts in self.tensors:
            ts.build_random()
            
    def fill_identity(self):
        """Fill MPS reduced matrices with identity matrices whenever possible."""
        for ts in self.tensors:
            ts.build_identity()
            
    def canonicalize(self):
        """Canonicalization."""
        for i in range(0, self.center):
            rs = self[i].left_canonicalize()
            if i + 1 < self.n_sites:
                ts = self.select({i + 1}).tensors[0]
                if i + 1 == self.center:
                    l = self.lcp.left_dims[i]
                    ts.unfuse_index(0, l, self.lcp.basis[i + 1])
                ts.left_multiply(rs)
                if i + 1 == self.center:
                    ld = self.lcp.tensor_product(l, self.lcp.basis[i + 1])
                    ts.fuse_index(0, ld, target=self.lcp.target)
        for i in range(self.n_sites - 1, self.center + self.dot - 1, -1):
            ls = self[i].right_canonicalize()
            if i - 1 >= 0:
                ts = self.select({i - 1}).tensors[0]
                if i - 1 == self.center + self.dot - 1:
                    r = self.lcp.right_dims[i]
                    ts.unfuse_index(1, self.lcp.basis[i - 1], r)
                ts.right_multiply(ls)
                if i - 1 == self.center + self.dot - 1:
                    rd = self.lcp.tensor_product(self.lcp.basis[i - 1], r)
                    ts.fuse_index(1, rd, target=self.lcp.target)
