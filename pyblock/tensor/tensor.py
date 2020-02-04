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
Block-sparse tensor and tensor network.
"""

import numpy as np
from itertools import accumulate, groupby


class SubTensor:
    """
    A block in block-sparse tensor.
    
    Attributes:
        q_labels : tuple(DirectProdGroup..)
            Quantum labels for this sub-tensor block.
            Each element in the tuple corresponds one rank of the tensor.
        rank : int
            Rank of the tensor. ``rank == len(q_labels)``.
        ng : int
            Number of sub-symmetry groups. ``ng == q_labels[0].ng``.
        reduced : numpy.ndarray
            Rank-:attr:`rank` dense reduced matrix.
        reduced_shape : tuple(int..)
            Shape of :attr:`reduced`
        cgs : list(numpy.ndarray)
            A list of CG factors of length :attr:`ng` with each element being
            Rank-:attr:`rank` dense reduced matrix representing
            CG coefficients for projected quantum numbers in each sub-symmetry group.
    """
    def __init__(self, q_labels=None, reduced=None, cgs=None):
        self.q_labels = q_labels if q_labels is not None else []
        # assuming that q_labels must be set at the beginning
        self.rank = len(q_labels)
        self.ng = 0 if self.rank == 0 else q_labels[0].ng
        self.reduced = reduced
        self.reduced_shape = [0] * \
            self.rank if reduced is None else reduced.shape
        self.cgs = cgs
        if self.rank != 0:
            if reduced is not None:
                assert len(self.reduced.shape) == self.rank
            assert all(q.ng == self.ng for q in q_labels)
            if self.cgs is not None:
                assert len(self.cgs) == self.ng
                assert all(len(cg.shape) == self.rank for cg in cgs)

    def build_rank3_cg(self):
        """Generate :attr:`cgs` for rank-3 tensor."""
        assert self.rank == 3
        syms = [ir.__class__ for ir in self.q_labels[0].irs]
        self.cgs = [syms[ig].clebsch_gordan(*[q.irs[ig] for q in self.q_labels])
                    for ig in range(self.ng)]

    def build_random(self):
        """Set reduced matrix with random numbers in [0, 1)."""
        self.reduced = np.random.random(self.reduced_shape)
    
    def add_noise(self, noise):
        """
        Add noise to reduced matrix by random numbers in [-0.5 * noise, 0.5 * noise).
        
        Args:
            noise : float
                prefactor for the noise.
        """
        self.reduced += (np.random.random(self.reduced_shape) - 0.5) * noise

    def build_zero(self):
        """Set reduced matrix to zero."""
        self.reduced = np.zeros(self.reduced_shape)

    # tranpose of operator
    # (ket, op, bra) -> (bra, -op, ket)
    @property
    def T(self):
        """Transpose."""
        q_ket, q_op, q_bra = self.q_labels
        return SubTensor(q_labels=(q_bra, -q_op, q_ket),
                         reduced=np.transpose(self.reduced, (2, 1, 0)),
                         cgs=[np.transpose(cg, (2, 1, 0)) for cg in self.cgs])

    def __mul__(self, o):
        return SubTensor(q_labels=self.q_labels, reduced=o * reduced, cgs=self.cgs)
    
    def __eq__(self, o):
        return self.q_labels == o.q_labels and np.allclose(self.reduced, o.reduced) \
            and all(np.allclose(cgx, cgy) for cgx, cgy in zip(self.cgs, o.cgs))

    def __repr__(self):
        return "(Q=) %r (R=) %r" % (self.q_labels, self.reduced)


class Tensor:
    """
    Block-sparse tensor.
    
    Attributes:
        blocks : list(SubTensor)
            A list of (non-zero) blocks.
        tags : set(str or int)
            Tags of the tensor for labeling
            the type or site index of the tensor.
        contractor :
            If not None, this is used to perform specialized tensor contraction
            and diagonalization.
    """
    # blocks: list of (non-zero) TensorBlock's
    # tags: list of tags for each rank
    def __init__(self, blocks=None, tags=None, contractor=None):
        self.blocks = blocks if blocks is not None else []
        self.tags = tags if tags is not None else set()
        if not isinstance(self.tags, set):
            self.tags = {self.tags}
        self.contractor = contractor

    def copy(self):
        """Shallow copy."""
        return Tensor(blocks=self.blocks, tags=self.tags.copy(),
                      contractor=self.contractor)
    
    def deep_copy(self):
        """Deep copy."""
        return Tensor(blocks=self.blocks[:], tags=self.tags.copy(),
                      contractor=self.contractor)

    @property
    def rank(self):
        """Rank of the tensor."""
        return 0 if len(self.blocks) == 0 else self.blocks[0].rank

    @property
    def ng(self):
        """Number of sub-symmetry groups."""
        return 0 if len(self.blocks) == 0 else self.blocks[0].ng

    @property
    def n_blocks(self):
        """Number of (non-zero) blocks."""
        return len(self.blocks)
    
    def modify(self, other):
        """Modify the blocks according to another Tensor's blocks."""
        self.blocks[:] = other.blocks

    # build the tensor by coupling states in pre and basis into states in post
    @staticmethod
    def rank3_init(pre, basis, post):
        """
        Build the MPS rank-3 tensor by coupling states
        in ``pre`` and ``basis`` into states in ``post``.
        :attr:`cgs` and :attr:`reduced` will not be calculated.
        
        Args:
            pre : dict(DirectProdGroup -> int)
                Left basis.
            basis : dict(DirectProdGroup -> int)
                Site basis.
            post : dict(DirectProdGroup -> int)
                Right basis. Right basis should be obtained from
                truncating the direct product of left and site basis.
        
        Returns:
            tensor : Tensor
        """
        blocks = []
        for kp, vp in sorted(pre.items(), key=lambda x:x[0]):
            for kb, vb in sorted(basis.items(), key=lambda x:x[0]):
                rs = kp + kb
                for kr in (rs if isinstance(rs, list) else [rs]):
                    if kr in sorted(post.keys()):
                        blocks.append(SubTensor(q_labels=(kp, kb, kr)))
                        blocks[-1].reduced_shape = [vp, vb, post[kr]]
        return Tensor(blocks)

    # repr is a list of dense matrices, in different op_q_labels
    # op_q_labels is a list of operator quantum numbers
    @staticmethod
    def operator_init(basis, repr, op_q_labels):
        """
        Build tensor for single-site primiary operator.
        When this is a tensor operator, ``repr`` and ``op_q_labels``
        will contain multiple items.
        :attr:`cgs` and :attr:`reduced` will be calculated.
        
        Args:
            basis : dict(DirectProdGroup -> int)
                Site basis.
            repr : list(numpy.ndarray)
                A list of dense matrices for different ``op_q_label`` s.
            op_q_labels : list(DirectProdGroup)
                A list of operator quantum labels.
        
        Returns:
            tensor : Tensor
                Operator Tensor.
        """
        blocks = []
        for q in range(len(repr.shape)):
            for i in range(repr[q].shape[0]):
                for j in range(repr[q].shape[1]):
                    if not np.isclose(repr[q][i, j], 0.0):
                        q_labels = (basis[j], op_q_labels[q], basis[i])
                        reduced = np.array([[[repr[q][i, j]]]], dtype=float)
                        blocks.append(SubTensor(q_labels, reduced))
        t = Tensor(blocks)
        t.build_rank3_cg()
        return t

    def build_rank3_cg(self):
        """Generate CG coefficients for rank-3 tensor."""
        for block in self.blocks:
            block.build_rank3_cg()

    def build_random(self):
        """Fill the reduced matrix with random numbers in [0, 1)."""
        for block in self.blocks:
            block.build_random()
    
    def add_noise(self, noise):
        """
        Add noise to reduced matrix by random numbers in [-0.5 * noise, 0.5 * noise).
        
        Args:
            noise : float
                prefactor for the noise.
        """
        for block in self.blocks:
            block.add_noise(noise)

    def build_zero(self):
        """Fill the reduced matrix with zero."""
        for block in self.blocks:
            block.build_zero()

    def build_identity(self):
        """Set the reduced matrix to identity. Not work for general situations."""
        assert self.rank == 3
        cur_idx = {}
        for block in self.blocks:
            q_labels_r = block.q_labels[2:]
            if q_labels_r not in cur_idx:
                cur_idx[q_labels_r] = 0
            k = cur_idx[q_labels_r]
            for i in range(block.reduced.shape[0]):
                for j in range(block.reduced.shape[1]):
                    # if MPS is initialized from LineCoupling (both_dir = False)
                    # then k will never bigger than block.reduced.shape[2]
                    # otherwise there is truncation in right labels
                    # then the corresponding extra left rows will be set zero
                    if k < block.reduced.shape[2]:
                        block.reduced[i, j, k] = 1.0
                        k += 1
            cur_idx[q_labels_r] = k
    
    # the indices in idx_l will be combined
    # the indices in idx_r will also be combined
    # then for each entry if q_label(idx_l) == q_label(idx_r), the term will be included
    @staticmethod
    def partial_trace(ts, idx_l, idx_r, target_q_labels=None):
        """
        Partial trace of a tensor.
        The indices in ``idx_l`` will be combined.
        The indices in ``idx_r`` will also be combined.
        Then for each tensor block with q_label[idx_l] == q_label[idx_r],
        the term will be included, with its reduced matrix traced in corresponding indices.
        
        Args:
            ts : Tensor
                Tensor to be traced.
            idx_l : list(int)
                Left traced indices.
            idx_r : list(int)
                Right traced indices.
            target_q_labels : None
                Defaults to None. Currently only the default is implemented.
        
        Returns:
            tensor : Tensor
                The resulting Tensor with indices ``idx_l`` and ``idx_r`` are traced.
        """
        out_idx = list(set(range(0, ts.rank)) - set(idx_l) - set(idx_r))

        trace_scr = list(range(0, ts.rank))
        for ia, ib in zip(idx_l, idx_r):
            trace_scr[ib] = trace_scr[ia]

        if target_q_labels is None:
            map_idx_out = {}
            for block in ts.blocks:
                sub_l = tuple(block.q_labels[id] for id in idx_l)
                sub_r = tuple(block.q_labels[id] for id in idx_r)
                if sub_l != sub_r:
                    continue
                outg = tuple(block.q_labels[id] for id in out_idx)
                mat = np.einsum(block.reduced, trace_scr)
                if outg not in map_idx_out:
                    cgs = [np.einsum(cg, trace_scr) for cg in block.cgs] \
                          if block.cgs is not None else None
                    map_idx_out[outg] = SubTensor(
                        q_labels=outg, reduced=mat, cgs=cgs)
                else:
                    map_idx_out[outg].reduced += mat
        else:
            raise TensorNetworkError('not implemented yet!')
        
        return Tensor(tensors=map_idx_out.values(), tags=ts.tags, contractor=ts.contractor)

    @staticmethod
    def contract(tsa, tsb, idxa, idxb, target_q_labels=None):
        """
        Contract two Tensor to form a new Tensor.
        
        Args:
            tsa : Tensor
                Tensor a, as left operand.
            tsb : Tensor
                Tensor b, as right operand.
            idxa : list(int)
                Indices to be contracted in tensor a.
            idxb : list(int)
                Indices to be contracted in tensor b.
            target_q_labels : None or DirectProdGroup or [DirectProdGroup]
                If None, this is contraction of states. For all other cases,
                this is contraction of operators.
                If DirectProdGroup, the resulting direct product operator
                will have the specific quantum label.
                If [DirectProdGroup], the resulting direct product operator
                will have mixed quantum labels (not very useful).
        
        Returns:
            tensor : Tensor
                The contracted Tensor.
        """
        out_idx_a = list(set(range(0, tsa.rank)) - set(idxa))
        out_idx_b = list(set(range(0, tsb.rank)) - set(idxb))

        map_idx_b = {}
        for block in tsb.blocks:
            subg = tuple(block.q_labels[id] for id in idxb)
            if subg not in map_idx_b:
                map_idx_b[subg] = []
            map_idx_b[subg].append(block)

        # if target is None, assuming target is the S=0 state
        # ie, all quantum numbers should be equal in contracted indices
        # this is the abelian or S=0 non-abelian case (MPO-MPS contraction case or MPS-MPS contraction case)
        # this works because MPS can be considered as an identity operator (with the operator index implicitly zero)
        # then any S q-number operator contract with S=0 operator, the operator q-number index will not change
        # then only need to contract other state representation indices.
        # MPS indices contraction auto handled by CGC. Operator contraction need to specify target.
        if target_q_labels is None:
            map_idx_out = {}
            for block_a in tsa.blocks:
                subg = tuple(block_a.q_labels[id] for id in idxa)
                if subg in map_idx_b:
                    outga = tuple(block_a.q_labels[id] for id in out_idx_a)
                    for block_b in map_idx_b[subg]:
                        outg = outga + \
                            tuple(block_b.q_labels[id] for id in out_idx_b)
                        mat = np.tensordot(
                            block_a.reduced, block_b.reduced, axes=(idxa, idxb))
                        if outg not in map_idx_out:
                            cgs = [np.tensordot(cga, cgb, axes=(idxa, idxb))
                                   for cga, cgb in zip(block_a.cgs, block_b.cgs)] \
                                   if block_a.cgs is not None and block_b.cgs is not None else None
                            map_idx_out[outg] = SubTensor(
                                q_labels=outg, reduced=mat, cgs=cgs)
                        else:
                            map_idx_out[outg].reduced += mat
        # non-abelian case (operator blocking case)
        # can only contract one index at a time
        # a rank-3 operator contracted rank-3 operator, in operator q-number index
        # will generate a rank-5 operator, the additional index is for new operator q-number index, in the middle
        else:
            assert len(idxa) == 1 and len(idxb) == 1
            if not isinstance(target_q_labels, list):
                target_q_labels = [target_q_labels]
            map_idx_out = {}
            for target in target_q_labels:
                for block_a in tsa.blocks:
                    a_rank = tuple(block_a.q_labels[id] for id in idxa)[0]
                    b_ranks = target + (-a_rank)
                    for b_rank in (b_ranks if isinstance(b_ranks, list) else [b_ranks]):
                        if (b_rank, ) in map_idx_b:
                            syms = [ir.__class__ for ir in target.irs]
                            target_cgs = [syms[ig].clebsch_gordan(a_rank.irs[ig], b_rank.irs[ig], target.irs[ig])
                                          for ig in range(target.ng)]
                            a_out_ranks = tuple(
                                block_a.q_labels[id] for id in out_idx_a)
                            for block_b in map_idx_b[(b_rank, )]:
                                outg = a_out_ranks + \
                                    (target, ) + \
                                    tuple(block_b.q_labels[id]
                                          for id in out_idx_b)
                                mat = np.tensordot(
                                    block_a.reduced, block_b.reduced, axes=(idxa, idxb))
                                mat = mat.reshape(
                                    mat.shape[:len(a_out_ranks)] + (1, ) + mat.shape[len(a_out_ranks):])
                                if outg not in map_idx_out:
                                    cgs = [np.tensordot(cga, np.tensordot(cgt, cgb, axes=([1], idxb)), axes=(idxa, [0]))
                                           for cga, cgb, cgt in zip(block_a.cgs, block_b.cgs, target_cgs)]
                                    map_idx_out[outg] = SubTensor(
                                        q_labels=outg, reduced=mat, cgs=cgs)
                                else:
                                    map_idx_out[outg].reduced += mat
        return Tensor(map_idx_out.values())
    
    def svd(self, k=-1):
        """
        Singular Value Decomposition of rank-2 block-diagonal Tensor.
        
        Args:
            k : int
                Maximal bond length. Defaults to -1 (no truncation).
        
        Returns:
            left_tensor : Tensor
                Left tensor.
            right_tensor : Tensor
                Right tensor.
            svd_s : list(numpy.ndarray)
                A list including singular values for each tensor block.
            error : float
                Sum of Discarded weights.
        """
        assert self.rank == 2
        blocks_l = []
        blocks_r = []
        svd_s = []
        total_k = 0
        for block in self.blocks:
            assert block.q_labels[0] == block.q_labels[1]
            u, s, vh = np.linalg.svd(block.reduced, full_matrices=False)
            blocks_l.append(SubTensor(block.q_labels, u))
            blocks_r.append(SubTensor(block.q_labels, vh))
            svd_s.append(s)
            total_k += len(s)
        
        if k == -1 or total_k <= k:
            # no truncation
            return Tensor(blocks_l), Tensor(blocks_r), svd_s, 0.0
        else:
            error = 0.0
            ss = [(i, j, v) for i, ps in enumerate(svd_s) for j, v in enumerate(ps)]
            assert len(ss) == total_k
            ss.sort(key=lambda x: -x[2])
            ss_trunc = ss[:k]
            ss_trunc.sort(key=lambda x: (x[0], x[1]))
            blocks_covered = [False] * len(self.blocks)
            for ik, g in groupby(ss_trunc, key=lambda x: x[0]):
                gl = [ig[1] for ig in g]
                gll = len(gl)
                assert gll == abs(gl[-1] - gl[0]) + 1
                glb, gle = sorted([gl[0], gl[-1]])
                blocks_covered[ik] = True
                if gll != len(svd_s[ik]):
                    blocks_l[ik].reduced = blocks_l[ik].reduced[:, glb:gle + 1]
                    blocks_l[ik].reduced_shape = blocks_l[ik].reduced.shape
                    blocks_r[ik].reduced = blocks_r[ik].reduced[glb:gle + 1, :]
                    blocks_r[ik].reduced_shape = blocks_r[ik].reduced.shape
                    svd_s[ik] = svd_s[ik][glb:gle + 1]
                    error += svd_s[ik][:glb].sum() + svd_s[ik][gle + 1:].sum()
            blocks_l_trunc = []
            blocks_r_trunc = []
            svd_s_trunc = []
            for ik, cov in enumerate(blocks_covered):
                if cov:
                    blocks_l_trunc.append(blocks_l[ik])
                    blocks_r_trunc.append(blocks_r[ik])
                    svd_s_trunc.append(svd_s[ik])
                else:
                    error += svd_s[ik].sum()
            return Tensor(blocks_l_trunc), Tensor(blocks_r_trunc), svd_s_trunc, error
    
    # split rank-2 block-diagonal Tensor to two tensors
    # using svd
    # k: maximal bond length; k == -1 -> no truncation
    # return left_tensor, right_tensor, error
    # if absorb_right is Ture, singlular values will be multiplied into right Tensor
    # otherwise, multiplied into right Tensor
    def split(self, absorb_right, k=-1):
        """
        Split rank-2 block-diagonal Tensor to two tensors (using SVD).
        
        Args:
            absorb_right : bool
                If absorb_right is True, singlular values will be multiplied into right Tensor.
                Otherwise, They will be multiplied into left Tensor.
            k : int
                Maximal bond length. Defaults to -1 (no truncation).
        
        Returns:
            left_tensor : Tensor
                Left tensor.
            right_tensor : Tensor
                Right tensor.
            error : float
                Sum of Discarded weights.
        """
        ts_l, ts_r, svd_s, error = self.svd(k=k)
        assert ts_l.n_blocks == ts_r.n_blocks and ts_l.n_blocks == len(svd_s)
        for l, r, s in zip(ts_l.blocks, ts_r.blocks, svd_s):
            if absorb_right:
                # np.diag(g).dot(x) == g[:, None] * x (faster)
                r.reduced = s[:, None] * r.reduced
            else:
                # x.dot(np.diag(g)) == g[None, :] * x (faster)
                l.reduced = s[None, :] * l.reduced
        return ts_l, ts_r, error
    
    # left normalization needs to collect all left indices for each specific right index
    # so that we will only have one R, but left dim of q is unchanged
    # at: where to divide the tensor into matrix => (0, at) x (at, n_ranks)
    def left_normalize(self):
        """
        Left normalization (using QR factorization).
        
        Returns:
            r_blocks : dict((DirectProdGroup, ) -> numpy.ndarray)
                The R matrix for each right-index quantum label.
        """
        at = self.rank - 1
        collected_rows = {}
        for block in self.blocks:
            q_labels_r = tuple(block.q_labels[id]
                               for id in range(at, self.rank))
            if q_labels_r not in collected_rows:
                collected_rows[q_labels_r] = []
            collected_rows[q_labels_r].append(block)
        r_blocks = {}
        for q_labels_r, blocks in collected_rows.items():
            l_shapes = [np.prod([b.reduced.shape[id]
                                 for id in range(at)]) for b in blocks]
            mat = np.concatenate([b.reduced.reshape((sh, -1))
                                  for sh, b in zip(l_shapes, blocks)], axis=0)
            q, r = np.linalg.qr(mat)
            r_blocks[q_labels_r] = r
            qs = np.split(q, list(accumulate(l_shapes[:-1])), axis=0)
            assert(len(qs) == len(blocks))
            for q, b in zip(qs, blocks):
                b.reduced = q.reshape(b.reduced.shape[:at] + (r.shape[0], ))
                b.reduced_shape = b.reduced.shape
        return r_blocks

    def right_normalize(self):
        """
        Right normalization (using LQ factorization).
        
        Returns:
            l_blocks : dict((DirectProdGroup, ) -> numpy.ndarray)
                The L matrix for each left-index quantum label.
        """
        at = 1
        collected_cols = {}
        for block in self.blocks:
            q_labels_l = tuple(block.q_labels[id]
                               for id in range(0, at))
            if q_labels_l not in collected_cols:
                collected_cols[q_labels_l] = []
            collected_cols[q_labels_l].append(block)
        l_blocks = {}
        for q_labels_l, blocks in collected_cols.items():
            r_shapes = [np.prod([b.reduced.shape[id]
                                 for id in range(at, self.rank)]) for b in blocks]
            mat = np.concatenate([b.reduced.reshape((-1, sh)).T
                                  for sh, b in zip(r_shapes, blocks)], axis=0)
            q, r = np.linalg.qr(mat)
            l_blocks[q_labels_l] = r.T
            qs = np.split(q, list(accumulate(r_shapes[:-1])), axis=0)
            assert(len(qs) == len(blocks))
            for q, b in zip(qs, blocks):
                b.reduced = q.T.reshape((r.shape[0], ) + b.reduced.shape[at:])
                b.reduced_shape = b.reduced.shape
        return l_blocks
    
    def left_multiply(self, mats):
        """
        Left Multiplication.
        Currently only used for multiplying R obtained from Left-normalization.
        
        Args:
            mats : dict((DirectProdGroup, ) -> numpy.ndarray)
                The R matrix for each right-index quantum label.
        """
        for block in self.blocks:
            q_labels_r = (block.q_labels[0], )
            if q_labels_r in mats:
                block.reduced = np.tensordot(
                    mats[q_labels_r], block.reduced, axes=([1], [0]))
                block.reduced_shape = block.reduced.shape

    def set_tags(self, tags):
        """Change the tags, return ``self`` for chain notation."""
        self.tags = tags
        return self

    def set_contractor(self, contractor):
        """Change the contractor, return ``self`` for chain notation."""
        self.contractor = contractor
        return self

    def __add__(self, o):
        assert self.rank == o.rank and self.ng == o.ng
        map_s = {tuple(b.q_labels): b for b in self.blocks}
        map_o = {tuple(b.q_labels): b for b in o.blocks}
        blocks = []
        for q, b in map_s:
            if q in map_o:
                blocks.append(
                    SubTensor(q, b.reduced + map_o[q].reduced, b.cgs))
            else:
                blocks.append(b)
        return Tensor(blocks=blocks, tags=self.tags, contractor=self.contractor)
    
    def __eq__(self, o):
        lb = sorted(self.blocks, key=lambda x: x.q_labels)
        rb = sorted(o.blocks, key=lambda x: x.q_labels)
        if len(lb) != len(rb):
            return False
        for l, r in zip(lb, rb):
            if l != r:
                return False
        return True
    
    def sort(self):
        """Sort non-zero blocks in increasing order of quantum labels."""
        self.blocks = sorted(self.blocks, key=lambda x: x.q_labels)
    
    def __mul__(self, o):
        return Tensor(blocks=[b * o for b in self.blocks], tags=self.tags.copy(),
                      contractor=self.contractor)

    def __repr__(self):
        return "\n".join(
            ("%3d " % ib) + b.__repr__() for ib, b in enumerate(self.blocks))


class TensorNetworkError(Exception):
    pass


class TensorNetwork:
    """
    An inefficient implementation for Quimb TensorNetwork.
    
    Attributes:
        tensors : list(Tensor)
            List of tensors in the network.
    """
    def __init__(self, tensors=None):
        self.tensors = list(tensors) if tensors is not None else []

    def select(self, tags, which='all', inverse=False):
        """
        Extract a sub tensor network specified by tags.
        
        Args:
            tags : set(str or int)
                Tags of sub tensor network.
        
        Kwargs:
            which : 'all' or 'any' or 'exact'
                Defaults to 'all'. Indicating how the ``tags`` should be applied for selection.
            inverse : bool
                Defaults to False. Indicating whether the selection should be inversed.
        
        Returns:
            tn : TensorNetwork
                The selected tensor network.
        """
        r = []
        if not isinstance(tags, set):
            tags = {tags}
        for tensor in self.tensors:
            if which == 'all':
                p = tags.issubset(tensor.tags)
            elif which == 'any':
                p = len(tags.intersection(tensor.tags)) != 0
            elif which == 'exact':
                p = tags == tensor.tags
            else:
                raise TensorNetworkError('invalid which parameter.')
            if inverse ^ p:
                r.append(tensor)
        return self.__class__(tensors=r)

    def remove(self, tags, which='all', in_place=False):
        """
        Remove a sub tensor network specified by tags.
        
        Args:
            tags : set(str or int)
                Tags of sub tensor network.
        
        Kwargs:
            which : 'all' or 'any' or 'exact'
                Defaults to 'all'. Indicating how the ``tags`` should be applied for selection.
            in_place : bool
                Defaults to False. Indicating whether the current tensor network should be changed.
        
        Returns:
            tn : TensorNetwork
                The remaining tensor network.
        """
        if not in_place:
            return self.select(tags, which, inverse=True)
        else:
            self.tensors = self.select(tags, which, inverse=True).tensors
            return self

    def add(self, tn):
        """Add a Tensor or TensorNetwork to the tensor network."""
        if isinstance(tn, Tensor):
            self.tensors.append(tn)
        elif isinstance(tn, TensorNetwork):
            for tensor in tn.tensors:
                self.tensors.append(tensor)
        else:
            raise TensorNetworkError(
                'Unable to add this object to the network.')

    def add_tags(self, tags):
        """Add tags to all tensors in the tensor network."""
        for tensor in self.tensors:
            tensor.tags |= tags

    def remove_tags(self, tags):
        """Remove tags from all tensors in the tensor network."""
        for tensor in self.tensors:
            tensor.tags -= tags
    
    def copy(self):
        """Shallow copy."""
        return self.__class__(tensors=[t.copy() for t in self.tensors])
    
    def deep_copy(self):
        """Deep copy."""
        return self.__class__(tensors=[t.deep_copy() for t in self.tensors])

    def contract(self, tags, in_place=False):
        """
        Contract a sub tensor network specified by tags.
        If any Tensor in the sub tensor network has a ``contractor``,
        the specialized contraction will be used.
        Currently the general contraction is not implemented.
        
        Args:
            tags : tuple(str or int..)
                Tags of sub tensor network.
        
        Kwargs:
            in_place : bool
                Defaults to False. Indicating whether the current tensor network should be changed.
        
        Returns:
            tn : TensorNetwork
                The tensor network with selected sub tensor network replaced by its contraction.
        """
        if isinstance(tags, str) or isinstance(tags, int):
            tags = (tags, )
        cont_tn = self.select(set(tags), which='any')
        ctr = None
        for t in cont_tn.tensors:
            if t.contractor is not None:
                ctr = t.contractor
                break
        if ctr is None:
            raise TensorNetworkError('Regular contraction not implemented.')
        else:
            cont_ts = ctr.contract(cont_tn, tags)
            if in_place:
                self.remove(set(tags), which='any', in_place=True)
                self |= cont_ts
                return self
            else:
                return self.remove(set(tags), which='any') | cont_ts

    def __xor__(self, tags):
        return self.contract(tags)

    def __ixor__(self, tags):
        return self.contract(tags, in_place=True)

    def __ior__(self, tensors):
        self.add(tensors)
        return self

    def __len__(self):
        return len(self.tensors)

    # this may be different from quimb impl
    def __getitem__(self, tags):
        t = self.select(tags, which='exact')
        if len(t) == 0:
            raise TensorNetworkError(
                'Unable to find an item with the given tags.')
        return t if len(t) > 1 else t.tensors[0]

    def __or__(self, other):
        if isinstance(other, Tensor):
            return self.__class__(tensors=self.tensors + [other])
        elif isinstance(other, TensorNetwork):
            if self.__class__ == other.__class__:
                return self.__class__(tensors=self.tensors + other.tensors)
            else:
                return TensorNetwork(tensors=self.tensors + other.tensors)
        else:
            raise TensorNetworkError(
                'Unable to create the network using this object.')

    @property
    def tags(self):
        """Return a list of tags collected from all tensors in the tensor network."""
        return [ts.tags for ts in self.tensors]
