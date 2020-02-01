
from block import VectorInt, VectorVectorInt, VectorMatrix, Matrix
from block import save_rotation_matrix, DiagonalMatrix
from block.io import Global, read_input, Input, AlgorithmTypes
from block.io import init_stack_memory, release_stack_memory, AlgorithmTypes
from block.dmrg import MPS_init, MPS, get_dot_with_sys
from block.symmetry import VectorStateInfo, get_commute_parity
from block.symmetry import state_tensor_product, SpinQuantum, VectorSpinQuantum
from block.symmetry import StateInfo, SpinSpace, IrrepSpace, state_tensor_product_target
from block.operator import Wavefunction, OpTypes, StackSparseMatrix
from block.block import Block, StorageTypes
from block.block import init_starting_block, init_new_system_block
from block.rev import tensor_scale, tensor_trace, tensor_rotate, tensor_product
from block.rev import tensor_scale_add, tensor_scale_add_no_trans, tensor_precondition
from block.rev import tensor_trace_diagonal, tensor_product_diagonal, tensor_dot_product
from block.rev import tensor_trace_multiply, tensor_product_multiply

from ..symmetry.symmetry import ParticleN, SU2, PointGroup, point_group
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

    def __init__(self, empty, n_sites, basis, target):
        self.empty = empty
        self.n_sites = n_sites
        self.basis = basis
        self.target = target
        self.left_block_basis = []
        self.right_block_basis = []
        self.left_state_info = None
        self.right_state_info = None
        self.left_state_info_no_trunc = None
        self.right_state_info_no_trunc = None
    
    @staticmethod
    def from_line_coupling(lcp):
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
        right, no_truc = self.get_left_state_info(i, left=left)
        self.left_state_info[i] = right
        self.left_state_info_no_trunc[i] = no_truc
        return right
    
    def update_local_right_state_info(self, i, right=None):
        left, no_truc = self.get_right_state_info(i, right=right)
        self.right_state_info[i] = left
        self.right_state_info_no_trunc[i] = no_truc
        return left
    
    # basis is a list of pairs: [(q_label, n_staes)]
    def update_local_block_basis(self, i, block_basis):
        
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
    
    # no cgs will be generated for twodot
    # expressed in 2-index tensor, can be used for svd
    def from_wavefunction_fused(self, i, wfn):
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
    
    def unfuse_left(self, i, tensor):
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
    # expressed in direct product 4-index tensor
    # may not be useful
    def from_wavefunction(self, i, wfn):
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

# a wrapper of StackWavefunction for Davidson algorithm
class BlockWavefunction:
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
        mat = self.data.__class__()
        mat.deep_copy(self.data)
        return BlockWavefunction(mat, self.factor)
    
    def clear_copy(self):
        mat = self.data.__class__()
        mat.deep_clear_copy(self.data)
        return BlockWavefunction(mat, 1.0)
    
    def copy_data(self, other):
        self.data.copy_data(other.data)
        self.factor = other.factor
    
    def dot(self, other):
        return tensor_dot_product(self.data, other.data) * self.factor * other.factor
    
    def precondition(self, ld, diag):
        tensor_precondition(self.data, ld, diag)
    
    def normalize(self):
        self.factor = 1.0
        tensor_scale(1 / np.sqrt(self.dot(self)), self.data)
    
    def deallocate(self):
        assert self.data is not None
        self.data.deallocate()
        self.data = None
    
    def __repr__(self):
        return repr(self.factor) + " * " + repr(self.data)

# a wrapper of MultiplyH for Davidson algorithm
class BlockMultiplyH:
    def __init__(self, mpo):
        self.mpo = mpo
        self.st = state_tensor_product_target(mpo.left_op_names, mpo.right_op_names)
        self.diag_mat = BlockEvaluation.expr_diagonal_eval(mpo.mat[0, 0], mpo.ops[0], mpo.ops[1], self.st)
    
    def diag(self):
        return self.diag_mat
    
    def apply(self, other, result):
        assert isinstance(result, BlockWavefunction)
        result.factor = 1.0
        BlockEvaluation.expr_multiply_eval(self.mpo.mat[0, 0], self.mpo.ops[0], self.mpo.ops[1],
            other.data, result.data, self.st)

class BlockEvaluation:
    @classmethod
    def tensor_rotate(self, mpo, old_st, new_st, rmat):
        new_ops = {}
        for k, v in mpo.ops.items():
            nmat = StackSparseMatrix()
            nmat.delta_quantum = v.delta_quantum
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
        old_st = info.left_state_info_no_trunc[i]
        new_st = info.left_state_info[i]
        rmat = info.get_left_rotation_matrix(i, mps)
        return self.tensor_rotate(mpo, old_st, new_st, rmat)
    
    @classmethod
    def right_rotate(self, mpo, mps, info, i):
        old_st = info.right_state_info_no_trunc[i]
        new_st = info.right_state_info[i]
        rmat = info.get_right_rotation_matrix(i, mps)
        return self.tensor_rotate(mpo, old_st, new_st, rmat)
    
    @classmethod
    def left_contract(self, mpol, mpo, info, i):
        new_mat = mpol.mat @ mpo.mat
        st = info.left_state_info_no_trunc[i]
        new_ops = {}
        for j in range(new_mat.shape[1]):
            if mpo.right_op_names[j].sign == -1:
                new_ops[-mpo.right_op_names[j]] = self.expr_eval(-new_mat[0, j], mpol.ops, mpo.ops, st)
            else:
                new_ops[mpo.right_op_names[j]] = self.expr_eval(new_mat[0, j], mpol.ops, mpo.ops, st)
        return mpo.__class__(mat=mpo.right_op_names.reshape(new_mat.shape),
                             ops=new_ops, tags=mpo.tags,
                             lop=mpol.left_op_names,
                             rop=mpo.right_op_names,
                             contractor=mpo.contractor)
    
    @classmethod
    def right_contract(self, mpor, mpo, info, i):
        new_mat = mpo.mat @ mpor.mat
        st = info.right_state_info_no_trunc[i]
        new_ops = {}
        for j in range(new_mat.shape[0]):
            if mpo.left_op_names[j].sign == -1:
                new_ops[-mpo.left_op_names[j]] = self.expr_eval(-new_mat[j, 0], mpo.ops, mpor.ops, st)
            else:
                new_ops[mpo.left_op_names[j]] = self.expr_eval(new_mat[j, 0], mpo.ops, mpor.ops, st)
        return mpo.__class__(mat=mpo.left_op_names.reshape(new_mat.shape),
                             ops=new_ops, tags=mpo.tags,
                             lop=mpo.left_op_names,
                             rop=mpor.right_op_names,
                             contractor=mpo.contractor)
    
    @classmethod
    def left_right_contract(self, mpol, mpor, info, i):
        new_mat = mpol.mat @ mpor.mat
        st_l = info.left_state_info_no_trunc[i]
        st_r = info.right_state_info_no_trunc[i + 1]
        assert new_mat.shape == (1, 1)
        return mpol.__class__(mat=new_mat, ops=(mpol.ops, mpor.ops),
                              tags={'_HAM'},
                              lop=st_l, rop=st_r, contractor=mpol.contractor)
    
    @classmethod
    def expr_diagonal_eval(self, expr, a, b, st):
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
    def expr_eval(self, expr, a, b, st):
        state_info = VectorStateInfo([st.left_state_info, st.right_state_info, st])
        if isinstance(expr, OpString):
            assert len(expr.ops) == 2
            nmat = StackSparseMatrix()
            if expr.ops[0] == OpElement(OpNames.I, ()):
                nmat.delta_quantum = b[expr.ops[1]].delta_quantum
                nmat.allocate(st)
                nmat.initialized = True
                assert b[expr.ops[1]].rows == b[expr.ops[1]].cols
                assert b[expr.ops[1]].rows == len(state_info[1].quanta)
                assert nmat.rows == nmat.cols
                assert nmat.rows == len(state_info[2].quanta)
                tensor_trace(b[expr.ops[1]], nmat, state_info, False, float(expr.sign))
            elif expr.ops[1] == OpElement(OpNames.I, ()):
                nmat.delta_quantum = a[expr.ops[0]].delta_quantum
                nmat.allocate(st)
                nmat.initialized = True
                assert a[expr.ops[0]].rows == a[expr.ops[0]].cols
                assert a[expr.ops[0]].rows == len(state_info[0].quanta)
                assert nmat.rows == nmat.cols
                assert nmat.rows == len(state_info[2].quanta)
                tensor_trace(a[expr.ops[0]], nmat, state_info, True, float(expr.sign))
            else:
                # TODO here we did not consider S != 0 products
                aq, bq = a[expr.ops[0]].delta_quantum[0], b[expr.ops[1]].delta_quantum[0]
                nmat.delta_quantum = VectorSpinQuantum([(aq + bq)[0]])
                if expr.sign == -1:
                    scale = get_commute_parity(aq, bq, nmat.delta_quantum[0])
                else:
                    scale = 1.0
                nmat.allocate(st)
                nmat.initialized = True
                tensor_product(a[expr.ops[0]], b[expr.ops[1]], nmat, state_info, scale)
            return nmat
        elif isinstance(expr, OpSum):
            nmat = self.expr_eval(expr.strings[0], a, b, st)
            assert nmat.conjugacy == 'n'
            for x in expr.strings[1:]:
                t = self.expr_eval(x, a, b, st)
                tensor_scale_add(1.0, t, nmat, st)
                t.deallocate()
            return nmat
        else:
            assert False
    
    @classmethod
    def eigen_values(self, mpo):
        assert mpo.mat.shape == (1, 1)
        mat = mpo.ops[mpo.mat[0, 0]]
        evs = []
        for k, v in mat.non_zero_blocks:
            p, pp = np.linalg.eigh(v.ref)
            ppp = sorted(zip(p, pp.T), key=lambda x : x[0])
            evs.append(ppp[0][0])
        evs.sort()
        return np.array(evs)

class BlockSymmetry:
    # translate a DirectProdGroup object to SpinQuantum (block code)
    @classmethod
    def to_spin_quantum(self, dpg):
        subg = [ParticleN, SU2, PointGroup]
        if dpg.ng != 3 or not all(isinstance(ir, c) for c, ir in zip(subg, dpg.irs)):
            raise BlockError('Representation not supported by block code.')
        return SpinQuantum(dpg.irs[0].ir, SpinSpace(dpg.irs[1].ir), IrrepSpace(dpg.irs[2].ir))

    # translate SpinQuantum (block code) to a DirectProdGroup object
    @classmethod
    def from_spin_quantum(self, sq):
        PG = point_group(Global.point_group)
        return ParticleN(sq.n) * SU2(sq.s.irrep) * PG(sq.symm.irrep)

    # translate a [(DirectProdGroup, int)] to StateInfo (block code)
    @classmethod
    def to_state_info(self, states):
        qs = VectorSpinQuantum()
        ns = VectorInt()
        for k, v in states:
            qs.append(self.to_spin_quantum(k))
            ns.append(v)
        return StateInfo(qs, ns)

    @classmethod
    def initial_state_info(self, i=0):
        return MPS.site_blocks[i].ket_state_info

    # Translate the last one or two mps tensors to Wavefunction (block code)
    # in two_dot scheme, wavefunction is repr'd in 2M(n-3 site) x M(n-1 site)
    # in one_dot scheme, wavefunction is repr'd in  M(n-2 site) x M(n-1 site)
    # in two_dot, left_state_info is gen'd from :func:`to_rotation_matrix`
    # with `i = n - 3`
    # in one_dot, left_state_info is gen'd with `i = n - 2`
    @classmethod
    def to_wavefunction(self, one_dot, left_state_info, n_sites, mps, target):

        # one dot case
        if one_dot:

            map_last = {
                block.q_labels[:2]: block for block in mps[n_sites - 1].blocks}

            l = left_state_info
            r = MPS.site_blocks[n_sites - 1].ket_state_info
            big = state_tensor_product_target(l, r)

            target_state_info = self.to_state_info([(target, 1)])
            wfn = Wavefunction()
            wfn.initialize(target_state_info.quanta, l, r, True)
            wfn.onedot = True

            for (ib, ik), mat in wfn.non_zero_blocks:
                sqs = [l.quanta[ib], r.quanta[ik]]
                q_labels = tuple(self.from_spin_quantum(sq) for sq in sqs)
                if q_labels in map_last:
                    mat.ref[:, :] = map_last[q_labels].reduced.reshape(
                        mat.ref.shape)
                else:
                    mat.ref[:, :] = 0

            big.collect_quanta()
            return wfn, big

        # two dot case
        else:

            last_two = Tensor.contract(
                mps[n_sites - 2], mps[n_sites - 1], [2], [0])
            map_last = {block.q_labels[:3]: block for block in last_two.blocks}

            l = left_state_info
            r = MPS.site_blocks[n_sites - 2].ket_state_info
            rr = MPS.site_blocks[n_sites - 1].ket_state_info
            ll = state_tensor_product(l, r)
            big = state_tensor_product_target(ll, rr)
            ll_idl = ll.left_unmap_quanta
            ll_idr = ll.right_unmap_quanta

            ll_collected = ll.copy()
            ll_collected.collect_quanta()
            otn = ll_collected.old_to_new_state

            qq = self.to_spin_quantum(target)

            target_state_info = self.to_state_info([(target, 1)])

            wfn = Wavefunction()
            wfn.initialize(target_state_info.quanta, ll_collected, rr, False)
            wfn.onedot = False
            wfn.clear()

            for (ibc, ik), mat in wfn.non_zero_blocks:
                mats = []
                for ib in otn[ibc]:
                    sqs = [l.quanta[ll_idl[ib]],
                           r.quanta[ll_idr[ib]], rr.quanta[ik]]
                    q_labels = tuple(self.from_spin_quantum(sq) for sq in sqs)
                    if q_labels in map_last:
                        rd_shape = map_last[q_labels].reduced.shape
                        shape = (rd_shape[0] * rd_shape[1], rd_shape[2])
                        mats.append(map_last[q_labels].reduced.reshape(shape))
                if mats != []:
                    all_mat = np.concatenate(mats, axis=0)
                    assert(all_mat.shape == mat.ref.shape)
                    mat.ref[:, :] = all_mat
                else:
                    mat.ref[:, :] = 0

            big.collect_quanta()
            big.left_state_info = ll_collected
            big.right_state_info.left_state_info = rr
            # TODO: this should be empty state
            big.right_state_info.right_state_info = self.to_state_info([
                                                                       (target, 1)])
            return wfn, big

    # Translate a site in MPS to rotation matrix (block code) for that site
    # left_state_info: state_info to the left of site i
    # i: site index [left_state_info] \otimes [site i]
    # tensor: mps tensor at site i
    # the first two sites only contributes one rotation matrix
    # this is handled by i = 1 and tensor0 = mps[0]
    @classmethod
    def to_rotation_matrix(self, left_state_info, tensor, i, tensor0=None):
        l = left_state_info
        r = MPS.site_blocks[i].ket_state_info
        lr = state_tensor_product(l, r)
        lr_idl = lr.left_unmap_quanta
        lr_idr = lr.right_unmap_quanta

        lr_collected = lr.copy()
        lr_collected.collect_quanta()

        map_tensor = {block.q_labels: block for block in tensor.blocks}
        if tensor0 is not None:
            map_tensor0 = {block.q_labels[2]: block for block in tensor0.blocks}

        # if there are repeated q in lr.quanta,
        # current rot Matrix should be None and
        # the reduced matrix should be concatenated to that of the first unique q
        rot = []
        collected = {}
        for i, q in enumerate(lr.quanta):
            sqs = [l.quanta[lr_idl[i]], r.quanta[lr_idr[i]], q]
            q_labels = tuple(self.from_spin_quantum(sq) for sq in sqs)
            if q_labels in map_tensor:
                block = map_tensor[q_labels]
                reduced = block.reduced.reshape(block.reduced.shape[0::2])
                if tensor0 is not None:
                    block0 = map_tensor0[q_labels[0]]
                    reduced = block0.reduced.reshape(
                        block0.reduced.shape[0::2]) @ reduced
                if q_labels[2] not in collected:
                    rot.append([reduced])
                    collected[q_labels[2]] = len(rot) - 1
                else:
                    rot[collected[q_labels[2]]].append(reduced)
                    rot.append(())
            else:
                if q_labels[2] not in collected:
                    rot.append(None)
                    collected[q_labels[2]] = len(rot) - 1
                else:
                    rot.append(())

        rot_uncollected = VectorMatrix(Matrix() if r is None or r is () else Matrix(
            np.concatenate(r, axis=0)) for r in rot)

        # order the quanta according to the order in lr_collected
        rot_collected = []
        for q in lr_collected.quanta:
            m = rot[collected[self.from_spin_quantum(q)]]
            rot_collected.append(
                Matrix() if m is None else Matrix(np.concatenate(m, axis=0)))

        rot_collected = VectorMatrix(rot_collected)

        lr_truncated = StateInfo()
        StateInfo.transform_state(rot_uncollected, lr, lr_truncated)
        lr_truncated.left_state_info = lr_collected.left_state_info
        lr_truncated.right_state_info = lr_collected.right_state_info

        return rot_collected, lr_truncated


class BlockHamiltonian:
    memory_initialzed = False

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

        if not self.__class__.memory_initialzed:
            init_stack_memory()
            self.__class__.memory_initialzed = True

        MPS_init(True)
    
    def get_site_operators(self, i):
        ops = {}
        block = Block(i, i, 0, False)
        
        mat = StackSparseMatrix()
        mat.deep_copy(block.ops[OpTypes.Hamiltonian].local_element_linear(0)[0])
        ops[OpElement(OpNames.H, ())] = mat
        
        mat = StackSparseMatrix()
        mat.deep_copy(block.ops[OpTypes.Overlap].local_element_linear(0)[0])
        ops[OpElement(OpNames.I, ())] = mat
        
        mat = StackSparseMatrix()
        mat.deep_copy(block.ops[OpTypes.Cre].local_element_linear(0)[0])
        ops[OpElement(OpNames.C, ())] = mat
        
        mat = StackSparseMatrix()
        mat.deep_copy(block.ops[OpTypes.Des].local_element_linear(0)[0])
        ops[OpElement(OpNames.D, ())] = mat
        
        for j in range(0, i):
            
            mat = StackSparseMatrix()
            mat.deep_copy(block.ops[OpTypes.Des].local_element_linear(0)[0])
            tensor_scale(self.t[j, i] * np.sqrt(2), mat)
            ops[OpElement(OpNames.S, (j, ))] = mat
            
            mat = StackSparseMatrix()
            mat.deep_copy(block.ops[OpTypes.Cre].local_element_linear(0)[0])
            tensor_scale(self.t[j, i] * np.sqrt(2), mat)
            ops[OpElement(OpNames.SD, (j, ))] = mat
        
        # TODO :: need to store Block to deallocate it later
        return ops
    
    @staticmethod
    def block_operator_summary(block):
        r = []
        for k, v in block.ops.items():
            r.append(repr(k) + " :: local_elements")
            g = []
            for vv in v.local_indices:
                g.append("%r -> %d" % (tuple(vv), len(v.local_element(*vv))))
            r.append(', '.join(g))
        return "\n".join(r)

    def make_starting_block(self, forward):
        system = Block()
        init_starting_block(system, forward, -1, -1, 1, 1, 0, False, False, 0)
        system.store(forward, system.sites, -1, -1)
        return system

    def make_big_block(self, system):

        forward = system.sites[0] == 0

        dot_with_sys = get_dot_with_sys(system, self.dot == 1, forward)

        if forward:
            sys_dot_site = system.sites[-1] + 1
            env_dot_site = sys_dot_end + 1
        else:
            sys_dot_site = system.sites[0] - 1
            env_dot_site = sys_dot_end - 1
        
        system_dot = Block(sys_dot_site, sys_dot_site, 0, True)
        environment_dot = Block(env_dot_site, env_dot_site, 0, True)

        sys_have_norm_ops = dot_with_sys
        sys_have_comp_ops = not dot_with_sys

        env_have_norm_ops = not sys_have_norm_ops
        env_have_comp_ops = not sys_have_comp_ops

        if sys_have_comp_ops and OpTypes.CreDesComp not in system.ops:
            system.add_all_comp_ops()

        system.add_additional_ops()

        new_system = Block()

        if not self.dot == 1 or dot_with_sys:
            init_new_system_block(system, system_dot, new_system, -1, -1, 1, True, 0,
                StorageTypes.DistributedStorage, sys_have_norm_ops, sys_have_comp_ops)
        
        environment = Block()
        new_environment = Block()

        init_new_environment_block(
            environment,
            system_dot if self.dot == 1 and not dot_with_sys else environment_dot,
            new_environment, system, system_dot,
            -1, -1, 1, 1, forward, True, self.dot == 1, False, 0,
            env_have_norm_ops, env_have_comp_ops, dot_with_sys)
        
        new_system.loop_block = dot_with_sys
        system.loop_block = dot_with_sys
        new_environment.loop_block = not dot_with_sys
        environment.loop_block = not dot_with_sys

        if self.dot == 1 and not dot_with_sys:
            left_block = system
            right_block = new_environment
        else:
            left_block = new_system
            right_block = new_environment
        
        big = Block()

        init_big_block(left_block, right_block, big)

        return system, system_dot, new_system, environment, big

    def block_rotation(self, new_system, system, rot_mat):
        save_rotation_matrix(new_system.sites, rot_mat, 0)
        save_rotation_matrix(new_system.sites, rot_mat, -1)

        new_system.transform_operators(rot_mat)
        new_system.move_and_free_memory(system)

        return new_system

    def enlarge_block(self, forward, system, rot_mat):
        dot_with_sys = get_dot_with_sys(system, self.dot == 1, forward)
        if forward:
            dot_site = system.sites[-1] + 1
        else:
            dot_site = system.sites[0] - 1
        print(forward, system.sites, dot_site)
        system_dot = Block(dot_site, dot_site, 0, True)

        do_norms = dot_with_sys
        do_comp = not dot_with_sys
        if do_comp and OpTypes.CreDesComp not in system.ops:
            system.add_all_comp_ops()
        system.add_additional_ops()

        new_system = Block()
        init_new_system_block(system, system_dot, new_system, -1, -1, 1, True,
                              0, StorageTypes.DistributedStorage, do_norms, do_comp)

        return self.block_rotation(new_system, system, rot_mat)


if __name__ == '__main__':
    # BlockHamiltonian.read_fcidump('N2.STO3G.FCIDUMP')
    pass
