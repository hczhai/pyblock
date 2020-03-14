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

from block import VectorInt, VectorVectorInt, DiagonalMatrix
from block import VectorVectorMatrix
from block.io import Global, read_input, Input, AlgorithmTypes
from block.io import init_stack_memory, release_stack_memory, AlgorithmTypes
from block.io import get_current_stack_memory, set_current_stack_memory
from block.dmrg import MPS_init, MPS
from block.symmetry import VectorStateInfo
from block.symmetry import SpinQuantum, VectorSpinQuantum
from block.symmetry import StateInfo, SpinSpace, IrrepSpace
from block.operator import Wavefunction, OpTypes, StackSparseMatrix
from block.block import Block, VectorBlock
from block.rev import tensor_scale, tensor_trace, tensor_rotate, tensor_product
from block.rev import tensor_trace_diagonal, tensor_product_diagonal
from block.rev import tensor_trace_multiply, tensor_product_multiply, product
from block.rev import tensor_scale_add_no_trans, tensor_dot_product

from ..symmetry.symmetry import ParticleN, SU2, SZ, PointGroup, point_group
from ..symmetry.symmetry import DirectProdGroup
from .operator import OpElement, OpNames, OpString, OpSum
from .mpo import OperatorTensor, DualOperatorTensor
from .simplifier import NoSimplifier, OpCollection, OpShell
from .fcidump import read_fcidump
from fractions import Fraction
import contextlib
import numpy as np
import time
import copy
                
class BlockError(Exception):
    pass


class BlockEvaluation:
    simplifier = NoSimplifier()
    parallelizer = None
    """Explicit evaluation of symbolic expression for operators."""
    @classmethod
    def tensor_rotate(self, opt, sts, rmats):
        """
        Transform basis of MPO using rotation matrix.
        
        Args:
            opt : OperatorTensor or DualOperatorTensor
                Operator tensor in (untruncated) old basis.
            sts : VectorStateInfo
                Old (untruncated) and new (truncated) basis.
            rmats : VectorVectorMatrix
                Rotation matrices for ket (or bra and ket).
        
        Returns:
            new_opt : OperatorTensor or DualOperatorTensor
                Operator tensor in (truncated) new basis.
        """
        exprs = self.simplifier.simplify(opt.ops.items())
        if self.parallelizer is not None:
            exprs = self.parallelizer.parallelize(exprs)
        with exprs() as (zipped, new_ops):
            for op, mat in zipped:
                if mat == 0:
                    new_ops[op] = 0
                else:
                    nmat = StackSparseMatrix()
                    if isinstance(mat, OpShell):
                        nmat.delta_quantum = mat.data.delta_quantum
                        nmat.fermion = mat.data.fermion
                    else:
                        nmat.delta_quantum = mat.delta_quantum
                        nmat.fermion = mat.fermion
                    nmat.allocate(VectorStateInfo(sts[1::2]))
                    nmat.initialized = True
                    if not isinstance(mat, OpShell):
                        tensor_rotate(mat, nmat, sts, rmats, mat.symm_scale)
                    new_ops[op] = nmat
        new_opt = copy.copy(opt)
        new_opt.ops = new_ops
        return new_opt
    
    @classmethod
    def left_rotate(self, i, opt, mpst, mps_info, bra_mpst=None, bra_mps_info=None):
        """Perform rotation <MPS|MPO|MPS> for left block.
        
        Args:
            i : int
                Site index.
            opt : OperatorTensor or DualOperatorTensor
                Operator tensor in (untruncated) old basis.
            mpst : Tensor
                MPS tensor defining rotation in ket side.
            mps_info : MPSInfo
                MPSInfo object for ket state.
            bra_mpst : Tensor or None (if same as mpst)
                MPS tensor defining rotation in bra side.
            bra_mps_info : MPSInfo
                MPSInfo object for bra state.
        
        Returns:
            new_opt : OperatorTensor or DualOperatorTensor
                Operator tensor in (truncated) new basis.
        """
        if mps_info is None:
            return opt
        old_st = mps_info.left_state_info_no_trunc[i]
        new_st = mps_info.left_state_info[i]
        rmat = mps_info.get_left_rotation_matrix(i, mpst)
        if bra_mps_info is not None:
            old_bra_st = bra_mps_info.left_state_info_no_trunc[i]
            new_bra_st = bra_mps_info.left_state_info[i]
            bra_rmat = bra_mps_info.get_left_rotation_matrix(i, bra_mpst)
            sts = VectorStateInfo([old_bra_st, new_bra_st, old_st, new_st])
            rmats = VectorVectorMatrix([bra_rmat, rmat])
        else:
            sts = VectorStateInfo([old_st, new_st])
            rmats = VectorVectorMatrix([rmat])
        return self.tensor_rotate(opt, sts, rmats)
    
    @classmethod
    def right_rotate(self, i, opt, mpst, mps_info, bra_mpst=None, bra_mps_info=None):
        """Perform rotation <MPS|MPO|MPS> for right block.
        
        Args:
            i : int
                Site index.
            opt : OperatorTensor or DualOperatorTensor
                Operator tensor in (untruncated) old basis.
            mpst : Tensor
                MPS tensor defining rotation in ket side.
            mps_info : MPSInfo
                MPSInfo object for ket state.
            bra_mpst : Tensor or None (if same as mpst)
                MPS tensor defining rotation in bra side.
            bra_mps_info : MPSInfo or None (if same as mps_info)
                MPSInfo object for bra state.
        
        Returns:
            new_opt : OperatorTensor or DualOperatorTensor
                Operator tensor in (truncated) new basis.
        """
        if mps_info is None:
            return opt
        old_st = mps_info.right_state_info_no_trunc[i]
        new_st = mps_info.right_state_info[i]
        rmat = mps_info.get_right_rotation_matrix(i, mpst)
        if bra_mps_info is not None:
            old_bra_st = bra_mps_info.right_state_info_no_trunc[i]
            new_bra_st = bra_mps_info.right_state_info[i]
            bra_rmat = bra_mps_info.get_right_rotation_matrix(i, bra_mpst)
            sts = VectorStateInfo([old_bra_st, new_bra_st, old_st, new_st])
            rmats = VectorVectorMatrix([bra_rmat, rmat])
        else:
            sts = VectorStateInfo([old_st, new_st])
            rmats = VectorVectorMatrix([rmat])
        return self.tensor_rotate(opt, sts, rmats)
    
    @classmethod
    def left_contract(self, i, optl, optd, mpo_info, mps_info, bra_mps_info=None):
        """Perform blocking MPO x MPO for left block.
        
        Args:
            i : int
                Site index.
            optl: OperatorTensor or DualOperatorTensor
                Contracted MPO operator tensor at previous left block.
            optd : OperatorTensor or DualOperatorTensor
                MPO operator tensor at dot block.
            mpo_info : MPOInfo
                MPOInfo object.
            mps_info : MPSInfo
                MPSInfo object.
            bra_mps_info : MPSInfo or None (if same as mps_info)
                MPSInfo object for bra state.
        
        Returns:
            new_opt : OperatorTensor or DualOperatorTensor
                Operator tensor in untruncated basis in current left block.
        """
        op_names = mpo_info.left_operator_names[i]
        if mps_info is None:
            return optd.__class__(mat=op_names.reshape((1, -1)), ops={}, tags=optd.tags, contractor=optd.contractor)
        elif bra_mps_info is None:
            sts = VectorStateInfo([mps_info.left_state_info_no_trunc[i]])
        else:
            sts = VectorStateInfo([bra_mps_info.left_state_info_no_trunc[i], mps_info.left_state_info_no_trunc[i]])
        exprs = mpo_info.cached_exprs.get((i, '_LEFT'), None)
        if exprs is None:
            if isinstance(optd, OperatorTensor):
                new_mat = optl.mat @ optd.mat
            elif isinstance(optd, DualOperatorTensor):
                new_mat = optl.lmat @ optd.lmat
            zipped = [(op, expr) if op.factor == 1 else (abs(op), expr / op.factor)
                      for op, expr in zip(op_names, new_mat[0, :])]
            exprs = self.simplifier.simplify(zipped)
            if self.parallelizer is not None:
                exprs = self.parallelizer.parallelize(exprs, do_partial=True)
            if mpo_info.cache_contraction:
                mpo_info.cached_exprs[(i, '_LEFT')] = exprs
        with exprs() as (zipped, new_ops):
            for op, expr in zipped:
                new_ops[op] = self.expr_eval(expr, optl.ops, optd.ops, sts, op.q_label)
        if isinstance(optd, OperatorTensor):
            return OperatorTensor(mat=op_names.reshape((1, -1)),
                                  ops=new_ops, tags=optd.tags,
                                  contractor=optd.contractor)
        elif isinstance(optd, DualOperatorTensor):
            return DualOperatorTensor(lmat=op_names.reshape((1, -1)),
                                  ops=new_ops, tags=optd.tags,
                                  contractor=optd.contractor)
    
    @classmethod
    def right_contract(self, i, optr, optd, mpo_info, mps_info, bra_mps_info=None):
        """Perform blocking MPO x MPO for right block.
        
        Args:
            i : int
                Site index.
            optr: OperatorTensor or DualOperatorTensor
                Contracted MPO operator tensor at previous right block.
            optd : OperatorTensor or DualOperatorTensor
                MPO operator tensor at dot block.
            mpo_info : MPOInfo
                MPOInfo object.
            mps_info : MPSInfo
                MPSInfo object.
            bra_mps_info : MPSInfo or None (if same as mps_info)
                MPSInfo object for bra state.
        
        Returns:
            new_opt : OperatorTensor or DualOperatorTensor
                Operator tensor in untruncated basis in current right block.
        """
        op_names = mpo_info.right_operator_names[i]
        if mps_info is None:
            return optd.__class__(mat=op_names.reshape((-1, 1)), ops={}, tags=optd.tags, contractor=optd.contractor)
        elif bra_mps_info is None:
            sts = VectorStateInfo([mps_info.right_state_info_no_trunc[i]])
        else:
            sts = VectorStateInfo([bra_mps_info.right_state_info_no_trunc[i], mps_info.right_state_info_no_trunc[i]])
        exprs = mpo_info.cached_exprs.get((i, '_RIGHT'), None)
        if exprs is None:
            if isinstance(optd, OperatorTensor):
                new_mat = optd.mat @ optr.mat
            elif isinstance(optd, DualOperatorTensor):
                new_mat = optd.rmat @ optr.rmat
            zipped = [(op, expr) if op.factor == 1 else (abs(op), expr / op.factor)
                      for op, expr in zip(op_names, new_mat[:, 0])]
            exprs = self.simplifier.simplify(zipped)
            if self.parallelizer is not None:
                exprs = self.parallelizer.parallelize(exprs, do_partial=True)
            if mpo_info.cache_contraction:
                mpo_info.cached_exprs[(i, '_RIGHT')] = exprs
        with exprs() as (zipped, new_ops):
            for op, expr in zipped:
                new_ops[op] = self.expr_eval(expr, optd.ops, optr.ops, sts, op.q_label)
        if isinstance(optd, OperatorTensor):
            return OperatorTensor(mat=op_names.reshape((-1, 1)),
                                  ops=new_ops, tags=optd.tags,
                                  contractor=optd.contractor)
        elif isinstance(optd, DualOperatorTensor):
            return DualOperatorTensor(rmat=op_names.reshape((-1, 1)),
                                  ops=new_ops, tags=optd.tags,
                                  contractor=optd.contractor)
    
    @classmethod
    def left_right_contract(self, i, optl, optr, mpo_info, tag):
        """Symbolically construct the super block MPO.
        
        Args:
            i : int
                Site index of first/left dot block.
            optl: OperatorTensor
                Contracted MPO operator at (enlarged) left block.
            optr: OperatorTensor
                Contracted MPO operator at (enlarged) right block.
            mpo_info : MPOInfo
                MPOInfo object.
            tag : str
                Extra tag for caching.
        
        Returns:
            new_opt : OperatorTensor
                Operator tensor for super block.
                This method does not evaluate the super block operator experssion.
        """
        exprs = mpo_info.cached_exprs.get((i, tag, '_HAM'), None)
        if exprs is None:
            if mpo_info.middle_operators is None:
                new_mat = optl.mat @ optr.mat
                assert new_mat.shape == (1, 1)
                zipped = [(OpElement(OpNames.H, ()), new_mat[0, 0])]
            else:
                if tag == '_FUSE_LR' or tag == '_FUSE_L' or tag == '_NO_FUSE':
                    zipped = mpo_info.middle_operators[i]
                else:
                    zipped = mpo_info.middle_operators[i - 1]
            exprs = self.simplifier.simplify(zipped)
            if self.parallelizer is not None:
                exprs = self.parallelizer.parallelize(exprs, do_partial=True, bcast_all=True)
            if mpo_info.cache_contraction:
                mpo_info.cached_exprs[(i, tag, '_HAM')] = exprs
        new_mat = np.array([[exprs]], dtype=object)
        return OperatorTensor(mat=new_mat, ops=(optl.ops, optr.ops),
                              tags={'_HAM'}, contractor=optl.contractor)

    @classmethod
    def expr_diagonal_eval(self, expr, a, b, sts):
        """
        Evaluate the diagonal elements of the result of a symbolic operator expression.
        The diagonal elements are required for perconditioning in Davidson algorithm.
        
        Args:
            expr : OpString or OpCollection or ParaOpCollection
                The operator expression to evaluate.
            a : dict(OpElement -> StackSparseMatrix)
                A map from operator symbol in left block to its matrix representation.
            b : dict(OpElement -> StackSparseMatrix)
                A map from operator symbol in right block to its matrix representation.
            sts : VectorStateInfo
                StateInfo in which the result of the operator expression is represented.
        
        Returns:
            diag : DiagonalMatrix
        """
        diag = DiagonalMatrix()
        assert len(sts) == 1
        diag.resize(sts[0].n_total_states)
        diag.ref[:] = 0.0
        if isinstance(expr, OpString):
            assert len(expr.ops) == 2
            if a[expr.ops[0]] == 0 or b[expr.ops[1]] == 0:
                return diag
            factor = float(expr.factor) * a[expr.ops[0]].symm_scale * b[expr.ops[1]].symm_scale
            if expr.ops[0] == OpElement(OpNames.I, ()):
                tensor_trace_diagonal(b[expr.ops[1]], diag, sts, False, factor)
            elif expr.ops[1] == OpElement(OpNames.I, ()):
                tensor_trace_diagonal(a[expr.ops[0]], diag, sts, True, factor)
            else:
                tensor_product_diagonal(a[expr.ops[0]], b[expr.ops[1]], diag, sts, factor)
            return diag
        elif isinstance(expr, OpCollection):
            with expr() as (zipped, new_ops):
                (op, expr), = zipped
                assert op == OpElement(OpNames.H, ())
                if expr != 0:
                    for x in expr.strings if isinstance(expr, OpSum) else [expr]:
                        diag = diag + self.expr_diagonal_eval(x, a, b, sts)
                new_ops[op] = diag
            return diag
        else:
            assert False
    
    @classmethod
    def expr_multiply_eval(self, expr, a, b, c, nwave, sts):
        """
        Evaluate the result of a symbolic operator expression applied on a wavefunction.
        
        Args:
            expr : OpString or OpCollection or ParaOpCollection
                The operator expression.
            a : dict(OpElement -> StackSparseMatrix)
                A map from operator symbol in left block to its matrix representation.
            b : dict(OpElement -> StackSparseMatrix)
                A map from operator symbol in right block to its matrix representation.
            c : Wavefunction
                The input wavefuction.
            nwave : Wavefunction
                The output wavefuction.
            sts : VectorStateInfo
                StateInfo in which the wavefuction is represented.
        """
        if isinstance(expr, OpString):
            assert len(expr.ops) == 2
            if a[expr.ops[0]] == 0 or b[expr.ops[1]] == 0:
                return
            factor = float(expr.factor) * a[expr.ops[0]].symm_scale * b[expr.ops[1]].symm_scale
            if expr.ops[0] == OpElement(OpNames.I, ()) and len(sts) == 1:
                tensor_trace_multiply(b[expr.ops[1]], c, nwave, sts[0], False, factor)
            elif expr.ops[1] == OpElement(OpNames.I, ()) and len(sts) == 1:
                tensor_trace_multiply(a[expr.ops[0]], c, nwave, sts[0], True, factor)
            else:
                aq, bq = a[expr.ops[0]].delta_quantum[0], b[expr.ops[1]].delta_quantum[0]
                op_q = (aq + bq)[0]
                tensor_product_multiply(a[expr.ops[0]], b[expr.ops[1]], c, nwave, sts, op_q, factor)
        elif isinstance(expr, OpCollection):
            with expr() as (zipped, new_ops):
                (op, expr), = zipped
                assert op == OpElement(OpNames.H, ())
                if expr != 0:
                    for x in expr.strings if isinstance(expr, OpSum) else [expr]:
                        self.expr_multiply_eval(x, a, b, c, nwave, sts)
                new_ops[op] = nwave
        else:
            assert False

    @classmethod
    def expr_expectation(self, expr, a, b, ket, bra, work, sts):
        if isinstance(expr, OpCollection):
            with expr() as (zipped, new_ops):
                for op, expr in zipped:
                    if expr != 0:
                        work.clear()
                        for x in expr.strings if isinstance(expr, OpSum) else [expr]:
                            self.expr_multiply_eval(x, a, b, ket, work, sts)
                        new_ops[op] = tensor_dot_product(bra, work)
                    else:
                        new_ops[op] = 0
            return new_ops
        else:
            assert False

    @classmethod
    def expr_eval(self, expr, a, b, sts, q_label, nmat=0):
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
            sts : VectorStateInfo
                StateInfo in which the result of the operator expression is represented.
            q_label : DirectProdGroup
                Quantum label of the result operator
                (indicating how it changes the state quantum labels).
        
        Returns:
            nmat : StackSparseMatrix
        """
        if isinstance(expr, OpString):
            assert len(expr.ops) == 2
            if a[expr.ops[0]] == 0 or b[expr.ops[1]] == 0:
                return nmat
            factor = float(expr.factor) * a[expr.ops[0]].symm_scale * b[expr.ops[1]].symm_scale
            if nmat == 0:
                nmat = StackSparseMatrix()
                cq = BlockSymmetry.to_spin_quantum(q_label)
                nmat.delta_quantum = VectorSpinQuantum([cq])
                nmat.fermion = a[expr.ops[0]].fermion ^ b[expr.ops[1]].fermion
                nmat.allocate(sts)
                nmat.initialized = True
            if expr.ops[0] == OpElement(OpNames.I, ()) and len(sts) == 1:
                tensor_trace(b[expr.ops[1]], nmat, sts, False, factor)
            elif expr.ops[1] == OpElement(OpNames.I, ()) and len(sts) == 1:
                tensor_trace(a[expr.ops[0]], nmat, sts, True, factor)
            else:
                tensor_product(a[expr.ops[0]], b[expr.ops[1]], nmat, sts, factor)
            return nmat
        elif expr == 0:
            return nmat
        elif isinstance(expr, OpSum):
            for x in expr.strings:
                assert not isinstance(x, OpSum)
                nmat = self.expr_eval(x, a, b, sts, q_label, nmat)
            return nmat
        else:
            assert isinstance(expr, OpShell)
            nmat = StackSparseMatrix()
            cq = BlockSymmetry.to_spin_quantum(q_label)
            nmat.delta_quantum = VectorSpinQuantum([cq])
            nmat.fermion = a[expr.data.ops[0]].fermion ^ b[expr.data.ops[1]].fermion
            nmat.allocate(sts)
            nmat.initialized = True
            return nmat
    
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
    def from_state_info(self, state_info):
        """Translate from StateInfo (block code) to [(:class:`DirectProdGroup`, int)]."""
        n = len(state_info.quanta)
        states = []
        for i in range(n):
            q = self.from_spin_quantum(state_info.quanta[i])
            nq = state_info.n_states[i]
            states.append((q, nq))
        return states

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
        
        if 'page' in kwargs:
            self.page = kwargs['page']
            del kwargs['page']
        else:
            self.page = None

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
            
            if self.page is not None:
                input['scratch'] = str(self.page.save_dir)

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
                elif k == 'omp_threads':
                    input['quanta_thrds'] = str(v)
                elif k == 'mkl_threads':
                    input['mkl_thrds'] = str(v)
                elif k == 'max_m':
                    input['maxM'] = str(v)
                elif k == 'memory':
                    input['memory'] = str(v) + ' m'
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
        self.spin_adapted = Global.dmrginp.is_spin_adapted
        if Global.dmrginp.algorithm_type == AlgorithmTypes.TwoDotToOneDot:
            raise BlockError('Currently two dot to one dot is not supported.')
        self.dot = 1 if Global.dmrginp.algorithm_type == AlgorithmTypes.OneDot else 2

        if self.page is None:
            init_stack_memory()
            MPS.site_blocks = VectorBlock([])
            MPS_init(True)
        else:
            self.page.initialize()
        
        self.PG = point_group(self.point_group)
        self.spatial = [self.PG.IrrepNames[ir] for ir in self.spatial_syms]

        if self.spin_adapted:
            self.empty = ParticleN(0) * SU2(0) * self.PG(0)
            self.site_basis = [{
                ParticleN(0) * SU2(0) * self.PG(0): 1,
                ParticleN(1) * SU2(Fraction(1, 2)) * self.PG(sp): 1,
                ParticleN(2) * SU2(0) * self.PG(0): 1
            } for sp in self.spatial]
            self.target = ParticleN(self.n_electrons) * SU2(self.target_s) * self.PG(self.target_spatial_sym)
        else:
            self.empty = ParticleN(0) * SZ(0) * self.PG(0)
            self.site_basis = [{
                ParticleN(0) * SZ(0) * self.PG(0): 1,
                ParticleN(1) * SZ(-Fraction(1, 2)) * self.PG(sp): 1,
                ParticleN(1) * SZ(Fraction(1, 2)) * self.PG(sp): 1,
                ParticleN(2) * SZ(0) * self.PG(0): 1
            } for sp in self.spatial]
            self.target = ParticleN(self.n_electrons) * SZ(self.target_s) * self.PG(self.target_spatial_sym)
        
        self.one_site_q = np.array([ParticleN(1) * SU2(Fraction(1, 2)) * self.PG(sp)
                                    for sp in self.spatial], dtype=object)
        self.two_site_plus_q = np.array([[self.one_site_q[i] + self.one_site_q[j] for i in range(self.n_sites)]
                                    for j in range(self.n_sites)], dtype=object)
        self.two_site_minus_q = np.array([[self.one_site_q[i] - self.one_site_q[j] for i in range(self.n_sites)]
                                    for j in range(self.n_sites)], dtype=object)
        
        self.site_state_info = [VectorStateInfo([BlockSymmetry.to_state_info(b.items())]) for b in self.site_basis]
    
    def get_site_operators(self, m, op_set):
        """Return operator representations dict(OpElement -> StackSparseMatrix) at site m."""
        ops = {}
        
        if self.spin_adapted:
            
            mat = StackSparseMatrix()
            mat.fermion = False
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.empty)])
            mat.allocate(self.site_state_info[m])
            mat.initialized = True
            mat.operator_element(0, 0).ref[0, 0] = 1.0
            mat.operator_element(1, 1).ref[0, 0] = 1.0
            mat.operator_element(2, 2).ref[0, 0] = 1.0
            ops[OpElement(OpNames.I, ())] = mat
            
            if op_set == { OpElement(OpNames.I, ()) }:
                return ops
            
            if OpElement(OpNames.N, ()) in op_set or OpElement(OpNames.NN, ()) in op_set:
                
                mat = StackSparseMatrix()
                mat.fermion = False
                mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.empty)])
                mat.allocate(self.site_state_info[m])
                mat.initialized = True
                mat.operator_element(0, 0).ref[0, 0] = 0.0
                mat.operator_element(1, 1).ref[0, 0] = 1.0
                mat.operator_element(2, 2).ref[0, 0] = 2.0
                ops[OpElement(OpNames.N, ())] = mat
                
                if OpElement(OpNames.NN, ()) in op_set:
                    
                    mat2 = StackSparseMatrix()
                    mat2.deep_clear_copy(mat)
                    product(mat, mat, mat2, self.site_state_info[m][0], 1.0)
                    ops[OpElement(OpNames.NN, ())] = mat2
                
                return ops
            
            mat = StackSparseMatrix()
            mat.fermion = False
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.empty)])
            mat.allocate(self.site_state_info[m])
            mat.initialized = True
            mat.operator_element(0, 0).ref[0, 0] = 0.0
            mat.operator_element(1, 1).ref[0, 0] = self.t[m, m]
            mat.operator_element(2, 2).ref[0, 0] = self.t[m, m] * 2 + self.v[m, m, m, m]
            ops[OpElement(OpNames.H, ())] = mat
            
            mat = StackSparseMatrix()
            mat.fermion = True
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.one_site_q[m])])
            mat.allocate(self.site_state_info[m])
            mat.initialized = True
            mat.operator_element(1, 0).ref[0, 0] = 1.0
            mat.operator_element(2, 1).ref[0, 0] = -np.sqrt(2)
            ops[OpElement(OpNames.C, (m, ))] = mat
            
            mat = StackSparseMatrix()
            mat.fermion = True
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(-self.one_site_q[m])])
            mat.allocate(self.site_state_info[m])
            mat.initialized = True
            mat.operator_element(0, 1).ref[0, 0] = np.sqrt(2)
            mat.operator_element(1, 2).ref[0, 0] = 1.0
            ops[OpElement(OpNames.D, (m, ))] = mat
        
        else:
            raise BlockError('non spin-adapted case not implemented.')
        
        for s in [0, 1]:
            
            mat = StackSparseMatrix()
            mat.fermion = False
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.two_site_plus_q[m, m][s])])
            mat.allocate(self.site_state_info[m])
            mat.initialized = True
            product(ops[OpElement(OpNames.C, (m, ))], ops[OpElement(OpNames.C, (m, ))],
                    mat, self.site_state_info[m][0], 1.0)
            ops[OpElement(OpNames.A, (m, m, s))] = mat

            mat = StackSparseMatrix()
            mat.fermion = False
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(-self.two_site_plus_q[m, m][s])])
            mat.allocate(self.site_state_info[m])
            mat.initialized = True
            product(ops[OpElement(OpNames.D, (m, ))], ops[OpElement(OpNames.D, (m, ))],
                    mat, self.site_state_info[m][0], 1.0)
            ops[OpElement(OpNames.AD, (m, m, s))] = mat
        
            mat = StackSparseMatrix()
            mat.fermion = False
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.two_site_minus_q[m, m][s])])
            mat.allocate(self.site_state_info[m])
            mat.initialized = True
            product(ops[OpElement(OpNames.C, (m, ))], ops[OpElement(OpNames.D, (m, ))],
                    mat, self.site_state_info[m][0], 1.0)
            ops[OpElement(OpNames.B, (m, m, s))] = mat
        
        for i in range(0, self.n_sites):
            
            if i == m:
                continue
            
            if np.isclose(self.t[i, m], 0) and np.isclose(self.v[i, m, m, m], 0):
                
                if OpElement(OpNames.R, (i, )) in op_set:
                    ops[OpElement(OpNames.R, (i, ))] = 0
                
                if OpElement(OpNames.RD, (i, )) in op_set:
                    ops[OpElement(OpNames.RD, (i, ))] = 0
                
            else:
                
                if OpElement(OpNames.R, (i, )) in op_set:
                
                    mat = StackSparseMatrix()
                    mat.deep_copy(ops[OpElement(OpNames.D, (m, ))])
                    assert mat.fermion
                    tensor_scale(self.t[i, m] * np.sqrt(2) * 0.25, mat)
                    mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(-self.one_site_q[m])])

                    mat2 = StackSparseMatrix()
                    mat2.deep_clear_copy(ops[OpElement(OpNames.D, (m, ))])
                    product(ops[OpElement(OpNames.B, (m, m, 0))], ops[OpElement(OpNames.D, (m, ))],
                            mat2, self.site_state_info[m][0], 1.0)
                    tensor_scale_add_no_trans(self.v[i, m, m, m], mat2, mat)
                    mat2.deallocate()
                    ops[OpElement(OpNames.R, (i, ))] = mat
                
                if OpElement(OpNames.RD, (i, )) in op_set:
            
                    mat = StackSparseMatrix()
                    mat.deep_copy(ops[OpElement(OpNames.C, (m, ))])
                    assert mat.fermion
                    tensor_scale(self.t[i, m] * np.sqrt(2) * 0.25, mat)
                    mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.one_site_q[m])])

                    mat2 = StackSparseMatrix()
                    mat2.deep_clear_copy(ops[OpElement(OpNames.C, (m, ))])
                    product(ops[OpElement(OpNames.C, (m, ))], ops[OpElement(OpNames.B, (m, m, 0))],
                            mat2, self.site_state_info[m][0], 1.0)
                    tensor_scale_add_no_trans(self.v[i, m, m, m], mat2, mat)
                    mat2.deallocate()
                    ops[OpElement(OpNames.RD, (i, ))] = mat
            
            if self.spatial_syms[m] != self.spatial_syms[i]:
                assert np.isclose(self.t[m, i], 0.0)
        
        for s in [0, 1]:
            for i in range(0, self.n_sites):
                for k in range(0, self.n_sites):
                    if i != m and k != m:
                        
                        if OpElement(OpNames.P, (i, k, s)) not in op_set:
                            continue
                        
                        if np.isclose(self.v[i, m, k, m], 0):
                            ops[OpElement(OpNames.P, (i, k, s))] = 0
                        else:
                            mat = StackSparseMatrix()
                            mat.deep_copy(ops[OpElement(OpNames.AD, (m, m, s))])
                            assert not mat.fermion
                            tensor_scale(self.v[i, m, k, m], mat)
                            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(-self.two_site_plus_q[i, k][s])])
                            ops[OpElement(OpNames.P, (i, k, s))] = mat
        
        for s in [0, 1]:
            for i in range(0, self.n_sites):
                for k in range(0, self.n_sites):
                    if i != m and k != m:
                        
                        if OpElement(OpNames.PD, (i, k, s)) not in op_set:
                            continue
                        
                        if np.isclose(self.v[i, m, k, m], 0):
                            ops[OpElement(OpNames.PD, (i, k, s))] = 0
                        else:
                            mat = StackSparseMatrix()
                            mat.deep_copy(ops[OpElement(OpNames.A, (m, m, s))])
                            assert not mat.fermion
                            tensor_scale(self.v[i, m, k, m], mat)
                            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.two_site_plus_q[i, k][s])])
                            ops[OpElement(OpNames.PD, (i, k, s))] = mat
        
        for i in range(0, self.n_sites):
            for j in range(0, self.n_sites):
                if i != m and j != m:
                    
                    if OpElement(OpNames.Q, (i, j, 0)) not in op_set:
                        continue
                    
                    if np.isclose(2 * self.v[i, j, m, m] -  self.v[i, m, m, j], 0):
                        ops[OpElement(OpNames.Q, (i, j, 0))] = 0
                    else:
                        mat = StackSparseMatrix()
                        mat.deep_copy(ops[OpElement(OpNames.B, (m, m, 0))])
                        assert not mat.fermion
                        tensor_scale(2 * self.v[i, j, m, m] -  self.v[i, m, m, j], mat)
                        mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.two_site_minus_q[i, j][0])])
                        ops[OpElement(OpNames.Q, (i, j, 0))] = mat
                
        for i in range(0, self.n_sites):
            for j in range(0, self.n_sites):
                if i != m and j != m:
                    
                    if OpElement(OpNames.Q, (i, j, 1)) not in op_set:
                        continue
                    
                    if np.isclose(self.v[i, m, m, j], 0):
                        ops[OpElement(OpNames.Q, (i, j, 1))] = 0
                    else:
                        mat = StackSparseMatrix()
                        mat.deep_copy(ops[OpElement(OpNames.B, (m, m, 1))])
                        assert not mat.fermion
                        tensor_scale(self.v[i, m, m, j], mat)
                        mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.two_site_minus_q[i, j][1])])
                        ops[OpElement(OpNames.Q, (i, j, 1))] = mat
        
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
    
    @staticmethod
    @contextlib.contextmanager
    def get(fcidump, pg, su2, dot=2, output_level=0, memory=2000, page=None, omp_threads=1, **kwargs):
        ham = BlockHamiltonian(fcidump=fcidump, point_group=pg,
                               dot=dot, spin_adapted=su2, page=page,
                               output_level=output_level, memory=memory,
                               omp_threads=omp_threads, **kwargs)
        try:
            yield ham
        finally:
            if page is None:
                BlockHamiltonian.set_current_memory(0)
                BlockHamiltonian.release_memory()
            else:
                page.release()
