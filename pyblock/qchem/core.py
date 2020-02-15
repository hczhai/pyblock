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
from block.rev import tensor_scale_add, tensor_trace_diagonal, tensor_product_diagonal
from block.rev import tensor_trace_multiply, tensor_product_multiply, product
from block.rev import tensor_scale_add_no_trans

from ..symmetry.symmetry import ParticleN, SU2, SZ, PointGroup, point_group
from ..symmetry.symmetry import DirectProdGroup
from .operator import OpElement, OpNames, OpString, OpSum
from .fcidump import read_fcidump
from fractions import Fraction
import contextlib
import numpy as np
                
class BlockError(Exception):
    pass


class BlockEvaluation:
    """Explicit evaluation of symbolic expression for operators."""
    @classmethod
    def tensor_rotate(self, opt, old_st, new_st, rmat):
        """
        Transform basis of MPO using rotation matrix.
        
        Args:
            opt : OperatorTensor
                One-site operator tensor in (untruncated) old basis.
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
        for k, v in opt.ops.items():
            nmat = StackSparseMatrix()
            nmat.delta_quantum = v.delta_quantum
            nmat.fermion = v.fermion
            nmat.allocate(new_st)
            nmat.initialized = True
            state_info = VectorStateInfo([old_st, new_st])
            tensor_rotate(v, nmat, state_info, rmat)
            new_ops[k] = nmat
        return opt.__class__(mat=opt.mat, ops=new_ops, tags=opt.tags,
                             contractor=opt.contractor)
    
    @classmethod
    def left_rotate(self, opt, mpst, mps_info, i):
        """Perform rotation <MPS|MPO|MPS> for left block.
        
        Args:
            opt : OperatorTensor
                One-site operator tensor in (untruncated) old basis.
            mpst : Tensor
                MPS tensor
            mps_info : MPSInfo
                MPSInfo object.
            i : int
                Site index.
        
        Returns:
            new_mpo : BlockMPO
                One-site MPO in (truncated) new basis.
        """
        old_st = mps_info.left_state_info_no_trunc[i]
        new_st = mps_info.left_state_info[i]
        rmat = mps_info.get_left_rotation_matrix(i, mpst)
        return self.tensor_rotate(opt, old_st, new_st, rmat)
    
    @classmethod
    def right_rotate(self, opt, mpst, mps_info, i):
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
        old_st = mps_info.right_state_info_no_trunc[i]
        new_st = mps_info.right_state_info[i]
        rmat = mps_info.get_right_rotation_matrix(i, mpst)
        return self.tensor_rotate(opt, old_st, new_st, rmat)
    
    @classmethod
    def left_contract(self, optl, optd, mps_info, mpo_info, i):
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
        new_mat = optl.mat @ optd.mat
        st = mps_info.left_state_info_no_trunc[i]
        op_names = mpo_info.right_operator_names[i]
        new_ops = {}
        for j in range(new_mat.shape[1]):
            ql = op_names[j].q_label
            if op_names[j].factor != 1:
                new_ops[abs(op_names[j])] = \
                    self.expr_eval(new_mat[0, j] / op_names[j].factor, optl.ops, optd.ops, st, ql)
            else:
                new_ops[op_names[j]] = self.expr_eval(new_mat[0, j], optl.ops, optd.ops, st, ql)
        return optd.__class__(mat=op_names.reshape(new_mat.shape),
                              ops=new_ops, tags=optd.tags,
                              contractor=optd.contractor)
    
    @classmethod
    def right_contract(self, optr, optd, mps_info, mpo_info, i):
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
        new_mat = optd.mat @ optr.mat
        st = mps_info.right_state_info_no_trunc[i]
        op_names = mpo_info.left_operator_names[i]
        new_ops = {}
        for j in range(new_mat.shape[0]):
            ql = op_names[j].q_label
            if op_names[j].factor != 1:
                new_ops[abs(op_names[j])] = \
                    self.expr_eval(new_mat[j, 0] / op_names[j].factor, optd.ops, optr.ops, st, ql)
            else:
                new_ops[op_names[j]] = self.expr_eval(new_mat[j, 0], optd.ops, optr.ops, st, ql)
        return optd.__class__(mat=op_names.reshape(new_mat.shape),
                              ops=new_ops, tags=optd.tags,
                              contractor=optd.contractor)
    
    @classmethod
    def left_right_contract(self, optl, optr):
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
        new_mat = optl.mat @ optr.mat
        assert new_mat.shape == (1, 1)
        return optl.__class__(mat=new_mat, ops=(optl.ops, optr.ops),
                              tags={'_HAM'}, contractor=optl.contractor)
    
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
                tensor_trace_diagonal(b[expr.ops[1]], diag, state_info, False, float(expr.factor))
            elif expr.ops[1] == OpElement(OpNames.I, ()):
                tensor_trace_diagonal(a[expr.ops[0]], diag, state_info, True, float(expr.factor))
            else:
                tensor_product_diagonal(a[expr.ops[0]], b[expr.ops[1]], diag, state_info, float(expr.factor))
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
                tensor_trace_multiply(b[expr.ops[1]], c, nwave, st, False, float(expr.factor))
            elif expr.ops[1] == OpElement(OpNames.I, ()):
                tensor_trace_multiply(a[expr.ops[0]], c, nwave, st, True, float(expr.factor))
            else:
                aq, bq = a[expr.ops[0]].delta_quantum[0], b[expr.ops[1]].delta_quantum[0]
                op_q = (aq + bq)[0]
                tensor_product_multiply(a[expr.ops[0]], b[expr.ops[1]], c, nwave, st, op_q, float(expr.factor))
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
                tensor_trace(b[expr.ops[1]], nmat, state_info, False, float(expr.factor))
            elif expr.ops[1] == OpElement(OpNames.I, ()):
                nmat.delta_quantum = a[expr.ops[0]].delta_quantum
                nmat.fermion = a[expr.ops[0]].fermion
                nmat.allocate(st)
                nmat.initialized = True
                tensor_trace(a[expr.ops[0]], nmat, state_info, True, float(expr.factor))
            else:
                cq = BlockSymmetry.to_spin_quantum(q_label)
                nmat.delta_quantum = VectorSpinQuantum([cq])
                nmat.fermion = a[expr.ops[0]].fermion ^ b[expr.ops[1]].fermion
                nmat.allocate(st)
                nmat.initialized = True
                tensor_product(a[expr.ops[0]], b[expr.ops[1]], nmat, state_info, float(expr.factor))
            return nmat
        elif isinstance(expr, OpSum):
            nmat = StackSparseMatrix()
            cq = BlockSymmetry.to_spin_quantum(q_label)
            nmat.delta_quantum = VectorSpinQuantum([cq])
            nmat.fermion = a[expr.strings[0].ops[0]].fermion ^ b[expr.strings[0].ops[1]].fermion
            nmat.allocate(st)
            nmat.initialized = True
            assert nmat.conjugacy == 'n'
            for x in expr.strings:
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
                elif k == 'memory':
                    input['memory'] = str(v) + ' g'
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

        init_stack_memory()
        
        MPS.site_blocks = VectorBlock([])
        MPS_init(True)
        
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
    
    def get_site_operators_1e(self, i):
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
        ops[OpElement(OpNames.C, (i, ))] = mat
        
        mat = StackSparseMatrix()
        mat.deep_copy(block.ops[OpTypes.Des].local_element_linear(0)[0])
        assert mat.fermion
        ops[OpElement(OpNames.D, (i, ))] = mat
        
        for j in range(0, self.n_sites):
            
            if j == i:
                continue
            
            mat = StackSparseMatrix()
            mat.deep_copy(block.ops[OpTypes.Des].local_element_linear(0)[0])
            assert mat.fermion
            tensor_scale(self.t[j, i] * np.sqrt(2) * 0.25, mat)
            ops[OpElement(OpNames.S, (j, ))] = mat
            ql = mat.delta_quantum[0]
            ql.symm = IrrepSpace(self.spatial_syms[j])
            mat.delta_quantum = VectorSpinQuantum([ql])
            
            mat = StackSparseMatrix()
            mat.deep_copy(block.ops[OpTypes.Cre].local_element_linear(0)[0])
            assert mat.fermion
            tensor_scale(self.t[j, i] * np.sqrt(2) * 0.25, mat)
            ops[OpElement(OpNames.SD, (j, ))] = mat
            ql = mat.delta_quantum[0]
            ql.symm = IrrepSpace(self.spatial_syms[j])
            mat.delta_quantum = VectorSpinQuantum([ql])
            
            if self.spatial_syms[j] != self.spatial_syms[i]:
                assert np.isclose(self.t[j, i], 0.0)
        
        # TODO :: need to store Block to deallocate it later
        return ops
    
    def get_site_operators(self, m):
        """Return operator representations dict(OpElement -> StackSparseMatrix) at site m."""
        ops = {}
        block = Block(m, m, 0, False)
        
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
        ops[OpElement(OpNames.C, (m, ))] = mat
        
        mat = StackSparseMatrix()
        mat.deep_copy(block.ops[OpTypes.Des].local_element_linear(0)[0])
        assert mat.fermion
        ops[OpElement(OpNames.D, (m, ))] = mat
        
        for s in [0, 1]:
            mat = StackSparseMatrix()
            mat.fermion = False
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.two_site_plus_q[m, m][s])])
            mat.allocate(BlockSymmetry.initial_state_info(m))
            mat.initialized = True
            product(ops[OpElement(OpNames.C, (m, ))], ops[OpElement(OpNames.C, (m, ))],
                    mat, BlockSymmetry.initial_state_info(m), 1.0)
            ops[OpElement(OpNames.A, (m, m, s))] = mat
        
        for s in [0, 1]:
            mat = StackSparseMatrix()
            mat.fermion = False
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(-self.two_site_plus_q[m, m][s])])
            mat.allocate(BlockSymmetry.initial_state_info(m))
            mat.initialized = True
            product(ops[OpElement(OpNames.D, (m, ))], ops[OpElement(OpNames.D, (m, ))],
                    mat, BlockSymmetry.initial_state_info(m), 1.0)
            ops[OpElement(OpNames.AD, (m, m, s))] = mat
        
        for s in [0, 1]:
            mat = StackSparseMatrix()
            mat.fermion = False
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.two_site_minus_q[m, m][s])])
            mat.allocate(BlockSymmetry.initial_state_info(m))
            mat.initialized = True
            product(ops[OpElement(OpNames.C, (m, ))], ops[OpElement(OpNames.D, (m, ))],
                    mat, BlockSymmetry.initial_state_info(m), 1.0)
            ops[OpElement(OpNames.B, (m, m, s))] = mat
        
        for i in range(0, self.n_sites):
            
            if i == m:
                continue
            
            mat = StackSparseMatrix()
            mat.deep_copy(ops[OpElement(OpNames.D, (m, ))])
            assert mat.fermion
            tensor_scale(self.t[i, m] * np.sqrt(2) * 0.25, mat)
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(-self.one_site_q[m])])
            
            mat2 = StackSparseMatrix()
            mat2.deep_clear_copy(ops[OpElement(OpNames.D, (m, ))])
            product(ops[OpElement(OpNames.B, (m, m, 0))], ops[OpElement(OpNames.D, (m, ))],
                    mat2, BlockSymmetry.initial_state_info(m), 1.0)
            tensor_scale_add_no_trans(self.v[i, m, m, m], mat2, mat)
            mat2.deallocate()
            ops[OpElement(OpNames.R, (i, ))] = mat
            
            mat = StackSparseMatrix()
            mat.deep_copy(ops[OpElement(OpNames.C, (m, ))])
            assert mat.fermion
            tensor_scale(self.t[i, m] * np.sqrt(2) * 0.25, mat)
            mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.one_site_q[m])])
            
            mat2 = StackSparseMatrix()
            mat2.deep_clear_copy(ops[OpElement(OpNames.C, (m, ))])
            product(ops[OpElement(OpNames.C, (m, ))], ops[OpElement(OpNames.B, (m, m, 0))], 
                    mat2, BlockSymmetry.initial_state_info(m), 1.0)
            tensor_scale_add_no_trans(self.v[i, m, m, m], mat2, mat)
            mat2.deallocate()
            ops[OpElement(OpNames.RD, (i, ))] = mat
            
            if self.spatial_syms[m] != self.spatial_syms[i]:
                assert np.isclose(self.t[m, i], 0.0)
        
        for s in [0, 1]:
            for i in range(0, self.n_sites):
                for k in range(0, self.n_sites):
                    if i != m and k != m:
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
                        mat = StackSparseMatrix()
                        mat.deep_copy(ops[OpElement(OpNames.A, (m, m, s))])
                        assert not mat.fermion
                        tensor_scale(self.v[i, m, k, m], mat)
                        mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.two_site_plus_q[i, k][s])])
                        ops[OpElement(OpNames.PD, (i, k, s))] = mat
        
        for i in range(0, self.n_sites):
            for j in range(0, self.n_sites):
                if i != m and j != m:
                    mat = StackSparseMatrix()
                    mat.deep_copy(ops[OpElement(OpNames.B, (m, m, 0))])
                    assert not mat.fermion
                    tensor_scale(2 * self.v[i, j, m, m] -  self.v[i, m, m, j], mat)
                    mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.two_site_minus_q[i, j][0])])
                    ops[OpElement(OpNames.Q, (i, j, 0))] = mat
                
        for i in range(0, self.n_sites):
            for j in range(0, self.n_sites):
                if i != m and j != m:
                    mat = StackSparseMatrix()
                    mat.deep_copy(ops[OpElement(OpNames.B, (m, m, 1))])
                    assert not mat.fermion
                    tensor_scale(self.v[i, m, m, j], mat)
                    mat.delta_quantum = VectorSpinQuantum([BlockSymmetry.to_spin_quantum(self.two_site_minus_q[i, j][1])])
                    ops[OpElement(OpNames.Q, (i, j, 1))] = mat
        
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
    
    @staticmethod
    @contextlib.contextmanager
    def get(fcidump, pg, su2, dot=2, output_level=0, memory=2):
        ham = BlockHamiltonian(fcidump=fcidump, point_group=pg,
                               dot=dot, spin_adapted=su2,
                               output_level=output_level, memory=memory)
        try:
            yield ham
        finally:
            BlockHamiltonian.set_current_memory(0)
            BlockHamiltonian.release_memory()
