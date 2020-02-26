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
Matrix Product Operator for quantum chemistry calculations.
"""

from .operator import OpElement, OpSum, OpNames
from ..tensor.tensor import Tensor, TensorNetwork
import numpy as np

class OperatorTensor(Tensor):
    """
    Represent MPO tensor or contracted MPO tensor.
    
    Attributes:
        mat : numpy.ndarray(dtype=OpExpression)
            2-D array of Symbolic operator expressions.
        ops : dict(OpElement -> StackSparseMatrix)
            Numeric representation of operator symbols.
            When the object is the super block MPO, :attr:`ops` is a pair of dicts representing
            operator symbols for left and right blocks, respectively.
    """
    def __init__(self, mat, ops, tags=None, contractor=None):
        self.mat = mat
        self.ops = ops
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
            tags=self.tags.copy(), contractor=self.contractor)

class MPOInfo:
    def __init__(self, hamil, cache_contraction=True):
        self.hamil = hamil
        self.n_sites = hamil.n_sites
        self._init_operator_names()
        self.cache_contraction = cache_contraction
        self.cached_exprs = {}
    
    def _init_operator_names(self):
        self.left_operator_names = [None] * self.n_sites
        self.right_operator_names = [None] * self.n_sites
        
        for i in range(self.n_sites):
            lshape = 2 + 2 * self.n_sites + 6 * i * i if i != 0 else 1
            rshape = 2 + 2 * self.n_sites + 6 * (i + 1) * (i + 1) if i != self.n_sites - 1 else 1
            lop = np.zeros((lshape, ), dtype=object)
            rop = np.zeros((rshape, ), dtype=object)
            rop[0] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
            if i != self.n_sites - 1:
                rop[1] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                p = 2
                for j in range(i + 1):
                    rop[p + j] = OpElement(OpNames.C, (j, ), q_label=self.hamil.one_site_q[j])
                p += i + 1
                for j in range(i + 1):
                    rop[p + j] = OpElement(OpNames.D, (j, ), q_label=-self.hamil.one_site_q[j])
                p += i + 1
                for j in range(i + 1, self.n_sites):
                    rop[p + j - i - 1] = 2.0 * OpElement(OpNames.RD, (j, ), q_label=self.hamil.one_site_q[j])
                p += self.n_sites - (i + 1)
                for j in range(i + 1, self.n_sites):
                    rop[p + j - i - 1] = 2.0 * OpElement(OpNames.R, (j, ), q_label=-self.hamil.one_site_q[j])
                p += self.n_sites - (i + 1)
                for s in [0, 1]:
                    for j in range(i + 1):
                        for k in range(i + 1):
                            rop[p + k] = OpElement(OpNames.A, (j, k, s), q_label=self.hamil.two_site_plus_q[j, k][s])
                        p += i + 1
                for s in [0, 1]:
                    for j in range(i + 1):
                        for k in range(i + 1):
                            rop[p + k] = OpElement(OpNames.AD, (j, k, s), q_label=-self.hamil.two_site_plus_q[j, k][s])
                        p += i + 1
                for s in [0, 1]:
                    for j in range(i + 1):
                        for k in range(i + 1):
                            rop[p + k] = OpElement(OpNames.B, (j, k, s), q_label=self.hamil.two_site_minus_q[j, k][s])
                        p += i + 1
                assert p == rop.shape[0]
            lop[0] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
            if i != 0:
                lop[1] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
                p = 2
                for j in range(i):
                    lop[p + j] = 2.0 * OpElement(OpNames.R, (j, ), q_label=-self.hamil.one_site_q[j])
                p += i
                for j in range(i):
                    lop[p + j] = 2.0 * OpElement(OpNames.RD, (j, ), q_label=self.hamil.one_site_q[j])
                p += i
                for j in range(i, self.n_sites):
                    lop[p + j - i] = OpElement(OpNames.D, (j, ), q_label=-self.hamil.one_site_q[j])
                p += self.n_sites - i
                for j in range(i, self.n_sites):
                    lop[p + j - i] = OpElement(OpNames.C, (j, ), q_label=self.hamil.one_site_q[j])
                p += self.n_sites - i
                su2_factor = [-0.5, -0.5 * np.sqrt(3.0)]
                for s in [0, 1]:
                    for j in range(i):
                        for k in range(i):
                            lop[p + k] = su2_factor[s] * \
                                OpElement(OpNames.P, (j, k, s), q_label=-self.hamil.two_site_plus_q[j, k][s])
                        p += i
                for s in [0, 1]:
                    for j in range(i):
                        for k in range(i):
                            lop[p + k] = su2_factor[s] * \
                                OpElement(OpNames.PD, (j, k, s), q_label=self.hamil.two_site_plus_q[j, k][s])
                        p += i
                su2_factor = [1.0, np.sqrt(3.0)]
                for s in [0, 1]:
                    for j in range(i):
                        for k in range(i):
                            lop[p + k] = su2_factor[s] * \
                                OpElement(OpNames.Q, (j, k, s), q_label=self.hamil.two_site_minus_q[j, k][s])
                        p += i
                assert p == lop.shape[0]
            self.left_operator_names[i] = lop
            self.right_operator_names[i] = rop


class MPO(TensorNetwork):
    def __init__(self, hamil, iprint=False):
        self.n_sites = hamil.n_sites
        self.hamil = hamil
        tensors = self._init_mpo_tensors(iprint=iprint)
        super().__init__(tensors)
    
    def _init_mpo_tensors(self, iprint, symmetrized_p=True):
        """Generate :attr:`tensors`."""
        tensors = []
        op_h = OpElement(OpNames.H, ())
        op_i = OpElement(OpNames.I, ())
        op_c = OpElement(OpNames.C, ())
        op_d = OpElement(OpNames.D, ())
        for m in range(self.n_sites):
            if iprint:
                print("\r%3d%% " % ((m + 1) * 100 // self.n_sites), end='')
            lshape = 2 + 2 * self.n_sites + 6 * m * m if m != 0 else 1
            rshape = 2 + 2 * self.n_sites + 6 * (m + 1) * (m + 1) if m != self.n_sites - 1 else 1
            mat = np.zeros((lshape, rshape), dtype=object)
            if m == 0:
                mat[-1, 0] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
                mat[-1, 1] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                mat[-1, 2] = OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                mat[-1, 3] = OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])
                p = 4
                for j in range(m + 1, self.n_sites):
                    mat[-1, p + j - m - 1] = 2.0 * OpElement(OpNames.RD, (j, ), q_label=self.hamil.one_site_q[j])
                p += self.n_sites - (m + 1)
                for j in range(m + 1, self.n_sites):
                    mat[-1, p + j - m - 1] = 2.0 * OpElement(OpNames.R, (j, ), q_label=-self.hamil.one_site_q[j])
                p += self.n_sites - (m + 1)
                for s in [0, 1]:
                    mat[-1, p + s] = OpElement(OpNames.A, (m, m, s), q_label=self.hamil.two_site_plus_q[m, m][s])
                p += 2
                for s in [0, 1]:
                    mat[-1, p + s] = OpElement(OpNames.AD, (m, m, s), q_label=-self.hamil.two_site_plus_q[m, m][s])
                p += 2
                for s in [0, 1]:
                    mat[-1, p + s] = OpElement(OpNames.B, (m, m, s), q_label=self.hamil.two_site_minus_q[m, m][s])
                p += 2
                assert p == rshape
            else:
                mat[0, 0] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                mat[1, 0] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
                p = 2
                for j in range(m):
                    mat[p + j, 0] = 2.0 * OpElement(OpNames.R, (j, ), q_label=-self.hamil.one_site_q[j])
                p += m
                for j in range(m):
                    mat[p + j, 0] = 2.0 * OpElement(OpNames.RD, (j, ), q_label=self.hamil.one_site_q[j])
                p += m
                for j in range(m, self.n_sites):
                    if j == m:
                        mat[p + j - m, 0] = OpElement(OpNames.D, (j, ), q_label=-self.hamil.one_site_q[j])
                p += self.n_sites - m
                for j in range(m, self.n_sites):
                    if j == m:
                        mat[p + j - m, 0] = OpElement(OpNames.C, (j, ), q_label=self.hamil.one_site_q[j])
                p += self.n_sites - m
                su2_factor = [-0.5, -0.5 * np.sqrt(3.0)]
                for s in [0, 1]:
                    for j in range(m):
                        for k in range(m):
                            mat[p + k, 0] = su2_factor[s] * \
                                OpElement(OpNames.P, (j, k, s), q_label=-self.hamil.two_site_plus_q[j, k][s])
                        p += m
                for s in [0, 1]:
                    for j in range(m):
                        for k in range(m):
                            mat[p + k, 0] = su2_factor[s] * \
                                OpElement(OpNames.PD, (j, k, s), q_label=self.hamil.two_site_plus_q[j, k][s])
                        p += m
                su2_factor = [1.0, np.sqrt(3.0)]
                for s in [0, 1]:
                    for j in range(m):
                        for k in range(m):
                            mat[p + k, 0] = su2_factor[s] * \
                                OpElement(OpNames.Q, (j, k, s), q_label=self.hamil.two_site_minus_q[j, k][s])
                        p += m
                assert p == lshape
            if m != 0 and m != self.n_sites - 1:
                mat[1, 1] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                p = 2
                # pointers
                pi = 1
                pc = 2
                pd = 2 + m
                prd = 2 + m + m - m
                pr = 2 + m + self.n_sites - m
                pa0 = 2 + self.n_sites * 2
                pa1 = 2 + self.n_sites * 2 + m * m
                pad0 = 2 + self.n_sites * 2 + m * m * 2
                pad1 = 2 + self.n_sites * 2 + m * m * 3
                pb0 = 2 + self.n_sites * 2 + m * m * 4
                pb1 = 2 + self.n_sites * 2 + m * m * 5
                # C
                for j in range(m):
                    mat[pc + j, p + j] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                mat[pi, p + m] = OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                p += m + 1
                # D
                for j in range(m):
                    mat[pd + j, p + j] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                mat[pi, p + m] = OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])
                p += m + 1
                # RD
                for i in range(m + 1, self.n_sites):
                    mat[prd + i, p + i - (m + 1)] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                    mat[pi, p + i - (m + 1)] = 2.0 * OpElement(OpNames.RD, (i, ), q_label=self.hamil.one_site_q[i])
                    for k in range(0, m):
                        mat[pd + k, p + i - (m + 1)] = 2.0 * (
                            (-0.5) * OpElement(OpNames.PD, (i, k, 0), q_label=self.hamil.two_site_plus_q[i, k][0]) \
                            + (-0.5 * np.sqrt(3)) * OpElement(OpNames.PD, (i, k, 1), q_label=self.hamil.two_site_plus_q[i, k][1])
                        )
                        mat[pc + k, p + i - (m + 1)] = 2.0 * (
                            0.5 * OpElement(OpNames.Q, (k, i, 0), q_label=self.hamil.two_site_minus_q[k, i][0]) \
                            + (-0.5 * np.sqrt(3)) * OpElement(OpNames.Q, (k, i, 1), q_label=self.hamil.two_site_minus_q[k, i][1])
                        )
                    for j in range(0, m):
                        for l in range(0, m):
                            if not symmetrized_p:
                                f0 = f1 = 2.0 * self.hamil.v[i, j, m, l]
                            else:
                                f0 = self.hamil.v[i, j, m, l] + self.hamil.v[i, l, m, j]
                                f1 = self.hamil.v[i, j, m, l] - self.hamil.v[i, l, m, j]
                            mat[pa0 + j * m + l, p + i - (m + 1)] = f0 * (-0.5) * \
                                OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])
                            mat[pa1 + j * m + l, p + i - (m + 1)] = f1 * (0.5 * np.sqrt(3)) * \
                                OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])
                    for k in range(0, m):
                        for l in range(0, m):
                            f = 2.0 * (2 * self.hamil.v[i, m, k, l] - self.hamil.v[i, l, k, m]) * 0.5
                            mat[pb0 + l * m + k, p + i - (m + 1)] = f * \
                                OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                    for j in range(0, m):
                        for k in range(0, m):
                            f = 2.0 * self.hamil.v[i, j, k, m] * np.sqrt(3) * 0.5
                            mat[pb1 + j * m + k, p + i - (m + 1)] = f * \
                                OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                p += self.n_sites - (m + 1)
                # R
                for i in range(m + 1, self.n_sites):
                    mat[pr + i, p + i - (m + 1)] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                    mat[pi, p + i - (m + 1)] = 2.0 * OpElement(OpNames.R, (i, ), q_label=-self.hamil.one_site_q[i])
                    for k in range(0, m):
                        mat[pc + k, p + i - (m + 1)] = 2.0 * (
                            (-0.5) * OpElement(OpNames.P, (i, k, 0), q_label=-self.hamil.two_site_plus_q[i, k][0]) \
                            + (0.5 * np.sqrt(3)) * OpElement(OpNames.P, (i, k, 1), q_label=-self.hamil.two_site_plus_q[i, k][1])
                        )
                        mat[pd + k, p + i - (m + 1)] = 2.0 * (
                            0.5 * OpElement(OpNames.Q, (i, k, 0), q_label=self.hamil.two_site_minus_q[i, k][0]) \
                            + (0.5 * np.sqrt(3)) * OpElement(OpNames.Q, (i, k, 1), q_label=self.hamil.two_site_minus_q[i, k][1])
                        )
                    for j in range(0, m):
                        for l in range(0, m):
                            if not symmetrized_p:
                                f0 = f1 = 2.0 * self.hamil.v[i, j, m, l]
                            else:
                                f0 = self.hamil.v[i, j, m, l] + self.hamil.v[i, l, m, j]
                                f1 = self.hamil.v[i, j, m, l] - self.hamil.v[i, l, m, j]
                            mat[pad0 + j * m + l, p + i - (m + 1)] = f0 * (-0.5) * \
                                OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                            mat[pad1 + j * m + l, p + i - (m + 1)] = f1 * (-0.5 * np.sqrt(3)) * \
                                OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                    for k in range(0, m):
                        for l in range(0, m):
                            f = 2.0 * (2 * self.hamil.v[i, m, k, l] - self.hamil.v[i, l, k, m]) * 0.5
                            mat[pb0 + k * m + l, p + i - (m + 1)] = f * \
                                OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])
                    for j in range(0, m):
                        for k in range(0, m):
                            f = 2.0 * self.hamil.v[i, j, k, m] * np.sqrt(3) * (-0.5)
                            mat[pb1 + k * m + j, p + i - (m + 1)] = f * \
                                OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])
                p += self.n_sites - (m + 1)
                # A
                for s in [0, 1]:
                    pa = [pa0, pa1][s]
                    for i in range(m):
                        for j in range(m):
                            mat[pa + i * m + j, p + i * (m + 1) + j] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                    for i in range(m):
                        f = [1.0, -1.0][s]
                        mat[pc + i, p + i * (m + 1) + m] = OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                        mat[pc + i, p + m * (m + 1) + i] = f * OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                    mat[pi, p + m * (m + 1) + m] = OpElement(OpNames.A, (m, m, s), q_label=self.hamil.two_site_plus_q[m, m][s])
                    p += (m + 1) * (m + 1)
                # AD
                for s in [0, 1]:
                    pad = [pad0, pad1][s]
                    for i in range(m):
                        for j in range(m):
                            mat[pad + i * m + j, p + i * (m + 1) + j] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                    for i in range(m):
                        f = [1.0, -1.0][s]
                        mat[pd + i, p + i * (m + 1) + m] = f * OpElement(OpNames.D, (m, ), q_label=self.hamil.one_site_q[m])
                        mat[pd + i, p + m * (m + 1) + i] = OpElement(OpNames.D, (m, ), q_label=self.hamil.one_site_q[m])
                    mat[pi, p + m * (m + 1) + m] = OpElement(OpNames.AD, (m, m, s), q_label=-self.hamil.two_site_plus_q[m, m][s])
                    p += (m + 1) * (m + 1)
                # B
                for s in [0, 1]:
                    pb = [pb0, pb1][s]
                    for i in range(m):
                        for j in range(m):
                            mat[pb + i * m + j, p + i * (m + 1) + j] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                    for i in range(m):
                        f = [1.0, -1.0][s]
                        mat[pc + i, p + i * (m + 1) + m] = OpElement(OpNames.D, (m, ), q_label=self.hamil.one_site_q[m])
                        mat[pd + i, p + m * (m + 1) + i] = f * OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                    mat[pi, p + m * (m + 1) + m] = OpElement(OpNames.B, (m, m, s), q_label=self.hamil.two_site_minus_q[m, m][s])
                    p += (m + 1) * (m + 1)
                assert p == rshape
            
            mat, ops = self._post_check_mpo_operators(mat, m)
            
            tensors.append(OperatorTensor(mat=mat, tags={m}, ops=ops))
            
        return tensors
    
    def _post_check_mpo_operators(self, mat, m):
        
        ops_set = set()
        
        for em in mat.reshape(mat.size):
            if em == 0:
                pass
            elif isinstance(em, OpElement):
                ops_set.add(abs(em))
            elif isinstance(em, OpSum):
                ops_set |= { abs(opd.op) for opd in em.strings }
            else:
                assert False
        
        ops = self.hamil.get_site_operators(m, ops_set)
        
        for il in range(mat.shape[0]):
            for ir in range(mat.shape[1]):
                em = mat[il, ir]
                if em == 0:
                    pass
                elif isinstance(em, OpElement):
                    if ops[abs(em)] == 0:
                        mat[il, ir] = 0
                elif isinstance(em, OpSum):
                    if all(ops[abs(opd.op)] == 0 for opd in em.strings):
                        mat[il, ir] = 0
        
        ops = { k : v for k, v in ops.items() if v != 0 }
        
        return mat, ops
