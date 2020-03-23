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


class DualOperatorTensor(Tensor):
    """
    MPO tensor or contracted MPO tensor with dual (left and right) representation.
    """
    def __init__(self, lmat=None, rmat=None, ops=None, tags=None, contractor=None):
        self.lmat = lmat
        self.rmat = rmat
        self.ops = ops
        super().__init__([], tags=tags, contractor=contractor)
    
    def __repr__(self):
        return repr(self.lmat) + "\n" + repr(self.rmat) + "\n" + "\n".join([repr(k) + " :: \n" + repr(v) for k, v in self.ops.items()])

    def copy(self):
        """Return shallow copy of this object."""
        assert isinstance(self.ops, dict)
        lmat = self.lmat.copy() if self.lmat is not None else None
        rmat = self.rmat.copy() if self.rmat is not None else None
        return DualOperatorTensor(lmat=lmat, rmat=rmat, ops=self.ops.copy(),
            tags=self.tags.copy(), contractor=self.contractor)


class MPOInfo:
    def __init__(self, hamil, cache_contraction=True):
        self.hamil = hamil
        self.n_sites = hamil.n_sites
        self.middle_operators = None
        self._init_operator_names()
        self.cache_contraction = cache_contraction
        self.cached_exprs = {}
    
    def _init_operator_names(self):
        if self.hamil.spin_adapted:
            self._init_operator_names_su2()
        else:
            self._init_operator_names_sz()

    def _init_operator_names_su2(self):
        self.left_operator_names = [None] * self.n_sites
        self.right_operator_names = [None] * self.n_sites

        for i in range(self.n_sites):
            lshape = 2 + 2 * self.n_sites + 6 * (i + 1) * (i + 1) if i != self.n_sites - 1 else 1
            rshape = 2 + 2 * self.n_sites + 6 * i * i if i != 0 else 1
            lop = np.zeros((lshape, ), dtype=object)
            rop = np.zeros((rshape, ), dtype=object)
            lop[0] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
            if i != self.n_sites - 1:
                lop[1] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                p = 2
                for j in range(i + 1):
                    lop[p + j] = OpElement(OpNames.C, (j, ), q_label=self.hamil.one_site_q[j])
                p += i + 1
                for j in range(i + 1):
                    lop[p + j] = OpElement(OpNames.D, (j, ), q_label=-self.hamil.one_site_q[j])
                p += i + 1
                for j in range(i + 1, self.n_sites):
                    lop[p + j - i - 1] = 2.0 * OpElement(OpNames.RD, (j, ), q_label=self.hamil.one_site_q[j])
                p += self.n_sites - (i + 1)
                for j in range(i + 1, self.n_sites):
                    lop[p + j - i - 1] = 2.0 * OpElement(OpNames.R, (j, ), q_label=-self.hamil.one_site_q[j])
                p += self.n_sites - (i + 1)
                for s in [0, 1]:
                    for j in range(i + 1):
                        for k in range(i + 1):
                            lop[p + k] = OpElement(OpNames.A, (j, k, s), q_label=self.hamil.two_site_plus_q[j, k][s])
                        p += i + 1
                for s in [0, 1]:
                    for j in range(i + 1):
                        for k in range(i + 1):
                            lop[p + k] = OpElement(OpNames.AD, (j, k, s), q_label=-self.hamil.two_site_plus_q[j, k][s])
                        p += i + 1
                for s in [0, 1]:
                    for j in range(i + 1):
                        for k in range(i + 1):
                            lop[p + k] = OpElement(OpNames.B, (j, k, s), q_label=self.hamil.two_site_minus_q[j, k][s])
                        p += i + 1
                assert p == lop.shape[0]
            rop[0] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
            if i != 0:
                rop[1] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
                p = 2
                for j in range(i):
                    rop[p + j] = 2.0 * OpElement(OpNames.R, (j, ), q_label=-self.hamil.one_site_q[j])
                p += i
                for j in range(i):
                    rop[p + j] = 2.0 * OpElement(OpNames.RD, (j, ), q_label=self.hamil.one_site_q[j])
                p += i
                for j in range(i, self.n_sites):
                    rop[p + j - i] = OpElement(OpNames.D, (j, ), q_label=-self.hamil.one_site_q[j])
                p += self.n_sites - i
                for j in range(i, self.n_sites):
                    rop[p + j - i] = OpElement(OpNames.C, (j, ), q_label=self.hamil.one_site_q[j])
                p += self.n_sites - i
                su2_factor = [-0.5, -0.5 * np.sqrt(3.0)]
                for s in [0, 1]:
                    for j in range(i):
                        for k in range(i):
                            rop[p + k] = su2_factor[s] * \
                                OpElement(OpNames.P, (j, k, s), q_label=-self.hamil.two_site_plus_q[j, k][s])
                        p += i
                for s in [0, 1]:
                    for j in range(i):
                        for k in range(i):
                            rop[p + k] = su2_factor[s] * \
                                OpElement(OpNames.PD, (j, k, s), q_label=self.hamil.two_site_plus_q[j, k][s])
                        p += i
                su2_factor = [1.0, np.sqrt(3.0)]
                for s in [0, 1]:
                    for j in range(i):
                        for k in range(i):
                            rop[p + k] = su2_factor[s] * \
                                OpElement(OpNames.Q, (j, k, s), q_label=self.hamil.two_site_minus_q[j, k][s])
                        p += i
                assert p == rop.shape[0]
            self.left_operator_names[i] = lop
            self.right_operator_names[i] = rop

    def _init_operator_names_sz(self):
        self.left_operator_names = [None] * self.n_sites
        self.right_operator_names = [None] * self.n_sites

        qs = [self.hamil.one_site_qa, self.hamil.one_site_qb]
        pqs = [self.hamil.two_site_plus_qaa, self.hamil.two_site_plus_qab, self.hamil.two_site_plus_qba, self.hamil.two_site_plus_qbb]
        mqs = [self.hamil.two_site_minus_qaa, self.hamil.two_site_minus_qab, self.hamil.two_site_minus_qba, self.hamil.two_site_minus_qbb]
        ss = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for i in range(self.n_sites):
            lshape = 2 + 4 * self.n_sites + 12 * (i + 1) * (i + 1) if i != self.n_sites - 1 else 1
            rshape = 2 + 4 * self.n_sites + 12 * i * i if i != 0 else 1
            lop = np.zeros((lshape, ), dtype=object)
            rop = np.zeros((rshape, ), dtype=object)
            lop[0] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
            if i != self.n_sites - 1:
                lop[1] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                p = 2
                for s, q in zip([0, 1], qs):
                    for j in range(i + 1):
                        lop[p + j] = OpElement(OpNames.C, (j, s), q_label=q[j])
                    p += i + 1
                for s, q in zip([0, 1], qs):
                    for j in range(i + 1):
                        lop[p + j] = OpElement(OpNames.D, (j, s), q_label=-q[j])
                    p += i + 1
                for s, q in zip([0, 1], qs):
                    for j in range(i + 1, self.n_sites):
                        lop[p + j - i - 1] = OpElement(OpNames.RD, (j, s), q_label=q[j])
                    p += self.n_sites - (i + 1)
                for s, q in zip([0, 1], qs):
                    for j in range(i + 1, self.n_sites):
                        lop[p + j - i - 1] = -1.0 * OpElement(OpNames.R, (j, s), q_label=-q[j])
                    p += self.n_sites - (i + 1)
                for (sl, sr), pq in zip(ss, pqs):
                    for j in range(i + 1):
                        for k in range(i + 1):
                            lop[p + k] = OpElement(OpNames.A, (j, k, sl, sr), q_label=pq[j, k])
                        p += i + 1
                for (sl, sr), pq in zip(ss, pqs):
                    for j in range(i + 1):
                        for k in range(i + 1):
                            lop[p + k] = OpElement(OpNames.AD, (j, k, sl, sr), q_label=-pq[j, k])
                        p += i + 1
                for (sl, sr), mq in zip(ss, mqs):
                    for j in range(i + 1):
                        for k in range(i + 1):
                            lop[p + k] = OpElement(OpNames.B, (j, k, sl, sr), q_label=mq[j, k])
                        p += i + 1
                assert p == lop.shape[0]
            rop[0] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
            if i != 0:
                rop[1] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
                p = 2
                for s, q in zip([0, 1], qs):
                    for j in range(i):
                        rop[p + j] = OpElement(OpNames.R, (j, s), q_label=-q[j])
                    p += i
                for s, q in zip([0, 1], qs):
                    for j in range(i):
                        rop[p + j] = -1.0 * OpElement(OpNames.RD, (j, s), q_label=q[j])
                    p += i
                for s, q in zip([0, 1], qs):
                    for j in range(i, self.n_sites):
                        rop[p + j - i] = OpElement(OpNames.D, (j, s), q_label=-q[j])
                    p += self.n_sites - i
                for s, q in zip([0, 1], qs):
                    for j in range(i, self.n_sites):
                        rop[p + j - i] = OpElement(OpNames.C, (j, s), q_label=q[j])
                    p += self.n_sites - i
                for (sl, sr), pq in zip(ss, pqs):
                    for j in range(i):
                        for k in range(i):
                            rop[p + k] = 0.5 * OpElement(OpNames.P, (j, k, sl, sr), q_label=-pq[j, k])
                        p += i
                for (sl, sr), pq in zip(ss, pqs):
                    for j in range(i):
                        for k in range(i):
                            rop[p + k] = 0.5 * OpElement(OpNames.PD, (j, k, sl, sr), q_label=pq[j, k])
                        p += i
                for (sl, sr), mq in zip(ss, mqs):
                    for j in range(i):
                        for k in range(i):
                            rop[p + k] = OpElement(OpNames.Q, (j, k, sl, sr), q_label=-mq[j, k])
                        p += i
                assert p == rop.shape[0]
            self.left_operator_names[i] = lop
            self.right_operator_names[i] = rop


class MPO(TensorNetwork):
    def __init__(self, hamil, iprint=False):
        self.n_sites = hamil.n_sites
        self.hamil = hamil
        tensors = self._init_mpo_tensors(iprint=iprint)
        super().__init__(tensors)
    
    def _init_mpo_tensors(self, *args, **kwargs):
        """Generate :attr:`tensors`."""
        if self.hamil.spin_adapted:
            return self._init_mpo_tensors_su2(*args, **kwargs)
        else:
            return self._init_mpo_tensors_sz(*args, **kwargs)
    
    def _init_mpo_tensors_su2(self, iprint, symmetrized_p=True):
        tensors = []
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
                        mat[pd + i, p + i * (m + 1) + m] = f * OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])
                        mat[pd + i, p + m * (m + 1) + i] = OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])
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
                        mat[pc + i, p + i * (m + 1) + m] = OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])
                        mat[pd + i, p + m * (m + 1) + i] = f * OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                    mat[pi, p + m * (m + 1) + m] = OpElement(OpNames.B, (m, m, s), q_label=self.hamil.two_site_minus_q[m, m][s])
                    p += (m + 1) * (m + 1)
                assert p == rshape
            
            [mat], ops = self._post_check_mpo_operators([mat], m)
            
            tensors.append(OperatorTensor(mat=mat, tags={m}, ops=ops))
            
        return tensors

    def _init_mpo_tensors_sz(self, iprint, symmetrized_p=True):
        tensors = []

        qs = [self.hamil.one_site_qa, self.hamil.one_site_qb]
        pqs = [self.hamil.two_site_plus_qaa, self.hamil.two_site_plus_qab, self.hamil.two_site_plus_qba, self.hamil.two_site_plus_qbb]
        mqs = [self.hamil.two_site_minus_qaa, self.hamil.two_site_minus_qab, self.hamil.two_site_minus_qba, self.hamil.two_site_minus_qbb]
        apqs = np.array(pqs, dtype=object).reshape((2, 2, self.n_sites, self.n_sites))
        amqs = np.array(mqs, dtype=object).reshape((2, 2, self.n_sites, self.n_sites))
        avs = np.array([self.hamil.vaa, self.hamil.vab, self.hamil.vba, self.hamil.vbb], dtype=object).reshape((2, 2))
        ss = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for m in range(self.n_sites):
            if iprint:
                print("\r%3d%% " % ((m + 1) * 100 // self.n_sites), end='')
            lshape = 2 + 4 * self.n_sites + 12 * m * m if m != 0 else 1
            rshape = 2 + 4 * self.n_sites + 12 * (m + 1) * (m + 1) if m != self.n_sites - 1 else 1
            mat = np.zeros((lshape, rshape), dtype=object)
            if m == 0:
                mat[-1, 0] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
                mat[-1, 1] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                mat[-1, 2] = OpElement(OpNames.C, (m, 0), q_label=self.hamil.one_site_qa[m])
                mat[-1, 3] = OpElement(OpNames.C, (m, 1), q_label=self.hamil.one_site_qb[m])
                mat[-1, 4] = OpElement(OpNames.D, (m, 0), q_label=-self.hamil.one_site_qa[m])
                mat[-1, 5] = OpElement(OpNames.D, (m, 1), q_label=-self.hamil.one_site_qb[m])
                p = 6
                for s, q in zip([0, 1], qs):
                    for j in range(m + 1, self.n_sites):
                        mat[-1, p + j - m - 1] = OpElement(OpNames.RD, (j, s), q_label=q[j])
                    p += self.n_sites - (m + 1)
                for s, q in zip([0, 1], qs):
                    for j in range(m + 1, self.n_sites):
                        mat[-1, p + j - m - 1] = -1.0 * OpElement(OpNames.R, (j, s), q_label=-q[j])
                    p += self.n_sites - (m + 1)
                for (sl, sr), pq in zip(ss, pqs):
                    mat[-1, p] = OpElement(OpNames.A, (m, m, sl, sr), q_label=pq[m, m])
                    p += 1
                for (sl, sr), pq in zip(ss, pqs):
                    mat[-1, p] = OpElement(OpNames.AD, (m, m, sl, sr), q_label=-pq[m, m])
                    p += 1
                for (sl, sr), mq in zip(ss, mqs):
                    mat[-1, p] = OpElement(OpNames.B, (m, m, sl, sr), q_label=mq[m, m])
                    p += 1
                assert p == rshape
            else:
                mat[0, 0] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                mat[1, 0] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
                p = 2
                for s, q in zip([0, 1], qs):
                    for j in range(m):
                        mat[p + j, 0] = OpElement(OpNames.R, (j, s), q_label=-q[j])
                    p += m
                for s, q in zip([0, 1], qs):
                    for j in range(m):
                        mat[p + j, 0] = -1.0 * OpElement(OpNames.RD, (j, s), q_label=q[j])
                    p += m
                for s, q in zip([0, 1], qs):
                    for j in range(m, self.n_sites):
                        if j == m:
                            mat[p + j - m, 0] = OpElement(OpNames.D, (j, s), q_label=-q[j])
                    p += self.n_sites - m
                for s, q in zip([0, 1], qs):
                    for j in range(m, self.n_sites):
                        if j == m:
                            mat[p + j - m, 0] = OpElement(OpNames.C, (j, s), q_label=q[j])
                    p += self.n_sites - m
                for (sl, sr), pq in zip(ss, pqs):
                    for j in range(m):
                        for k in range(m):
                            mat[p + k, 0] = 0.5 * OpElement(OpNames.P, (j, k, sl, sr), q_label=-pq[j, k])
                        p += m
                for (sl, sr), pq in zip(ss, pqs):
                    for j in range(m):
                        for k in range(m):
                            mat[p + k, 0] = 0.5 * OpElement(OpNames.PD, (j, k, sl, sr), q_label=pq[j, k])
                        p += m
                for (sl, sr), mq in zip(ss, mqs):
                    for j in range(m):
                        for k in range(m):
                            mat[p + k, 0] = OpElement(OpNames.Q, (j, k, sl, sr), q_label=-mq[j, k])
                        p += m
                assert p == lshape
            if m != 0 and m != self.n_sites - 1:
                mat[1, 1] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                p = 2
                # pointers
                pi = 1
                pca = 2
                pcb = 2 + m
                pda = 2 + m * 2
                pdb = 2 + m * 3
                prda = 2 + m * 4 - m
                prdb = 2 + m * 3 + self.n_sites - m
                pra = 2 + m * 2 + self.n_sites * 2 - m
                prb = 2 + m + self.n_sites * 3 - m
                paaa = 2 + self.n_sites * 4
                paab = 2 + self.n_sites * 4 + m * m
                paba = 2 + self.n_sites * 4 + m * m * 2
                pabb = 2 + self.n_sites * 4 + m * m * 3
                padaa = 2 + self.n_sites * 4 + m * m * 4
                padab = 2 + self.n_sites * 4 + m * m * 5
                padba = 2 + self.n_sites * 4 + m * m * 6
                padbb = 2 + self.n_sites * 4 + m * m * 7
                pbaa = 2 + self.n_sites * 4 + m * m * 8
                pbab = 2 + self.n_sites * 4 + m * m * 9
                pbba = 2 + self.n_sites * 4 + m * m * 10
                pbbb = 2 + self.n_sites * 4 + m * m * 11
                # C
                for s, q, pc in zip([0, 1], qs, [pca, pcb]):
                    for j in range(m):
                        mat[pc + j, p + j] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                    mat[pi, p + m] = OpElement(OpNames.C, (m, s), q_label=q[m])
                    p += m + 1
                # D
                for s, q, pd in zip([0, 1], qs, [pda, pdb]):
                    for j in range(m):
                        mat[pd + j, p + j] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                    mat[pi, p + m] = OpElement(OpNames.D, (m, s), q_label=-q[m])
                    p += m + 1
                # RD
                for s, q, prd in zip([0, 1], qs, [prda, prdb]):
                    for i in range(m + 1, self.n_sites):
                        mat[prd + i, p + i - (m + 1)] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                        mat[pi, p + i - (m + 1)] = OpElement(OpNames.RD, (i, s), q_label=q[i])
                        for sp, pd, pc in zip([0, 1], [pda, pdb], [pca, pcb]):
                            for k in range(0, m):
                                mat[pd + k, p + i - (m + 1)] = OpElement(OpNames.PD, (i, k, s, sp), q_label=apqs[s, sp][i, k])
                                mat[pc + k, p + i - (m + 1)] = OpElement(OpNames.Q, (k, i, sp, s), q_label=-amqs[sp, s][k, i])
                        if not symmetrized_p:
                            for sp, qp, paxx in zip([0, 1], qs, [[paaa, paab], [paba, pabb]][s]):
                                for j in range(0, m):
                                    for l in range(0, m):
                                        f = avs[s, sp][i, j, m, l]
                                        mat[paxx + j * m + l, p + i - (m + 1)] = f * OpElement(OpNames.D, (m, sp), q_label=-qp[m])
                        else:
                            pas = np.array([[paaa, paab], [paba, pabb]], dtype=object)
                            for sp, qp in zip([0, 1], qs):
                                for j in range(0, m):
                                    for l in range(0, m):
                                        f0 = 0.5 * avs[s, sp][i, j, m, l]
                                        f1 = -0.5 * avs[s, sp][i, l, m, j]
                                        mat[pas[s, sp] + j * m + l, p + i - (m + 1)] += f0 * OpElement(OpNames.D, (m, sp), q_label=-qp[m])
                                        mat[pas[sp, s] + j * m + l, p + i - (m + 1)] += f1 * OpElement(OpNames.D, (m, sp), q_label=-qp[m])
                        for sp, pbxx in zip([0, 1], [pbaa, pbbb]):
                            for k in range(0, m):
                                for l in range(0, m):
                                    f = avs[s, sp][i, m, k, l]
                                    mat[pbxx + l * m + k, p + i - (m + 1)] = f * OpElement(OpNames.C, (m, s), q_label=q[m])
                        for sp, qp, pbxx in zip([0, 1], qs, [[pbaa, pbab], [pbba, pbbb]][s]):
                            for j in range(0, m):
                                for k in range(0, m):
                                    f = -1.0 * avs[s, sp][i, j, k, m]
                                    mat[pbxx + j * m + k, p + i - (m + 1)] += f * OpElement(OpNames.C, (m, sp), q_label=qp[m])
                    p += self.n_sites - (m + 1)
                # R
                for s, q, pr in zip([0, 1], qs, [pra, prb]):
                    for i in range(m + 1, self.n_sites):
                        mat[pr + i, p + i - (m + 1)] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                        mat[pi, p + i - (m + 1)] = -1.0 * OpElement(OpNames.R, (i, s), q_label=-q[i])
                        for sp, pd, pc in zip([0, 1], [pda, pdb], [pca, pcb]):
                            for k in range(0, m):
                                mat[pc + k, p + i - (m + 1)] = -1.0 * OpElement(OpNames.P, (i, k, s, sp), q_label=-apqs[s, sp][i, k])
                                mat[pd + k, p + i - (m + 1)] = -1.0 * OpElement(OpNames.Q, (i, k, s, sp), q_label=-amqs[s, sp][i, k])
                        if not symmetrized_p:
                            for sp, qp, padxx in zip([0, 1], qs, [[padaa, padab], [padba, padbb]][s]):
                                for j in range(0, m):
                                    for l in range(0, m):
                                        f = -1.0 * avs[s, sp][i, j, m, l]
                                        mat[padxx + j * m + l, p + i - (m + 1)] = f * OpElement(OpNames.C, (m, sp), q_label=qp[m])
                        else:
                            pads = np.array([[padaa, padab], [padba, padbb]], dtype=object)
                            for sp, qp in zip([0, 1], qs):
                                for j in range(0, m):
                                    for l in range(0, m):
                                        f0 = -0.5 * avs[s, sp][i, j, m, l]
                                        f1 = 0.5 * avs[s, sp][i, l, m, j]
                                        mat[pads[s, sp] + j * m + l, p + i - (m + 1)] += f0 * OpElement(OpNames.C, (m, sp), q_label=qp[m])
                                        mat[pads[sp, s] + j * m + l, p + i - (m + 1)] += f1 * OpElement(OpNames.C, (m, sp), q_label=qp[m])
                        for sp, pbxx in zip([0, 1], [pbaa, pbbb]):
                            for k in range(0, m):
                                for l in range(0, m):
                                    f = -1.0 * avs[s, sp][i, m, k, l]
                                    mat[pbxx + k * m + l, p + i - (m + 1)] = f * OpElement(OpNames.D, (m, s), q_label=-q[m])
                        for sp, qp, pbxx in zip([0, 1], qs, [[pbaa, pbba], [pbab, pbbb]][s]):
                            for j in range(0, m):
                                for k in range(0, m):
                                    f = (-1.0) * (-1.0) * avs[s, sp][i, j, k, m]
                                    mat[pbxx + k * m + j, p + i - (m + 1)] += f * OpElement(OpNames.D, (m, sp), q_label=-qp[m])
                    p += self.n_sites - (m + 1)
                # A
                for (sl, sr), pq, paxx in zip(ss, pqs, [paaa, paab, paba, pabb]):
                    for i in range(m):
                        for j in range(m):
                            mat[paxx + i * m + j, p + i * (m + 1) + j] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                    for i in range(m):
                        mat[[pca, pcb][sl] + i, p + i * (m + 1) + m] = OpElement(OpNames.C, (m, sr), q_label=qs[sr][m])
                        mat[[pca, pcb][sr] + i, p + m * (m + 1) + i] = -1.0 * OpElement(OpNames.C, (m, sl), q_label=qs[sl][m])
                    mat[pi, p + m * (m + 1) + m] = OpElement(OpNames.A, (m, m, sl, sr), q_label=pq[m, m])
                    p += (m + 1) * (m + 1)
                # AD
                for (sl, sr), pq, padxx in zip(ss, pqs, [padaa, padab, padba, padbb]):
                    for i in range(m):
                        for j in range(m):
                            mat[padxx + i * m + j, p + i * (m + 1) + j] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                    for i in range(m):
                        mat[[pda, pdb][sl] + i, p + i * (m + 1) + m] = -1.0 * OpElement(OpNames.D, (m, sr), q_label=-qs[sr][m])
                        mat[[pda, pdb][sr] + i, p + m * (m + 1) + i] = OpElement(OpNames.D, (m, sl), q_label=-qs[sl][m])
                    mat[pi, p + m * (m + 1) + m] = OpElement(OpNames.AD, (m, m, sl, sr), q_label=-pq[m, m])
                    p += (m + 1) * (m + 1)
                # B
                for (sl, sr), mq, pbxx in zip(ss, mqs, [pbaa, pbab, pbba, pbbb]):
                    for i in range(m):
                        for j in range(m):
                            mat[pbxx + i * m + j, p + i * (m + 1) + j] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                    for i in range(m):
                        mat[[pca, pcb][sl] + i, p + i * (m + 1) + m] = OpElement(OpNames.D, (m, sr), q_label=-qs[sr][m])
                        mat[[pda, pdb][sr] + i, p + m * (m + 1) + i] = -1.0 * OpElement(OpNames.C, (m, sl), q_label=qs[sl][m])
                    mat[pi, p + m * (m + 1) + m] = OpElement(OpNames.B, (m, m, sl, sr), q_label=mq[m, m])
                    p += (m + 1) * (m + 1)
                assert p == rshape
            
            [mat], ops = self._post_check_mpo_operators([mat], m)
            
            tensors.append(OperatorTensor(mat=mat, tags={m}, ops=ops))

        return tensors

    def _post_check_mpo_operators(self, mats, m):
        
        ops_set = set()
        
        for mat in mats:
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
        
        for mat in mats:
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
        
        return mats, ops


class SquareMPOInfo(MPOInfo):
    def __init__(self, hamil, op_name, opsq_name, site_index=(), **kwargs):
        self.op = OpElement(op_name, site_index, q_label=hamil.empty)
        self.opsq = OpElement(opsq_name, site_index, q_label=hamil.empty)
        super().__init__(hamil, **kwargs)
    
    def _init_operator_names(self):
        self.left_operator_names = [None] * self.n_sites
        self.right_operator_names = [None] * self.n_sites
        iop = OpElement(OpNames.I, (), q_label=self.hamil.empty)
        for i in range(self.n_sites):
            if i == self.n_sites - 1:
                self.left_operator_names[i] = np.array([iop], dtype=object)
            else:
                self.left_operator_names[i] = np.array([self.opsq, self.op, iop], dtype=object)
            if i == 0:
                self.right_operator_names[i] = np.array([iop], dtype=object)
            else:
                self.right_operator_names[i] = np.array([iop, 2.0 * self.op, self.opsq], dtype=object)


class SquareMPO(MPO):
    def __init__(self, hamil, op_name, opsq_name, site_index=(), **kwargs):
        self.op = OpElement(op_name, site_index, q_label=hamil.empty)
        self.opsq = OpElement(opsq_name, site_index, q_label=hamil.empty)
        super().__init__(hamil, **kwargs)

    def _init_mpo_tensors(self, iprint):
        tensors = []
        iop = OpElement(OpNames.I, (), q_label=self.hamil.empty)
        for m in range(self.n_sites):
            if m == 0:
                mat = np.array([[self.opsq, self.op, iop]], dtype=object)
            elif m == self.n_sites - 1:
                mat = np.array([[iop], [2.0 * self.op], [self.opsq]], dtype=object)
            else:
                mat = np.array([[iop, 0, 0], [2.0 * self.op, iop, 0], [self.opsq, self.op, iop]], dtype=object)
            ops = self.hamil.get_site_operators(m, { iop, self.op, self.opsq })
            tensors.append(OperatorTensor(mat=mat, tags={m}, ops=ops))
        return tensors


class ProdMPOInfo(MPOInfo):
    def __init__(self, hamil, opa_name, opb_name, opab_name, site_index_a=(), site_index_b=(), site_index_ab=(), **kwargs):
        self.opa = OpElement(opa_name, site_index_a, q_label=hamil.empty)
        self.opb = OpElement(opb_name, site_index_b, q_label=hamil.empty)
        self.opab = OpElement(opab_name, site_index_ab, q_label=hamil.empty)
        super().__init__(hamil, **kwargs)
    
    def _init_operator_names(self):
        self.left_operator_names = [None] * self.n_sites
        self.right_operator_names = [None] * self.n_sites
        iop = OpElement(OpNames.I, (), q_label=self.hamil.empty)
        for i in range(self.n_sites):
            if i == self.n_sites - 1:
                self.left_operator_names[i] = np.array([iop], dtype=object)
            else:
                self.left_operator_names[i] = np.array([self.opab, self.opa, self.opb, iop], dtype=object)
            if i == 0:
                self.right_operator_names[i] = np.array([iop], dtype=object)
            else:
                self.right_operator_names[i] = np.array([iop, self.opb, self.opa, self.opab], dtype=object)


class ProdMPO(MPO):
    def __init__(self, hamil, opa_name, opb_name, opab_name, site_index_a=(), site_index_b=(), site_index_ab=(), **kwargs):
        self.opa = OpElement(opa_name, site_index_a, q_label=hamil.empty)
        self.opb = OpElement(opb_name, site_index_b, q_label=hamil.empty)
        self.opab = OpElement(opab_name, site_index_ab, q_label=hamil.empty)
        super().__init__(hamil, **kwargs)

    def _init_mpo_tensors(self, iprint):
        tensors = []
        iop = OpElement(OpNames.I, (), q_label=self.hamil.empty)
        for m in range(self.n_sites):
            if m == 0:
                mat = np.array([[self.opab, self.opa, self.opb, iop]], dtype=object)
            elif m == self.n_sites - 1:
                mat = np.array([[iop], [self.opb], [self.opa], [self.opab]], dtype=object)
            else:
                mat = np.array([
                    [iop, 0, 0, 0],
                    [self.opb, iop, 0, 0],
                    [self.opb, 0, iop, 0],
                    [self.opab, self.opa, self.opb, iop]
                ], dtype=object)
            ops = self.hamil.get_site_operators(m, { iop, self.opa, self.opb, self.opab })
            tensors.append(OperatorTensor(mat=mat, tags={m}, ops=ops))
        return tensors


class LocalMPOInfo(MPOInfo):
    def __init__(self, hamil, op_name, site_index=(), **kwargs):
        self.op = OpElement(op_name, site_index, q_label=hamil.empty)
        super().__init__(hamil, **kwargs)
    
    def _init_operator_names(self):
        self.left_operator_names = [None] * self.n_sites
        self.right_operator_names = [None] * self.n_sites
        iop = OpElement(OpNames.I, (), q_label=self.hamil.empty)
        for i in range(self.n_sites):
            if i == self.n_sites - 1:
                self.left_operator_names[i] = np.array([iop], dtype=object)
            else:
                self.left_operator_names[i] = np.array([self.op, iop], dtype=object)
            if i == 0:
                self.right_operator_names[i] = np.array([iop], dtype=object)
            else:
                self.right_operator_names[i] = np.array([iop, self.op], dtype=object)


class LocalMPO(MPO):
    def __init__(self, hamil, op_name, site_index=(), **kwargs):
        self.op = OpElement(op_name, site_index, q_label=hamil.empty)
        super().__init__(hamil, **kwargs)
    
    def _init_mpo_tensors(self, iprint):
        tensors = []
        iop = OpElement(OpNames.I, (), q_label=self.hamil.empty)
        for m in range(self.n_sites):
            if m == 0:
                mat = np.array([[self.op, iop]], dtype=object)
            elif m == self.n_sites - 1:
                mat = np.array([[iop], [self.op]], dtype=object)
            else:
                mat = np.array([[iop, 0], [self.op, iop]], dtype=object)
            ops = self.hamil.get_site_operators(m, { iop, self.op })
            tensors.append(OperatorTensor(mat=mat, tags={m}, ops=ops))
        return tensors


class IdentityMPOInfo(MPOInfo):
    def _init_operator_names(self):
        self.left_operator_names = [None] * self.n_sites
        self.right_operator_names = [None] * self.n_sites
        iop = OpElement(OpNames.I, (), q_label=self.hamil.empty)
        for i in range(self.n_sites):
            self.left_operator_names[i]  = np.array([iop], dtype=object)
            self.right_operator_names[i] = np.array([iop], dtype=object)


class IdentityMPO(MPO):
    def _init_mpo_tensors(self, iprint):
        tensors = []
        iop = OpElement(OpNames.I, (), q_label=self.hamil.empty)
        for m in range(self.n_sites):
            mat = np.array([[iop]], dtype=object)
            ops = self.hamil.get_site_operators(m, { iop })
            tensors.append(OperatorTensor(mat=mat, tags={m}, ops=ops))
        return tensors
