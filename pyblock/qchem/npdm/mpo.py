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
MPO for N-particle density matrix.
"""

from ..operator import OpElement, OpNames
from ..mpo import DualOperatorTensor, MPOInfo, MPO
import numpy as np


class PDM1MPOInfo(MPOInfo):
    def _init_operator_names(self):
        self.left_operator_names = [None] * self.n_sites
        self.right_operator_names = [None] * self.n_sites
        self.middle_operators = [None] * self.n_sites
        for i in range(self.n_sites):
            lshape = 1 + 2 * (i + 1) if i != self.n_sites - 1 else 1
            rshape = 1 if i != self.n_sites - 1 else 3
            lop = np.zeros((lshape, ), dtype=object)
            rop = np.zeros((rshape, ), dtype=object)
            lop[0] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
            if i != self.n_sites - 1:
                for j in range(0, i + 1):
                    lop[1 + j] = OpElement(OpNames.C, (j, ), q_label=self.hamil.one_site_q[j])
                    lop[1 + (i + 1) + j] = OpElement(OpNames.B, (j, i, 0), q_label=self.hamil.two_site_minus_q[j, i][0])
            rop[0] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
            if i == self.n_sites - 1:
                rop[1] = OpElement(OpNames.B, (i, i, 0), q_label=self.hamil.two_site_minus_q[i, i][0])
                rop[2] = OpElement(OpNames.D, (i, ), q_label=-self.hamil.one_site_q[i])
            self.left_operator_names[i] = lop
            self.right_operator_names[i] = rop
            ops = []
            if i != self.n_sites - 1:
                for j in range(0, i + 1):
                    expr = np.sqrt(2) * (OpElement(OpNames.B, (j, i, 0)) * OpElement(OpNames.I, ()))
                    ops.append((OpElement(OpNames.PDM1, (i, j), q_label=self.hamil.two_site_minus_q[i, j][0]), expr))
                    if i != j:
                        ops.append((OpElement(OpNames.PDM1, (j, i)), expr))
                if i == self.n_sites - 2:
                    for j in range(0, i + 1):
                        expr = np.sqrt(2) * (OpElement(OpNames.C, (j, )) * OpElement(OpNames.D, (i + 1, )))
                        ops.append((OpElement(OpNames.PDM1, (j, i + 1), q_label=self.hamil.two_site_minus_q[j, i + 1][0]), expr))
                        ops.append((OpElement(OpNames.PDM1, (i + 1, j), q_label=self.hamil.two_site_minus_q[i + 1, j][0]), expr))
                    expr = np.sqrt(2) * (OpElement(OpNames.I, ()) * OpElement(OpNames.B, (i + 1, i + 1, 0)))
                    ops.append((OpElement(OpNames.PDM1, (i + 1, i + 1), q_label=self.hamil.two_site_minus_q[i + 1, i + 1][0]), expr))
            self.middle_operators[i] = ops


class PDM1MPO(MPO):
    def _init_mpo_tensors(self, iprint):
        tensors = []
        for m in range(self.n_sites):
            lrshape = 1 + 2 * (m + 1) if m != self.n_sites - 1 else 1
            llshape = 1 + 2 * m
            lmat = np.zeros((llshape, lrshape), dtype=object)
            lmat[0, 0] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
            if m != self.n_sites - 1:
                pi = 0
                pad = 1
                p = 1
                for i in range(0, m):
                    lmat[pad + i, p + i] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                lmat[pi, p + m] = OpElement(OpNames.C, (m, ), q_label=self.hamil.one_site_q[m])
                p += m + 1
                for i in range(0, m):
                    lmat[pad + i, p + i] = OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])
                lmat[pi, p + m] = OpElement(OpNames.B, (m, m, 0), q_label=self.hamil.two_site_minus_q[m, m][0])
                p += m + 1
                assert p == lrshape
            if m == self.n_sites - 1:
                rmat = np.array([
                    [OpElement(OpNames.I, (), q_label=self.hamil.empty)],
                    [OpElement(OpNames.B, (m, m, 0), q_label=self.hamil.two_site_minus_q[m, m][0])],
                    [OpElement(OpNames.D, (m, ), q_label=-self.hamil.one_site_q[m])]
                ])
            elif m == self.n_sites - 2:
                rmat = np.array([[OpElement(OpNames.I, (), q_label=self.hamil.empty), 0, 0]])
            else:
                rmat = np.array([[OpElement(OpNames.I, (), q_label=self.hamil.empty)]])

            [lmat, rmat], ops = self._post_check_mpo_operators([lmat, rmat], m)

            tensors.append(DualOperatorTensor(lmat=lmat, rmat=rmat, tags={m}, ops=ops))

        return tensors
