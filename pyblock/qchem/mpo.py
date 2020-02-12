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

from ..tensor.operator import OpElement, OpNames
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
    def __init__(self, hamil):
        self.hamil = hamil
        self.n_sites = hamil.n_sites
        self._init_operator_names()
    
    def _init_operator_names(self):
        self.left_operator_names = [None] * self.n_sites
        self.right_operator_names = [None] * self.n_sites
        
        for i in range(self.n_sites):
            lshape = 2 + 2 * i if i != 0 else 1
            rshape = 4 + 2 * i if i != self.n_sites - 1 else 1
            lop = np.zeros((lshape, ), dtype=object)
            rop = np.zeros((rshape, ), dtype=object)
            lop[-1] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
            if i != 0:
                lop[0] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
                for j in range(0, i):
                    lop[1 + j * 2] = OpElement(OpNames.S, (j, ), q_label=-self.hamil.creation_q_labels[j])
                    lop[2 + j * 2] = -OpElement(OpNames.SD, (j, ), q_label=self.hamil.creation_q_labels[j])
            rop[0] = OpElement(OpNames.H, (), q_label=self.hamil.empty)
            if i != self.n_sites - 1:
                for j in range(i + 1):
                    rop[1 + j * 2] = OpElement(OpNames.C, (j, ), q_label=self.hamil.creation_q_labels[j])
                    rop[2 + j * 2] = OpElement(OpNames.D, (j, ), q_label=-self.hamil.creation_q_labels[j])
                rop[-1] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
            self.left_operator_names[i] = lop
            self.right_operator_names[i] = rop


class MPO(TensorNetwork):
    def __init__(self, hamil):
        self.n_sites = hamil.n_sites
        self.hamil = hamil
        tensors = self._init_mpo_tensors()
        super().__init__(tensors)
    
    def _init_mpo_tensors(self):
        """Generate :attr:`tensors`."""
        tensors = []
        op_h = OpElement(OpNames.H, ())
        op_i = OpElement(OpNames.I, ())
        op_c = OpElement(OpNames.C, ())
        op_d = OpElement(OpNames.D, ())
        for i in range(self.n_sites):
            if i == 0:
                mat = np.array([[op_h, op_c, op_d, op_i]], dtype=object)
            else:
                if i == self.n_sites - 1:
                    mat = np.zeros((2 + 2 * i, 1), dtype=object)
                else:
                    mat = np.zeros((2 + 2 * i, 4 + 2 * i), dtype=object)
                    mat[1 + 2 * i, 2 * i + 1] = op_c
                    mat[1 + 2 * i, 2 * i + 2] = op_d
                    mat[1 + 2 * i, 2 * i + 3] = op_i
                    for j in range(1, 2 * i + 1):
                        mat[j, j] = op_i
                mat[0, 0] = op_i
                mat[1 + 2 * i, 0] = op_h
                for j in range(0, i):
                    mat[1 + j * 2, 0] = OpElement(OpNames.S, (j, ))
                    mat[2 + j * 2, 0] = -OpElement(OpNames.SD, (j, ))
            tensors.append(OperatorTensor(mat=mat, tags={i},
                ops=self.hamil.get_site_operators(i)))
        return tensors
