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
Symbolic operators.
"""

import numpy as np
from enum import Enum, auto

class OpNames(Enum):
    """Operator Names."""
    H = auto()
    I = auto()
    C = auto()
    D = auto()
    S = auto()
    SD = auto()
    R = auto()
    RD = auto()
    A = auto()
    AD = auto()
    P = auto()
    PD = auto()
    B = auto()
    Q = auto()
    
    def __repr__(self):
        return self.name

op_names = list(OpNames.__members__.items())

class OpExpression:
    pass

class OpElement(OpExpression):
    """
    Single operator symbol.
    
    Attributes:
        name : :class:`OpNames`
            Type of the operator.
        site_index : () or tuple(int..)
            Site indices of the operator.
        factor : float
            scalar factor.
        q_label : DirectProdGroup
            Quantum label of the operator.
    """
    __slots__ = ['name', 'site_index', 'factor', 'q_label']
    def __init__(self, name, site_index, factor=1, q_label=None):
        self.name = name
        assert isinstance(site_index, tuple)
        self.site_index = site_index
        self.factor = factor
        self.q_label = q_label
    
    def __repr__(self):
        if self.factor != 1:
            return '(%10.5f %r)' % (self.factor, abs(self))
        if len(self.site_index) == 0:
            return repr(self.name)
        elif len(self.site_index) == 1:
            return repr(self.name) + repr(self.site_index[0])
        else:
            return repr(self.name) + repr(self.site_index)
    
    def __mul__(self, other):
        if other == 0:
            return 0
        elif isinstance(other, float):
            return OpElement(self.name, self.site_index, self.factor * other, self.q_label)
        elif isinstance(other, OpSum):
            return OpSum([OpString([self] + st.ops, st.factor) for st in other.strings])
        else:
            return OpString([self, other])
    
    def __rmul__(self, other):
        if other == 0:
            return 0
        elif isinstance(other, float):
            return OpElement(self.name, self.site_index, self.factor * other, self.q_label)
        else:
            return OpString([other, self])
    
    def __add__(self, other):
        if other == 0:
            return self
        else:
            return OpSum([OpString([self]), OpString([other])])
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return OpSum([OpString([other]), OpString([self])])
    
    def __abs__(self):
        return OpElement(self.name, self.site_index, 1, self.q_label)
    
    def __eq__(self, other):
        if not isinstance(other, OpElement):
            return False
        else:
            return self.name == other.name and self.site_index == other.site_index \
                and self.factor == other.factor
    
    def __hash__(self):
        return hash((self.name, self.site_index, self.factor))
    
    @staticmethod
    def parse_site_index(expr):
        if len(expr) == 0:
            return ()
        elif expr.startswith('('):
            return tuple([int(x.strip()) for x in expr[1:-1].split(',')])
        else:
            return (int(expr), )
    
    @staticmethod
    def parse(expr):
        """Parse a str to operator symbol."""
        for name, op in sorted(op_names, key=lambda x: -len(x[0])):
            if expr.startswith(name):
                return OpElement(op, OpElement.parse_site_index(expr[len(name):].strip()))
    
class OpString(OpExpression):
    """
    String of operator symbols representing direct product of single operator symbols.
    
    Attributes:
        ops : list(:class:`OpElement`)
            A list of single operator symbols.
        sign : int (1 or -1)
            Sign factor. With SU(2) factor considered
    """
    __slots__ = ['ops', 'factor']
    def __init__(self, ops, factor=1):
        self.factor = np.prod([x.factor for x in ops]) * factor
        self.ops = [abs(x) for x in ops]
    
    def __repr__(self):
        if self.factor != 1:
            return '(%10.5f %r)' % (self.factor, abs(self))
        else:
            return " ".join([repr(x) for x in self.ops])
    
    def __truediv__(self, other):
        if isinstance(other, float):
            return OpString(self.ops, self.factor / other)
        else:
            print(other.__class__)
            assert False
    
    def __abs__(self):
        return OpString(self.ops, 1)
    
    def __mul__(self, other):
        if isinstance(other, OpElement):
            return OpString(self.ops + [other], self.factor)
        elif other == 0:
            return 0
        elif isinstance(other, float):
            return OpString(self.ops, self.factor * other)
        else:
            print(other.__class__)
            assert False
    
    def __add__(self, other):
        if other == 0:
            return self
        else:
            return OpSum([self, other])
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return OpSum([other, self])

class OpSum(OpExpression):
    """
    Sum of direct product of single operator symbols.
    
    Attributes:
        strings : list(:class:`OpString`)
    """
    __slots__ = ['strings']
    def __init__(self, strings):
        self.strings = strings
    
    def __repr__(self):
        return " + ".join([repr(x) for x in self.strings])
    
    def __add__(self, other):
        if isinstance(other, OpString):
            return OpSum(self.strings + [other])
        elif isinstance(other, OpSum):
            return OpSum(self.strings + other.strings)
        elif isinstance(other, int):
            return self
        else:
            print(other.__class__)
            assert False
    
    def __truediv__(self, other):
        if isinstance(other, float):
            return OpSum([x / other for x in self.strings])
        else:
            print(other.__class__)
            assert False
    
    def __mul__(self, other):
        if isinstance(other, OpElement):
            return OpSum([x * other for x in self.strings])
        elif other == 0:
            return 0
        elif isinstance(other, float):
            return OpSum([x * other for x in self.strings])
        else:
            print(other.__class__)
            assert False
    
    def __rmul__(self, other):
        if isinstance(other, OpElement):
            return OpSum([other * x for x in self.strings])
        elif other == 0:
            return 0
        elif isinstance(other, float):
            return OpSum([x * other for x in self.strings])
        else:
            print(other.__class__)
            assert False

