
import numpy as np
from enum import Enum, auto

class OpNames(Enum):
    H = auto()
    S = auto()
    D = auto()
    C = auto()
    I = auto()
    SD = auto()
    
    def __repr__(self):
        return self.name

op_names = [
    ('a', OpNames.D), ('ad', OpNames.C), ('I', OpNames.I), ('H', OpNames.H),
    ('S', OpNames.S), ('Sd', OpNames.SD)
]

class OpExpression:
    pass

class OpElement(OpExpression):
    __slots__ = ['name', 'site_index', 'sign']
    def __init__(self, name, site_index, sign=1):
        self.name = name
        self.site_index = site_index
        self.sign = sign
    
    def __repr__(self):
        if self.sign == -1:
            return '(-%r)' % (-self)
        if len(self.site_index) == 0:
            return repr(self.name)
        elif len(self.site_index) == 1:
            return repr(self.name) + repr(self.site_index[0])
        else:
            return repr(self.name) + repr(self.site_index)
    
    def __mul__(self, other):
        if other == 0:
            return 0
        else:
            return OpString([self, other])
    
    def __rmul__(self, other):
        if other == 0:
            return 0
        else:
            return OpString([other, self])
    
    def __neg__(self):
        return OpElement(self.name, self.site_index, -self.sign)
    
    def __abs__(self):
        return OpElement(self.name, self.site_index, 1)
    
    def __eq__(self, other):
        if not isinstance(other, OpElement):
            return False
        else:
            return self.name == other.name and self.site_index == other.site_index \
                and self.sign == other.sign
    
    def __hash__(self):
        return hash((self.name, self.site_index, self.sign))
    
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
        for name, op in sorted(op_names, key=lambda x: -len(x[0])):
            if expr.startswith(name):
                return OpElement(op, OpElement.parse_site_index(expr[len(name):].strip()))
    
class OpString(OpExpression):
    __slots__ = ['ops', 'sign']
    def __init__(self, ops, sign=1):
        self.sign = np.prod([x.sign for x in ops]) * sign
        self.ops = [abs(x) for x in ops]
    
    def __repr__(self):
        if self.sign == -1:
            return '(-%r)' % (-self)
        else:
            return " ".join([repr(x) for x in self.ops])
    
    def __neg__(self):
        return OpString(self.ops, -self.sign)
    
    def __mul__(self, other):
        if isinstance(other, OpElement):
            return OpString(self.ops + [other], self.sign)
        elif other == 0:
            return 0
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
            
    def __neg__(self):
        return OpSum([-x for x in self.strings])
    
    def __mul__(self, other):
        if isinstance(other, OpElement):
            return OpSum([x * other for x in self.strings])
        elif other == 0:
            return 0
        else:
            print(other.__class__)
            assert False
