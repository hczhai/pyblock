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
Symmetry related data structures.
"""

import numpy as np
from fractions import Fraction
from .cg import SU2CG


class HashIrrep:
    """Base class for irreducible representation, supporting hashing."""
    def __init__(self, ir):
        self.ir = ir

    def __eq__(self, o):
        return self.ir == o.ir
    
    def __lt__(self, o):
        return self.ir < o.ir

    def __hash__(self):
        return hash(self.ir)

    # direct product of subspaces
    def __mul__(self, o):
        return DirectProdGroup(self) * o


class ParticleN(HashIrrep):
    """Irreducible representation for particle number symmetry."""
    CachedP = []
    CachedM = []
    
    def __init__(self, ir):
        pass
    
    def __new__(cls, ir):
        if ir >= 0:
            for irx in range(len(cls.CachedP), ir + 1):
                cls.CachedP.append(super().__new__(cls))
                super().__init__(cls.CachedP[irx], irx)
            return cls.CachedP[ir]
        else:
            for irx in range(len(cls.CachedM), -ir + 1):
                cls.CachedM.append(super().__new__(cls))
                super().__init__(cls.CachedM[irx], -irx)
            return cls.CachedM[-ir]

    # group multiplication
    def __add__(self, o):
        return self.__class__(self.ir + o.ir)

    # inverse element
    def __neg__(self):
        return self.__class__(-self.ir)

    def __repr__(self):
        return "N=" + str(self.ir)

    # this checks whether rhs irrep is achievable from lhs irrep
    def __le__(self, o):
        return self.ir <= o.ir and self.ir >= 0

    @staticmethod
    def clebsch_gordan(a, b, c):
        return np.array([[[1 if a + b == c else 0]]], dtype=float)


class SU2(HashIrrep):
    """Irreducible representation for SU(2) spin symmetry."""
    Cached = []
    
    def __init__(self, s):
        pass
    
    def __new__(cls, s):
        if isinstance(s, Fraction) or isinstance(s, float):
            s = int(s * 2)
        for irx in range(len(cls.Cached), s + 1):
            cls.Cached.append(super().__new__(cls))
            super().__init__(cls.Cached[irx], irx)
        return cls.Cached[s]

    def __add__(self, o):
        return [self.__class__(ir) for ir in range(abs(self.ir - o.ir), self.ir + o.ir + 1, 2)]

    # add a spin is the same as minus a spin, since it is triangle
    def __neg__(self):
        return self.__class__(self.ir)

    def __repr__(self):
        return "S=" + str(Fraction(self.ir, 2))

    def __le__(self, _):
        return True

    @staticmethod
    def clebsch_gordan(a, b, c):
        """Return rank-3 numpy.ndarray for CG coefficients with all possible projected quantum numbers."""
        if c not in a + b:
            return np.array([[[0]]], dtype=float)
        else:
            na = a.ir + 1
            nb = b.ir + 1
            nc = c.ir + 1
            ja = Fraction(a.ir, 2)
            jb = Fraction(b.ir, 2)
            jc = Fraction(c.ir, 2)
            r = np.zeros((na, nb, nc), dtype=float)
            for ima in range(na):
                ma = -ja + ima
                for imc in range(nc):
                    mc = -jc + imc
                    mb = mc - ma
                    if abs(mb) <= jb:
                        r[ima, int(mb + jb), imc] = SU2CG.clebsch_gordan(
                            ja, jb, jc, ma, mb, mc)
            return r

    @property
    def j(self):
        """SU(2) spin quantum number."""
        return Fraction(self.ir, 2)

    def to_multi(self):
        return self


class SZ(HashIrrep):
    """Irreducible representation for projected spin symmetry."""
    CachedP = []
    CachedM = []
    
    def __init__(self, sz):
        pass
    
    def __new__(cls, sz):
        if isinstance(sz, Fraction) or isinstance(sz, float):
            sz = int(sz * 2)
        if sz >= 0:
            for irx in range(len(cls.CachedP), sz + 1):
                cls.CachedP.append(super().__new__(cls))
                super().__init__(cls.CachedP[irx], irx)
            return cls.CachedP[sz]
        else:
            for irx in range(len(cls.CachedM), -sz + 1):
                cls.CachedM.append(super().__new__(cls))
                super().__init__(cls.CachedM[irx], -irx)
            return cls.CachedM[-sz]

    # group multiplication
    def __add__(self, o):
        return self.__class__(self.ir + o.ir)

    # inverse element
    def __neg__(self):
        return self.__class__(-self.ir)

    def __repr__(self):
        return "SZ=" + str(Fraction(self.ir, 2))

    def __le__(self, o):
        return True

    @staticmethod
    def clebsch_gordan(a, b, c):
        return np.array([[[1 if a + b == c else 0]]], dtype=float)


# SU2 irreducible repr with sz quantum number
class SU2Proj(SU2):
    """Irreducible representation for SU(2) spin symmetry with extra projected spin label."""
    Cached = []
    
    def __init__(self, s, sz):
        pass
    
    def __new__(cls, s, sz):
        if isinstance(s, Fraction) or isinstance(s, float):
            s = int(s * 2)
        if isinstance(sz, Fraction) or isinstance(sz, float):
            sz = int(sz * 2)
        for irx in range(len(cls.Cached), s + 1):
            cls.Cached.append([HashIrrep.__new__(cls) for _ in range(0, irx + 1)])
            for irz in range(0, irx + 1):
                cls.Cached[-1][irz].ir = irx
                cls.Cached[-1][irz].pir = -irx + irz * 2
        return cls.Cached[s][(sz + s) // 2]
    
    def __init__(self, s, sz):
        if isinstance(sz, Fraction) or isinstance(sz, float):
            self.pir = int(sz * 2)
        else:
            self.pir = sz
        super().__init__(s)
        assert abs(self.pir) <= self.ir and (self.pir - self.ir) % 2 == 0

    def __add__(self, o):
        return [self.__class__(ir, self.pir + o.pir)
                for ir in range(abs(self.ir - o.ir), self.ir + o.ir + 1, 2)]

    def __eq__(self, o):
        return self.ir == o.ir and self.pir == o.pir

    def __hash__(self):
        return hash((self.ir, self.pir))

    def __neg__(self):
        return self.__class__(self.ir, -self.pir)

    def __repr__(self):
        return "S,Sz=" + str(Fraction(self.ir, 2)) + "," + str(Fraction(self.pir, 2))

    @staticmethod
    def random_from_multi(multi):
        return SU2Proj(s=multi.ir, sz=-multi.ir + np.random.randint(multi.ir + 1) * 2)

    def to_multi(self):
        return SU2(s=self.ir)
    
    def copy(self):
        r = HashIrrep.__new__(self.__class__)
        r.ir = self.ir
        r.pir = self.pir
        return r

    @property
    def jz(self):
        """SU(2) projected spin quantum number."""
        return Fraction(self.pir, 2)

    @jz.setter
    def jz(self, sz):
        if isinstance(sz, Fraction) or isinstance(sz, float):
            self.pir = int(sz * 2)
        else:
            self.pir = sz


class PointGroup(HashIrrep):
    """
    Base class for irreducible representation for point group symmetry.
    
    Attributes:
        Table : rank-2 numpy.ndarray
            Mutiplication table of the group.
        InverseElem : rank-1 numpy.ndarray
            Inverse Element of each element in the group.
        IrrepNames : list(str)
            Name of each irreducible representation in the group.
    """
    Table = np.zeros((0, 0), dtype=int)
    InverseElem = np.zeros((0,), dtype=int)
    IrrepNames = []

    def __init__(self, ir):
        if isinstance(ir, str):
            ir = self.__class__.IrrepNames.index(ir)
        if self.__class__.Cached[ir] is self:
            return
        self.__class__.Cached[ir] = self
        self.ir = ir
    
    def __new__(cls, ir):
        if isinstance(ir, str):
            ir = cls.IrrepNames.index(ir)
        if cls.Cached[ir] is not None:
            return cls.Cached[ir]
        else:
            return super().__new__(cls)

    def __add__(self, o):
        return self.__class__(self.__class__.Table[self.ir, o.ir])

    def __neg__(self):
        return self.__class__(self.__class__.InverseElem[self.ir])

    def __repr__(self):
        return self.__class__.IrrepNames[self.ir]

    def __le__(self, _):
        return True

    @staticmethod
    def clebsch_gordan(a, b, c):
        return np.array([[[1 if a + b == c else 0]]], dtype=float)


class PGD2H(PointGroup):
    """:math:`D_{2h}` point group."""
    Table = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7],
         [1, 0, 3, 2, 5, 4, 7, 6],
         [2, 3, 0, 1, 6, 7, 4, 5],
         [3, 2, 1, 0, 7, 6, 5, 4],
         [4, 5, 6, 7, 0, 1, 2, 3],
         [5, 4, 7, 6, 1, 0, 3, 2],
         [6, 7, 4, 5, 2, 3, 0, 1],
         [7, 6, 5, 4, 3, 2, 1, 0]], dtype=int)
    InverseElem = np.array(range(0, 8), dtype=int)
    IrrepNames = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
    Cached = [None] * 8


class PGC1(PointGroup):
    """:math:`C_1` point group."""
    Table = np.array([[0]], dtype=int)
    InverseElem = np.array(range(0, 1), dtype=int)
    IrrepNames = ["A"]
    Cached = [None] * 1


def point_group(pg_name):
    """Return point group class corresponding to the point group name."""
    return {'c1': PGC1, 'd2h': PGD2H}[pg_name]


class DirectProdGroup:
    """
    Irreducible representation for symmetry formed by direct product of sub-symmetries.
    
    Attributes:
        irs : list(Group)
            A list of irreducible representations for sub-symmetries.
        ng : int
            Number of sub-symmetries.
    """
    def __init__(self, *args):
        self.irs = args
        self.ng = len(args)
        assert self.ng != 0

    def __add__(self, o):
        assert self.ng == o.ng

        def l_list(x):
            return x if isinstance(x, list) else [x]

        l = map(lambda x: [x], l_list(self.irs[0] + o.irs[0]))
        for d in range(1, self.ng):
            r = l_list(self.irs[d] + o.irs[d])
            l = [p + [q] for p in l for q in r]
        l = [self.__class__(*irs) for irs in l]
        return l[0] if len(l) == 1 else l

    def __neg__(self):
        return self.__class__(*[-ir for ir in self.irs])
    
    def __sub__(self, o):
        return self + (-o)

    def __repr__(self):
        return "< %s>" % ("%s " * self.ng) % tuple(str(ir) for ir in self.irs)

    def __eq__(self, o):
        return all(ir == oir for ir, oir in zip(self.irs, o.irs))

    # this is actually hard to determine for an arbitrary combination
    # not a simple relation
    def __le__(self, o):
        return all(ir <= oir for ir, oir in zip(self.irs, o.irs))
    
    def __lt__(self, o):
        for ir, oir in zip(self.irs, o.irs):
            if ir != oir:
                return ir < oir
        return False

    def __hash__(self):
        return hash(tuple(self.irs))

    def __mul__(self, o):
        if isinstance(o, self.__class__):
            return self.__class__(*(self.irs + o.irs))
        else:
            return self.__class__(*(self.irs + (o,)))

    def sub_group(self, idx):
        """Return a subgroup of this object, given a list of indices."""
        return DirectProdGroup(*[self.irs[id] for id in idx])
