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
        return DirectProdGroup(self, o)


class ParticleN(HashIrrep):
    """Irreducible representation for particle number symmetry."""
    def __init__(self, n):
        self.ir = n

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
    def __init__(self, s):
        if isinstance(s, Fraction) or isinstance(s, float):
            self.ir = int(s * 2)
        else:
            self.ir = s

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
            return np.array([[0]], dtype=float)
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
    def __init__(self, sz):
        if isinstance(sz, Fraction) or isinstance(sz, float):
            self.ir = int(sz * 2)
        else:
            self.ir = sz

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
            self.ir = self.__class__.IrrepNames.index(ir)
        else:
            self.ir = ir

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


class PGC1(PointGroup):
    """:math:`C_1` point group."""
    Table = np.array([[0]], dtype=int)
    InverseElem = np.array(range(0, 1), dtype=int)
    IrrepNames = ["A"]


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


class CounterDict(dict):
    """Dictionary for counting number of objects."""
    def adjust(self, k, v):
        """Add v (int) to the item with key k."""
        if k in self:
            self[k] += v
        else:
            self[k] = v

    @staticmethod
    def intersect(a, b):
        """Return the intersection of two :class:`CounterDict`.
        If a key appears in both dicts, the lesser value will be retained."""
        r = CounterDict()
        if len(a) > len(b):
            a, b = b, a
        for k, va in a.items():
            if k in b:
                r[k] = min(va, b[k])
        return r


class LineCoupling:
    """
    Couple adjacent site tensor basis to form basis for renomalized tensor.
    
    Attributes:
        l : int
            Number of sites/orbitals.
        basis : [dict(DirectProdGroup -> int)]
            Site basis.
        empty : DirectProdGroup
            Vaccum state.
        target : DirectProdGroup
            target state.
        both_dir : bool
            If True, coupling will be calcuated from both left and right, and
            the intersection will be used.
            If False, coupling will only be calcuated from left.
        bond_dim : int
            Bond dimension. If -1, no basis truncation has been performed.
            This is defined with respect to the total number of states.
        dims : [CounterDict(DirectProdGroup -> int)]
            Renormalized basis.
    """
    def __init__(self, l, basis, target=None, empty=None, both_dir=True):
        self.l = l
        self.basis = basis
        self.target = target
        self.empty = empty
        self.both_dir = both_dir
        self.bond_dim = -1
        assert l != 0
        dim_l = self.fill_from_left()
        if both_dir:
            dim_r = self.fill_from_right()
            self.dims = [CounterDict.intersect(
                dim_l[i], dim_r[i]) for i in range(self.l)]
        else:
            self.dims = dim_l

    def fill_from_left(self):
        dim_l = [None] * self.l
        for d in range(0, self.l):
            dim_l[d] = CounterDict()
            if d == 0:
                for bst, bv in self.basis[d].items():
                    if self.target is None or bst <= self.target:
                        dim_l[d].adjust(bst, bv)
            else:
                for pst, pv in dim_l[d - 1].items():
                    for sts, sv in ((pst + bst, pv * bv) for bst, bv in self.basis[d].items()):
                        for st in sts if isinstance(sts, list) else [sts]:
                            if self.target is None or st <= self.target:
                                dim_l[d].adjust(st, sv)
        return dim_l

    def fill_from_right(self):
        dim_r = [None] * self.l
        for d in range(self.l - 1, -1, -1):
            dim_r[d] = CounterDict()
            if d == self.l - 1:
                dim_r[d].adjust(self.target, 1)
            else:
                for pst, pv in dim_r[d + 1].items():
                    for sts, sv in ((pst + (-bst), pv * bv) for bst, bv in self.basis[d + 1].items()):
                        for st in sts if isinstance(sts, list) else [sts]:
                            if self.target is None or st <= self.target:
                                dim_r[d].adjust(st, sv)
        return dim_r

    def set_bond_dim(self, m):
        """
        Truncate the renormalized basis, using the given bond dimension.
        Note that the ceiling is used for rounding for each quantum number,
        so the actual bond dimension is often larger than the given value.
        """
        self.bond_dim = m
        if m == -1:
            return
        for i in range(self.l):
            x = sum(self.dims[i].values())
            if x > m:
                for k, v in self.dims[i].items():
                    self.dims[i][k] = int(np.ceil(v * m / x))

    def __repr__(self):
        r = ""
        for i in range(l):
            r += "====== site : %3d [M =%5d] ======\n" % (
                i, sum(lc.dims[i].values()))
            for k, v in lc.dims[i].items():
                r += "%20r = %5d\n" % (k, v)
        return r


if __name__ == "__main__":
    # matrixelements/N2.STO3G.FCIDUMP
    spatial = ['Ag', 'Ag', 'Ag', 'B2g', 'B3g',
               'B1u', 'B1u', 'B1u', 'B2u', 'B3u']
    empty = ParticleN(0) * SU2(0) * PGD2H(0)
    basis = [{
        ParticleN(0) * SU2(0) * PGD2H(0): 1,
        # if the following value is 2, instead of 1, then reduced matrix will be 2x2
        # which is incorrect because it is the CGC space being 2x2, not reduced
        ParticleN(1) * SU2(Fraction(1, 2)) * PGD2H(sp): 1,
        ParticleN(2) * SU2(0) * PGD2H(0): 1
    } for sp in spatial]
    target = ParticleN(14) * SU2(0) * PGD2H(0)
    l = 10
    lc = LineCoupling(10, basis, target, empty, both_dir=False)
    for i in range(0, len(lc.dims)):
        print("=======", i)
        for k, v in lc.dims[i].items():
            print(k, v)
