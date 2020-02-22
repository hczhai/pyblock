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
Basis transformation for site operators.
"""

import numpy as np
from .symmetry import ParticleN, SU2Proj, SU2, DirectProdGroup
from fractions import Fraction

slater_basis = [ParticleN(0) * SU2Proj(0, 0),
               ParticleN(1) * SU2Proj(Fraction(1, 2), Fraction(-1, 2)),
               ParticleN(1) * SU2Proj(Fraction(1, 2), Fraction(1, 2)),
               ParticleN(2) * SU2Proj(0, 0)]

su2_basis = [ParticleN(0) * SU2(0),
            ParticleN(1) * SU2(Fraction(1, 2)),
            ParticleN(2) * SU2(0)]

def basis_transform(mat, q_label, old_basis, new_basis):
    """
    Transform the matrix representation of an site operator
    from one basis to another basis, using Wigner-Eckart theorem."""
    old = old_basis[0]
    new = new_basis[0]
    trans = []
    for inb, ib in zip(new.irs, old.irs):
        if ib.__class__ in inb.__class__.__bases__:
            trans.append((lambda x: x.to_multi(), "new is proj"))
        elif inb.__class__ in ib.__class__.__bases__:
            trans.append(
                (lambda x, ib=ib: ib.__class__.random_from_multi(x), "old is proj"))
        elif ib.__class__ is inb.__class__:
            trans.append((lambda x: x, None))
        else:
            raise TypeError("cannot transform between the given basis")
    new_mat = np.zeros((len(new_basis), len(new_basis)))
    for ib, bra in enumerate(new_basis):
        for ik, ket in enumerate(new_basis):
            factor = 1.0
            old_bra = []
            old_ket = []
            for irb, irk, tr, iq in zip(bra.irs, ket.irs, trans, q_label.irs):
                if tr[1] is not None:
                    cgmat = irb.to_multi().__class__.clebsch_gordan(
                        irk.to_multi(), iq.to_multi(), irb.to_multi())
                    if np.allclose(cgmat, 0.0):
                        factor = None
                        break
                orb = tr[0](irb)
                ork = tr[0](irk)
                if tr[1] == "new is proj":
                    if irk.jz + iq.jz != irb.jz:
                        factor = None
                        break
                    else:
                        factor = factor * \
                            cgmat[int(irk.jz + irk.j), int(iq.jz + iq.j),
                                  int(irb.jz + irb.j)]
                elif tr[1] == "old is proj":
                    found = False
                    for iork_jz in range(0, cgmat.shape[0]):
                        ork.jz = -irk.j + iork_jz
                        orb.jz = -irk.j + iork_jz + iq.jz
                        if abs(orb.jz) <= irb.j and \
                                not np.isclose(cgmat[int(ork.jz + irk.j), int(iq.jz + iq.j),
                                                     int(orb.jz + irb.j)], 0.0):
                            found = True
                            break
                    assert found
                    factor = factor / cgmat[int(ork.jz + irk.j), int(iq.jz + iq.j),
                                            int(orb.jz + irb.j)]
                if np.isclose(factor, 0.0):
                    break
                old_bra.append(orb)
                old_ket.append(ork)
            if factor is None:
                continue
            old_bra = DirectProdGroup(*old_bra)
            old_ket = DirectProdGroup(*old_ket)
            new_mat[ib, ik] = mat[old_basis.index(old_bra), old_basis.index(old_ket)] * factor
    return new_mat
