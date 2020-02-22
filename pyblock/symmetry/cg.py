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
Clebsch-Gordan coefficients.
"""

import numpy as np
from itertools import accumulate, chain


class CG:
    """
    Precomputed numerical constants.
    
    Attributes:
        NSqrtFact : int
            Number of constants to generate.
        SqrtFact : [float]
            A list of size :attr:`NSqrtFact`, with element at index :math:`i` being :math:`\\sqrt{i!}`.
    """
    NSqrtFact = 200
    SqrtFact = list(accumulate([np.sqrt(x) for x in chain(
        [1], range(1, NSqrtFact))], lambda a, b: a * b))


# this is slow. Eventually this will be replaced by some more efficient code
class SU2CG:
    """SU(2) Clebsch-Gordan coefficients."""

    @staticmethod
    def sqrt_delta(ja, jb, jc):
        return CG.SqrtFact[int(ja + jb - jc)] * CG.SqrtFact[int(ja - jb + jc)] * \
            CG.SqrtFact[int(-ja + jb + jc)] / CG.SqrtFact[int(ja + jb + jc + 1)]
    
    @staticmethod
    def clebsch_gordan(ja, jb, jc, ma, mb, mc):
        pfactor = 1 if (mc + ja - jb) % 2 == 0 else - 1
        return pfactor * np.sqrt(int(2 * jc + 1)) * SU2CG.wigner_3j(ja, jb, jc, ma, mb, -mc)

    @staticmethod
    def wigner_3j(ja, jb, jc, ma, mb, mc):
        if int(ma + mb + mc) != 0:
            return 0.0

        alpha1 = int(jb - jc - ma)
        alpha2 = int(ja - jc + mb)
        beta1 = int(ja + jb - jc)
        beta2 = int(ja - ma)
        beta3 = int(jb + mb)

        max_alpha = max(0, alpha1, alpha2)
        min_beta = min(beta1, beta2, beta3)
        if min_beta < max_alpha:
            return 0.0

        num_terms = min_beta - max_alpha + 1

        prefactor = SU2CG.sqrt_delta(ja, jb, jc) * CG.SqrtFact[int(ja + ma)] * CG.SqrtFact[int(ja - ma)] * \
            CG.SqrtFact[int(jb + mb)] * CG.SqrtFact[int(jb - mb)] \
            * CG.SqrtFact[int(jc + mc)] * CG.SqrtFact[int(jc - mc)]
        
        r = 0.0
        factor = (1 if (((ja - jb - mc) + max_alpha) % 2 == 0) else - 1) * prefactor
        for term in range(num_terms):
            k = term + max_alpha
            t = CG.SqrtFact[k] * CG.SqrtFact[k - alpha1] * CG.SqrtFact[k - alpha2] * \
                CG.SqrtFact[beta1 - k] * CG.SqrtFact[beta2 - k] * CG.SqrtFact[beta3 - k]
            r += factor / (t * t)
            factor = -factor
        
        return r
