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
FCIDUMP file and storage of integrals.
"""

import numpy as np


# one-electron integrals
class TInt:
    """
    Symmetric rank-2 array (:math:`T_{ij} = T_{ji}`) for one-electron integral storage.
    
    Attributes:
        n : int
            Number of orbitals.
        data : numpy.ndarray
            1D flat array of size :math:`n(n+1)/2`.
    """
    def __init__(self, n):
        self.n = n
        self.data = np.zeros((n * (n + 1) // 2, ))

    @staticmethod
    def find_index(i, j):
        """Find linear index from full indices (i, j)."""
        if i < j:
            i, j = j, i
        return i * (i + 1) // 2 + j

    def __getitem__(self, idx):
        return self.data[self.__class__.find_index(*idx)]

    def __setitem__(self, idx, val):
        self.data[self.__class__.find_index(*idx)] = val

    def __repr__(self):
        return [(i, j, self[i, j]) for i in range(self.n) for j in range(i + 1)].__repr__()


# two-electron integrals
class VInt(TInt):
    """
    Symmetric rank-4 array (:math:`T_{ijkl} = T_{jikl} = T_{ijlk} = T_{klij}`) for two-electron integral storage.
    
    Attributes:
        n : int
            Number of orbitals.
        data : numpy.ndarray
            1D flat array of size :math:`m(m+1)/2` where :math:`m=n(n+1)/2`.
    """
    def __init__(self, n):
        self.n = n
        m = n * (n + 1) // 2
        self.data = np.zeros((m * (m + 1) // 2, ))

    @staticmethod
    def find_index(i, j, k, l):
        """Find linear index from full indices (i, j, k, l)."""
        p = TInt.find_index(i, j)
        q = TInt.find_index(k, l)
        return TInt.find_index(p, q)

    def __repr__(self):
        ri = range(self.n)

        def rj(i): return range(i + 1)

        def rk(i): return range(i + 1)

        def rl(i, j, k): return range(j + 1) if k == i else range(k + 1)

        return [(i, j, k, l, self[i, j, k, l]) for i in ri for j in rj(i)
                for k in rk(i) for l in rl(i, j, k)].__repr__()


# read FCIDUMP file
# return : options, (1-e array, 2-e array, const energy term)
def read_fcidump(filename):
    """
    Read FCI options and integrals from FCIDUMP file.
    
    Args:
        filename : str
    
    Returns:
        cont_dict : dict
            FCI options or input parameters.
        (t, v, e) : (TInt, VInt, float)
            One- and two-electron integrals and const energy.
    """
    with open(filename, 'r') as f:
        ff = f.read().lower()
        if '/' in ff:
            pars, ints = ff.split('/')
        elif '&end' in ff:
            pars, ints = ff.split('&end')
        cont = ' '.join(pars.split()[1:])
        cont = cont.split(',')
        cont_dict = {}
        p_key = None
        for c in cont:
            if '=' in c or p_key is None:
                p_key, b = c.split('=')
                cont_dict[p_key.strip().lower()] = b.strip()
            elif len(c.strip()) != 0:
                cont_dict[p_key.strip().lower()] += ',' + c.strip()

        n = int(cont_dict['norb'])
        t = TInt(n)
        v = VInt(n)
        e = 0.0
        for l in ints.split('\n'):
            ll = l.strip()
            if len(ll) == 0:
                continue
            ll = ll.split()
            d = float(ll[0])
            i, j, k, l = [int(x) for x in ll[1:]]
            if i + j + k + l == 0:
                e = d
            elif k + l == 0:
                t[i - 1, j - 1] = d
            else:
                v[i - 1, j - 1, k - 1, l - 1] = d
    return cont_dict, (t, v, e)


if __name__ == "__main__":
    opts, (t, v, e) = read_fcidump('N2.STO3G.FCIDUMP')
    print(e)
    print(v)
