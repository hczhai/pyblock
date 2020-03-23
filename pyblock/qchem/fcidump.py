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

    def find_index(self, i, j):
        """Find linear index from full indices (i, j)."""
        if i < j:
            i, j = j, i
        return i * (i + 1) // 2 + j

    def __getitem__(self, idx):
        return self.data[self.find_index(*idx)]

    def __setitem__(self, idx, val):
        self.data[self.find_index(*idx)] = val
    
    def __eq__(self, other):
        return self.n == other.n and np.allclose(self.data, other.data)

    def __repr__(self):
        return [(i, j, self[i, j]) for i in range(self.n) for j in range(i + 1)].__repr__()
    
    def copy(self):
        r = self.__class__(self.n)
        r.data = self.data.copy()
        return r


# two-electron integrals
class VInt(TInt):
    """
    Symmetric rank-4 array (:math:`V_{ijkl} = V_{jikl} = V_{ijlk} = V_{klij}`) for two-electron integral storage.
    
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

    def find_index(self, i, j, k, l):
        """Find linear index from full indices (i, j, k, l)."""
        p = super().find_index(i, j)
        q = super().find_index(k, l)
        return super().find_index(p, q)

    def __repr__(self):
        ri = range(self.n)

        def rj(i): return range(i + 1)

        def rk(i): return range(i + 1)

        def rl(i, j, k): return range(j + 1) if k == i else range(k + 1)

        return [(i, j, k, l, self[i, j, k, l]) for i in ri for j in rj(i)
                for k in rk(i) for l in rl(i, j, k)].__repr__()


# two-electron integrals
class UVInt(TInt):
    """
    Symmetric rank-4 array (:math:`V_{ijkl} = V_{jikl} = V_{ijlk}`) for two-electron integral storage.
    
    Attributes:
        n : int
            Number of orbitals.
        data : numpy.ndarray
            1D flat array of size :math:`m(m+1)/2` where :math:`m=n(n+1)/2`.
    """
    def __init__(self, n):
        self.n = n
        self.m = n * (n + 1) // 2
        self.data = np.zeros((self.m * self.m, ))

    def find_index(self, i, j, k, l):
        """Find linear index from full indices (i, j, k, l)."""
        p = super().find_index(i, j)
        q = super().find_index(k, l)
        return p * self.m + q

    def __repr__(self):
        ri = range(self.n)

        def rj(i): return range(i + 1)

        return [(i, j, k, l, self[i, j, k, l]) for i in ri for j in rj(i)
                for k in ri for l in rj(k)].__repr__()


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

    cont = ','.join(pars.split()[1:])
    cont = cont.split(',')
    cont_dict = {}
    p_key = None
    for c in cont:
        if '=' in c or p_key is None:
            p_key, b = c.split('=')
            cont_dict[p_key.strip().lower()] = b.strip()
        elif len(c.strip()) != 0:
            if len(cont_dict[p_key.strip().lower()]) != 0:
                cont_dict[p_key.strip().lower()] += ',' + c.strip()
            else:
                cont_dict[p_key.strip().lower()] = c.strip()
    
    for k, v in cont_dict.items():
        if ',' in v:
            v = cont_dict[k] = v.split(',')
    
    n = int(cont_dict['norb'])
    data = []
    for l in ints.split('\n'):
        ll = l.strip()
        if len(ll) == 0 or ll.strip()[0] == '!':
            continue
        ll = ll.split()
        d = float(ll[0])
        i, j, k, l = [int(x) for x in ll[1:]]
        data.append((i, j, k, l, d))
    if int(cont_dict.get('iuhf', 0)) == 0:
        t = TInt(n)
        v = VInt(n)
        e = 0.0
        for i, j, k, l, d in data:
            if i + j + k + l == 0:
                e = d
            elif k + l == 0:
                t[i - 1, j - 1] = d
            else:
                v[i - 1, j - 1, k - 1, l - 1] = d
        return cont_dict, (t, v, e)
    else:
        ts = (TInt(n), TInt(n))
        vs = (VInt(n), UVInt(n), UVInt(n), VInt(n))
        e = 0.0
        ip = 0
        for i, j, k, l, d in data:
            if i + j + k + l == 0:
                ip += 1
                if ip == 6:
                    e = d
            elif k + l == 0:
                assert ip == 3 or ip == 4
                if ip == 3:
                    ts[0][i - 1, j - 1] = d
                elif ip == 4:
                    ts[1][i - 1, j - 1] = d
            else:
                assert ip <= 2
                if ip == 0:
                    vs[0][i - 1, j - 1, k - 1, l - 1] = d
                elif ip == 1:
                    vs[3][i - 1, j - 1, k - 1, l - 1] = d
                elif ip == 2:
                    vs[1][i - 1, j - 1, k - 1, l - 1] = vs[2][k - 1, l - 1, i - 1, j - 1] = d
        return cont_dict, (ts, vs, e)

def write_fcidump(filename, h1e, h2e, nmo, nelec, nuc, ms, isym=1, orbsym=None, tol=1E-15):
    with open(filename, 'w') as fout:
        fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
        if orbsym is not None and len(orbsym) > 0:
            fout.write('  ORBSYM=%s,\n' % ','.join([str(x) for x in orbsym]))
        else:
            fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
        fout.write('  ISYM=%d,\n' % isym)
        if isinstance(h1e, tuple) and len(h1e) == 2:
            fout.write('  IUHF=1,\n')
        fout.write(' &END\n')
        output_format = '%20.16f%4d%4d%4d%4d\n'
        npair = nmo * (nmo + 1) // 2

        def write_eri(fout, eri):
            assert eri.ndim in [1, 2]
            if eri.ndim == 2:
                # 4-fold symmetry
                assert(eri.size == npair ** 2)
                ij = 0
                for i in range(nmo):
                    for j in range(0, i + 1):
                        kl = 0
                        for k in range(0, nmo):
                            for l in range(0, k + 1):
                                if abs(eri[ij, kl]) > tol:
                                    fout.write(output_format % (eri[ij, kl], i + 1, j + 1, k + 1, l + 1))
                                kl += 1
                        ij += 1
            else:
                # 8-fold symmetry
                assert(eri.size == npair * (npair + 1) // 2)
                ij = 0
                ijkl = 0
                for i in range(nmo):
                    for j in range(0, i + 1):
                        kl = 0
                        for k in range(0, i + 1):
                            for l in range(0, k + 1):
                                if ij >= kl:
                                    if abs(eri[ijkl]) > tol:
                                        fout.write(output_format % (eri[ijkl], i + 1, j + 1, k + 1, l + 1))
                                    ijkl += 1
                                kl += 1
                        ij += 1

        def write_h1e(fout, hx):
            h = hx.reshape(nmo, nmo)
            for i in range(nmo):
                for j in range(0, i + 1):
                    if abs(h[i, j]) > tol:
                        fout.write(output_format % (h[i, j], i + 1, j + 1, 0, 0))

        if isinstance(h2e, tuple):
            assert len(h2e) == 3
            vaa, vab, vbb = h2e
            assert vaa.ndim == vbb.ndim == 1 and vab.ndim == 2
            for eri in [vaa, vbb, vab]:
                write_eri(fout, eri)
                fout.write(output_format % (0, 0, 0, 0, 0))
            assert len(h1e) == 2
            for hx in h1e:
                write_h1e(fout, hx)
                fout.write(output_format % (0, 0, 0, 0, 0))
            fout.write(output_format % (nuc, 0, 0, 0, 0))
        else:
            write_eri(fout, h2e)
            write_h1e(fout, h1e)
            fout.write(output_format % (nuc, 0, 0, 0, 0))
