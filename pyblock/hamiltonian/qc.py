
import numpy as np
from fractions import Fraction
from ..symmetry.symmetry import point_group, ParticleN, SU2
from ..tensor.tensor import Tensor, SubTensor

# one-electron integrals


class TInt:
    def __init__(self, n):
        self.n = n
        self.data = np.zeros((n * (n + 1) // 2, ))

    @staticmethod
    def find_index(i, j):
        return i * (i + 1) // 2 + j

    def __getitem__(self, idx):
        return self.data[self.__class__.find_index(*idx)]

    def __setitem__(self, idx, val):
        self.data[self.__class__.find_index(*idx)] = val

    def __repr__(self):
        return [(i, j, self[i, j]) for i in range(self.n) for j in range(i + 1)].__repr__()


# two-electron integrals
class VInt(TInt):
    def __init__(self, n):
        self.n = n
        m = n * (n + 1) // 2
        self.data = np.zeros((m * (m + 1) // 2, ))

    @staticmethod
    def find_index(i, j, k, l):
        if i < j:
            i, j = j, i
        if k < l:
            k, l = l, k
        p = TInt.find_index(i, j)
        q = TInt.find_index(k, l)
        if p < q:
            p, q = q, p
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
    with open(filename, 'r') as f:
        pars, ints = f.read().split('/')
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


class QCHamiltonian:

    def __init__(self, fcidump, point_group):
        self.fcidump = fcidump
        opts, (t, v, e) = read_fcidump('N2.STO3G.FCIDUMP')

        self.t = t
        self.v = v
        self.e = e

        self.n_sites = int(opts['norb'])

        self.n_electrons = int(opts['nelec'])
        self.target_s = Fraction(int(opts['ms2']), 2)
        self.target_spatial_sym = int(opts['isym'])

        self.spatial_syms = [int(i) - 1 for i in opts['isym'].split(',')]
        self.point_group = point_group
        self.PG = point_group(self.point_group)

        self.empty = ParticleN(0) * SU2(0) * self.PG(0)
        self.spatial = [self.PG.IrrepNames[ir] for ir in self.spatial_syms]
        self.site_basis = [[
            ParticleN(0) * SU2(0) * self.PG(0),
            ParticleN(1) * SU2(Fraction(1, 2)) * self.PG(sp),
            ParticleN(2) * SU2(0) * self.PG(0)
        ] for sp in self.spatial]
        self.target = ParticleN(self.n_electrons) \
            * SU2(self.target_s) * self.PG(self.target_spatial_sym)

    # q_labels = (ket, operator, bra)
    # quantum number: ket + operator = bra
    def operator_cre(self, i_site):
        repr = np.array([[0, 0, 0], [1, 0, 0], [0, np.sqrt(2), 0]], dtype=float)
        return Tensor.operator_init(self.site_basis[i_site], [repr], [self.site_basis[i_site][1]])
    
    def operator_des(self, i_site):
        return self.operator_cre(i_site).T
    
    def operator_identity(self, i_site):
        return Tensor.operator_init(self.site_basis[i_site], [np.identity(3)], [self.site_basis[i_site][0]])


if __name__ == "__main__":
    opts, (t, v, e) = read_fcidump('N2.STO3G.FCIDUMP')
    print(e)
    print(v)
