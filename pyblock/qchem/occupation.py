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
Initialize quantum numbers using occupation numbers.
"""

class Occupation:
    def __init__(self, occ, n_sites, basis, empty, target, bias=1):
        self.occ = occ
        self.n_sites = n_sites
        self.basis = [sorted(b.items()) if isinstance(b, dict) else b for b in basis]
        self.empty = empty
        self.target = target
        self.bias = bias
    
    def _get_site_states(self, m, site_occ):
        if self.bias != 1:
            if site_occ > 1:
                site_occ = 1 + (site_occ - 1) ** self.bias
            elif site_occ < 1:
                site_occ = 1 - (1 - site_occ) ** self.bias
        alpha_occ = site_occ / 2
        assert 0 <= alpha_occ <= 1
        rr = []
        for k, v in self.basis[m]:
            if k.irs[0].ir == 0:
                r = (1 - alpha_occ) ** 2
            elif k.irs[0].ir == 1:
                # no need to * 2 here and the sum is not 1
                # since this is doublet
                r = (1 - alpha_occ) * alpha_occ
            else:
                r = alpha_occ ** 2
            rr.append((k, r))
        return rr
    
    def set_bond_dimension(self, fci_l, fci_r, m):
        occ_l = [None] * self.n_sites
        occ_r = [None] * self.n_sites
        self._fill_occ_from_left(occ_l)
        self._fill_occ_from_right(occ_r)
        self._occ_filter(occ_l, occ_r)
        dim_l = [None] * self.n_sites
        dim_r = [None] * self.n_sites
        for d in range(0, self.n_sites):
            sum_l = sum([v for k, v in occ_l[d]])
            dim_l[d] = [(k, min(fci_l[d][k], int(round(v / sum_l * m) + 0.1))) for k, v in occ_l[d]]
            dim_l[d] = [(k, v) for k, v in dim_l[d] if v != 0]
            if d != self.n_sites - 1:
                dd = dict(self.tensor_product(dim_l[d], self.basis[d + 1]))
                for k, v in fci_l[d + 1].copy().items():
                    if k in dd:
                        fci_l[d + 1][k] = min(dd[k], v)
                for ik, (k, _) in enumerate(occ_l[d + 1]):
                    if k not in dd:
                        occ_l[d + 1][ik] = (k, 0)
        for d in range(self.n_sites - 1, -1, -1):
            sum_r = sum([v for k, v in occ_r[d]])
            dim_r[d] = [(k, min(fci_r[d][k], int(round(v / sum_r * m) + 0.1))) for k, v in occ_r[d]]
            dim_r[d] = [(k, v) for k, v in dim_r[d] if v != 0]
            if d != 0:
                dd = dict(self.tensor_product(self.basis[d - 1], dim_r[d]))
                for k, v in fci_r[d - 1].copy().items():
                    if k in dd:
                        fci_r[d - 1][k] = min(dd[k], v)
                for ik, (k, _) in enumerate(occ_r[d - 1]):
                    if k not in dd:
                        occ_r[d - 1][ik] = (k, 0)
        return list(map(dict, dim_l)), list(map(dict, dim_r))

    def _fill_occ_from_left(self, dim_l):
        for d in range(0, self.n_sites):
            if d == 0:
                dim_l[d] = self.tensor_product(None, self._get_site_states(d, self.occ[d]))
            else:
                dim_l[d] = self.tensor_product(dim_l[d - 1], self._get_site_states(d, self.occ[d]))
    
    def _fill_occ_from_right(self, dim_r):
        for d in range(self.n_sites - 1, -1, -1):
            if d == self.n_sites - 1:
                dim_r[d] = self.tensor_product(self._get_site_states(d, self.occ[d]), None)
            else:
                dim_r[d] = self.tensor_product(self._get_site_states(d, self.occ[d]), dim_r[d + 1])
    
    def tensor_product(self, p, q):
        rr = {}
        for pk, pv in ([(self.empty, 1)] if p is None else p):
            for rks, rv in ((pk + qk, pv * qv) for qk, qv in ([(self.empty, 1)] if q is None else q)):
                for rk in rks if isinstance(rks, list) else [rks]:
                    if self.target is None or rk <= self.target:
                        rr[rk] = rr.get(rk, 0) + rv
        return sorted(rr.items())

    def _occ_filter(self, left_occs, right_occs):
        for d in range(-1, self.n_sites):
            ld = left_occs[d] if d >= 0 else [(self.empty, 1)]
            rd = right_occs[d + 1] if d + 1 < self.n_sites else [(self.empty, 1)]
            rdd = { k: v * k.multiplicity for k, v in rd }
            ldd = { k: v * k.multiplicity for k, v in ld }
            new_ld = []
            for k, v in ld:
                rk = self.target - k
                x = sum((rdd[r] if r in rdd else 0) for r in (rk if isinstance(rk, list) else [rk]))
                if x != 0:
                    new_ld.append((k, v * x))
            new_rd = []
            for k, v in rd:
                lk = self.target - k
                x = sum((ldd[l] if l in ldd else 0) for l in (lk if isinstance(lk, list) else [lk]))
                if x != 0:
                    new_rd.append((k, v * x))
            if d >= 0:
                left_occs[d] = new_ld
            if d + 1 < self.n_sites:
                right_occs[d + 1] = new_rd
