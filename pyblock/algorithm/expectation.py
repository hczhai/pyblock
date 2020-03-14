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
#

"""
Expectation calculation for <MPS1|MPO|MPS2>.
"""

from ..tensor.tensor import Tensor, TensorNetwork
from .dmrg import MovingEnvironment
import time
from mpi4py import MPI
import numpy as np

class ExpectationError(Exception):
    pass

def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)
        
class Expect:
    """
    Calculation of expectation value <MPS1|MPO|MPS2>.
    The expectation value can be evaluated at current canonical form, when `forward is None`.
    Otherwise, it is recommended that `bond_dim` (bond dimension) of the MPS is given, and
    one sweep will be performed and expectation value will be evaluated at each canonical form.
    The sweep will thus change the canonical form of MPS and MPSInfo in contractor.
    Therefore, it is recommended that a copy of MPS and MPSInfo is used here.
    
    Attributes:
        n_sites : int
            Number of sites/orbitals
        dot : int
            Two-dot (2) or one-dot (1) scheme.
    """
    def __init__(self, mpo, bra_mps, ket_mps, bra_canonical_form=None, ket_canonical_form=None, contractor=None):
        self.n_sites = len(mpo)
        self.dot = bra_mps.dot
        self.center = bra_mps.center
        assert bra_mps.dot == ket_mps.dot
        assert bra_mps.center == ket_mps.center

        self._k = ket_mps.deep_copy().add_tags({'_KET'})
        self._b = bra_mps.deep_copy().add_tags({'_BRA'})
        self._h = mpo.copy().add_tags({'_HAM'})

        self._h.set_contractor(contractor)

        if bra_canonical_form is None:
            self.bra_canonical_form  = ['L'] * min(self.center, self.n_sites)
            self.bra_canonical_form += ['C'] * self.dot
            self.bra_canonical_form += ['R'] * max(self.n_sites - self.center - self.dot, 0)
        else:
            self.bra_canonical_form = bra_canonical_form
        
        if ket_canonical_form is not None:
            self.ket_canonical_form = ket_canonical_form
        else:
            self.ket_canonical_form = self.bra_canonical_form.copy()
        
        assert self.ket_canonical_form == self.bra_canonical_form
        
        self._pre_sweep = contractor.pre_sweep if contractor is not None else lambda: None
        self._post_sweep = contractor.post_sweep if contractor is not None else lambda: None
    
    def construct_envs(self):
        t = time.perf_counter()
        self.eff_ham = MovingEnvironment(self.n_sites, self.center, self.dot, self._b | self._h | self._k, iprint=False)
    
    def update_one_dot(self, i, forward, bond_dim):
        """
        Update local sites in one-dot scheme.
        
        Args:
            i : int
                Site index of left dot
            forward : bool or None
                Direction of current sweep. If True, sweep is performed from left to right.
                If None, no sweep is performed (local evaluation).
            bond_dim : int
                Bond dimension of current sweep.
        
        Returns:
            expect : float
                Expectation value.
        """
        ctr = self._h[{i, '_HAM'}].contractor
        
        if ctr is None:
            raise ExpectationError('Need a contractor for updating local site!')
        
        fuse_left = i <= self.n_sites // 2
        fuse_tags = {'_FUSE_L'} if fuse_left else {'_FUSE_R'}
        
        for kb, tag, cf in [(self._k, '_KET', self.ket_canonical_form), (self._b, '_BRA', self.bra_canonical_form)]:
            kb[{i, tag}].contractor = ctr
            if fuse_left:
                ctr.fuse_left(i, kb[{i, tag}], cf[i])
            else:
                ctr.fuse_right(i, kb[{i, tag}], cf[i])
        
        self.eff_ham()[{i, '_HAM'}].tags |= fuse_tags
        h_eff = (self.eff_ham() ^ '_HAM')['_HAM']
        h_eff.tags |= fuse_tags
        psi_bra = self._b[{i, '_BRA'}]
        psi_ket = self._k[{i, '_KET'}]
        
        result = ctr.expect(h_eff, psi_bra, psi_ket)
        
        if len(result) == 1 and repr(list(result.keys())[0]) == 'H':
            result = list(result.values())[0]
        
        if forward is None:
            return result
        
        if not fuse_left and forward:
            psi_bra = ctr.unfuse_right(i, psi_bra.add_tags({'_BRA'})).add_tags({'_BRA'})
            ctr.fuse_left(i, psi_bra, self.bra_canonical_form[i])
            psi_ket = ctr.unfuse_right(i, psi_ket.add_tags({'_KET'})).add_tags({'_KET'})
            ctr.fuse_left(i, psi_ket, self.ket_canonical_form[i])
        elif fuse_left and not forward:
            psi_bra = ctr.unfuse_left(i, psi_bra.add_tags({'_BRA'})).add_tags({'_BRA'})
            ctr.fuse_right(i, psi_bra, self.bra_canonical_form[i])
            psi_ket = ctr.unfuse_left(i, psi_ket.add_tags({'_KET'})).add_tags({'_KET'})
            ctr.fuse_right(i, psi_ket, self.ket_canonical_form[i])

        if bond_dim == -1:
            if forward:
                bra_limit = ctr.bond_left({'_BRA'})[i]
                ket_limit = ctr.bond_left({'_KET'})[i]
            else:
                bra_limit = ctr.bond_right({'_BRA'})[i]
                ket_limit = ctr.bond_right({'_KET'})[i]
        else:
            if forward:
                bra_limit = ctr.bond_upper_limit_left({'_BRA'})[i]
                ket_limit = ctr.bond_upper_limit_left({'_KET'})[i]
            else:
                bra_limit = ctr.bond_upper_limit_right({'_BRA'})[i]
                ket_limit = ctr.bond_upper_limit_right({'_KET'})[i]

        dm_ket = psi_ket.get_diag_density_matrix(trace_right=forward)
        l_fused_ket, r_fused_ket, error_ket = \
            psi_ket.split_using_density_matrix(dm_ket, absorb_right=forward, k=bond_dim, limit=ket_limit)
        
        dm_bra = psi_bra.get_diag_density_matrix(trace_right=forward)
        l_fused_bra, r_fused_bra, error_bra = \
            psi_bra.split_using_density_matrix(dm_bra, absorb_right=forward, k=bond_dim, limit=bra_limit)
        
        TensorNetwork(tensors=[l_fused_bra, r_fused_bra]).add_tags({'_BRA'})
        TensorNetwork(tensors=[l_fused_ket, r_fused_ket]).add_tags({'_KET'})
        
        if forward:
            ctr.update_local_left_mps_info(i, l_fused_bra)
            ctr.update_local_left_mps_info(i, l_fused_ket)
        else:
            ctr.update_local_right_mps_info(i, r_fused_bra)
            ctr.update_local_right_mps_info(i, r_fused_ket)
        
        if forward:
            l_bra = ctr.unfuse_left(i, l_fused_bra)
            l_ket = ctr.unfuse_left(i, l_fused_ket)
            if i + 1 < self.n_sites:
                self.bra_canonical_form[i:i + 2] = self.ket_canonical_form[i:i + 2] = "LC"
                adj_bra = Tensor.contract(r_fused_bra, self._b[{i + 1, '_BRA'}], [1], [0])
                adj_ket = Tensor.contract(r_fused_ket, self._k[{i + 1, '_KET'}], [1], [0])
                self._b[{i + 1, '_BRA'}].modify(adj_bra)
                self._k[{i + 1, '_KET'}].modify(adj_ket)
                self.eff_ham.envs[self.eff_ham.pos + 1][{i + 1, '_BRA'}].modify(adj_bra)
                self.eff_ham.envs[self.eff_ham.pos + 1][{i + 1, '_KET'}].modify(adj_ket)
            else:
                l_bra *= r_fused_bra.to_scalar()
                l_ket *= r_fused_ket.to_scalar()
                self.bra_canonical_form[i] = self.ket_canonical_form[i] = 'K'
        else:
            r_bra = ctr.unfuse_right(i, r_fused_bra)
            r_ket = ctr.unfuse_right(i, r_fused_ket)
            if i - 1 >= 0:
                self.bra_canonical_form[i - 1:i + 1] = self.ket_canonical_form[i - 1:i + 1] = "CR"
                adj_bra = Tensor.contract(self._b[{i - 1, '_BRA'}], l_fused_bra, [2], [0])
                adj_ket = Tensor.contract(self._k[{i - 1, '_KET'}], l_fused_ket, [2], [0])
                self._b[{i - 1, '_BRA'}].modify(adj_bra)
                self._k[{i - 1, '_KET'}].modify(adj_ket)
                self.eff_ham.envs[self.eff_ham.pos - 1][{i - 1, '_BRA'}].modify(adj_bra)
                self.eff_ham.envs[self.eff_ham.pos - 1][{i - 1, '_KET'}].modify(adj_ket)
            else:
                r_bra *= l_fused_bra.to_scalar()
                r_ket *= l_fused_ket.to_scalar()
                self.bra_canonical_form[i] = self.ket_canonical_form[i] = 'S'

        self.eff_ham()[{i, '_HAM'} | fuse_tags].tags -= fuse_tags
        self._b[{i, '_BRA'}].modify(l_bra if forward else r_bra)
        self._k[{i, '_KET'}].modify(l_ket if forward else r_ket)
        self.eff_ham()[{i, '_BRA'}].modify(l_bra if forward else r_bra)
        self.eff_ham()[{i, '_KET'}].modify(l_ket if forward else r_ket)

        return result

    def update_two_dot(self, i, forward, bond_dim):
        """
        Update local sites in two-dot scheme.
        
        Args:
            i : int
                Site index of left dot
            forward : bool or None
                Direction of current sweep. If True, sweep is performed from left to right.
                If None, no sweep is performed (local evaluation).
            bond_dim : int
                Bond dimension of current sweep.
        
        Returns:
            expect : float
                Expectation value.
        """
        
        ctr = self._h[{i, '_HAM'}].contractor
        
        if ctr is None:
            raise CompressionError('Need a contractor for updating local site!')
        
        for kb, tag, cf in [(self._k, '_KET', self.ket_canonical_form), (self._b, '_BRA', self.bra_canonical_form)]:
            if len(kb.select({i, i + 1, tag}, which='exact')) == 0:
                kb[{i, tag}].contractor = ctr
                ctr.fuse_left(i, kb[{i, tag}], cf[i])
                ctr.fuse_right(i + 1, kb[{i + 1, tag}], cf[i + 1])
                two_site = kb.select({i, i + 1}, which='any') ^ (tag, i, i + 1)
                kb.replace({i, i + 1}, two_site, which='any')
                [self.eff_ham().remove({j, tag}, in_place=True) for j in [i, i + 1]]
                self.eff_ham().add(two_site)

        h_eff = (self.eff_ham() ^ '_HAM')['_HAM']
        psi_bra = self.eff_ham()[{i, i + 1, '_BRA'}]
        psi_ket = self.eff_ham()[{i, i + 1, '_KET'}]
        
        result = ctr.expect(h_eff, psi_bra, psi_ket)
        
        if len(result) == 1 and repr(list(result.keys())[0]) == 'H':
            result = list(result.values())[0]
        
        if forward is None:
            return result
        
        if bond_dim == -1:
            if forward:
                bra_limit = ctr.bond_left({'_BRA'})[i]
                ket_limit = ctr.bond_left({'_KET'})[i]
            else:
                bra_limit = ctr.bond_right({'_BRA'})[i + 1]
                ket_limit = ctr.bond_right({'_KET'})[i + 1]
        else:
            if forward:
                bra_limit = ctr.bond_upper_limit_left({'_BRA'})[i]
                ket_limit = ctr.bond_upper_limit_left({'_KET'})[i]
            else:
                bra_limit = ctr.bond_upper_limit_right({'_BRA'})[i + 1]
                ket_limit = ctr.bond_upper_limit_right({'_KET'})[i + 1]
        
        dm = psi_bra.get_diag_density_matrix(trace_right=forward)
        
        l_fused_bra, r_fused_bra, error_bra = \
            psi_bra.split_using_density_matrix(dm, absorb_right=forward, k=bond_dim, limit=bra_limit)
        
        dm = psi_ket.get_diag_density_matrix(trace_right=forward)
        
        l_fused_ket, r_fused_ket, error_ket = \
            psi_ket.split_using_density_matrix(dm, absorb_right=forward, k=bond_dim, limit=ket_limit)
        
        assert np.isclose(error_bra, 0) and np.isclose(error_ket, 0)
        
        TensorNetwork(tensors=[l_fused_bra, r_fused_bra]).add_tags({'_BRA'})
        TensorNetwork(tensors=[l_fused_ket, r_fused_ket]).add_tags({'_KET'})
        
        if forward:
            ctr.update_local_left_mps_info(i, l_fused_bra)
            ctr.update_local_left_mps_info(i, l_fused_ket)
        else:
            ctr.update_local_right_mps_info(i + 1, r_fused_bra)
            ctr.update_local_right_mps_info(i + 1, r_fused_ket)
        
        l_bra = ctr.unfuse_left(i, l_fused_bra)
        r_bra = ctr.unfuse_right(i + 1, r_fused_bra)
        l_ket = ctr.unfuse_left(i, l_fused_ket)
        r_ket = ctr.unfuse_right(i + 1, r_fused_ket)
        
        self.ket_canonical_form[i] = self.bra_canonical_form[i] = 'L' if forward else 'C'
        self.ket_canonical_form[i + 1] = self.bra_canonical_form[i + 1] =  'R' if not forward else 'C'
            
        tn_lr = TensorNetwork(tensors=[l_bra, r_bra])
        tn_lr_bra = tn_lr.copy().add_tags({'_BRA'})
        tn_lr = TensorNetwork(tensors=[l_ket, r_ket])
        tn_lr_ket = tn_lr.copy().add_tags({'_KET'})
        
        self._k.replace({i, i + 1}, tn_lr_ket)
        self._b.replace({i, i + 1}, tn_lr_bra)
        self.eff_ham().replace({i, i + 1}, tn_lr_ket | tn_lr_bra)
        
        return result
    
    def blocking(self, i, forward, bond_dim):
        """
        Perform one blocking iteration.
        
        Args:
            i : int
                Site index of left dot
            forward : bool or None
                Direction of current sweep. If True, sweep is performed from left to right.
                If None, no sweep is performed (local evaluation).
            bond_dim : int
                Bond dimension of current sweep.
        
        Returns:
            result : float
                Expectation value.
        """
        self.eff_ham.move_to(i)
        self.center = i

        if self.dot == 1:
            return self.update_one_dot(i, forward, bond_dim)
        else:
            return self.update_two_dot(i, forward, bond_dim)
    
    def solve(self, forward=None, bond_dim=-1):
        """
        Calculate expectation value.
        
        Args:
            forward : bool or None
                Direction of current sweep. If True, sweep is performed from left to right.
                If None, no sweep is performed (local evaluation).
            bond_dim : int
                Bond dimension of current sweep.
        
        Returns:
            expect : float
                Expectation value.
        """
        self._pre_sweep()

        self.construct_envs()

        if forward is None:
            sweep_range = [self.center]
        elif forward:
            sweep_range = range(self.center, self.n_sites - self.dot + 1)
        else:
            sweep_range = range(self.center, -1, -1)

        self.results = []

        for i in sweep_range:
            if self.dot == 2:
                pprint(" %3s Site = %4d-%4d .. " % ('-->' if forward else '<--', i, i + 1), end='', flush=True)
            else:
                pprint(" %3s Site = %4d .. " % ('-->' if forward else '<--', i), end='', flush=True)
            t = time.perf_counter()
            result = self.blocking(i, forward=forward, bond_dim=bond_dim)
            
            if isinstance(result, dict):
                pprint("Nterms = %4d T = %4.2f" % (len(result), time.perf_counter() - t))
            else:
                pprint("Result = %15.8f T = %4.2f" % (result, time.perf_counter() - t))
            
            self.results.append(result)

        self._post_sweep()

        return result
    
    def get_1pdm(self):
        """
        Spatial 1-particle density matrix.
        """
        pdmat = np.zeros((self.n_sites, self.n_sites))
        assert hasattr(self, "results")
        for r in self.results:
            for k, v in r.items():
                pdmat[k.site_index[0], k.site_index[1]] = v
        return pdmat
