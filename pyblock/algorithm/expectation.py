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
    
    Attributes:
        n_sites : int
            Number of sites/orbitals
        dot : int
            Two-dot (2) or one-dot (1) scheme.
        bond_dims : list(int) or int
            Bond dimension for each sweep.
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
    
    def update_one_dot(self, i):
        """
        Update local sites in one-dot scheme.
        
        Args:
            i : int
                Site index of left dot
        
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
        
        return ctr.expect(h_eff, psi_bra, psi_ket)
    
    def update_two_dot(self, i):
        """
        Update local sites in two-dot scheme.
        
        Args:
            i : int
                Site index of left dot
        
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
        
        return ctr.expect(h_eff, psi_bra, psi_ket)
    
    def solve(self):
        """
        Calculate expectation value.
        
        Returns:
            expect : float
                Expectation value.
        """
        self._pre_sweep()
        
        self.construct_envs()
        
        if self.dot == 2:
            return self.update_two_dot(self.center)
        else:
            return self.update_one_dot(self.center)
        
        self._post_sweep()
