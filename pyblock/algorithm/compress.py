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
#    Author: Seunghoon Lee, Huanchen Zhai
#

"""
Compression algorithm.
"""

from ..tensor.tensor import Tensor, TensorNetwork
from .dmrg import MovingEnvironment
import time
from mpi4py import MPI
import numpy as np

class CompressionError(Exception):
    pass

def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)
        
class Compress:
    """
    Compression after apply MPO on MPS.
    
    Attributes:
        n_sites : int
            Number of sites/orbitals
        dot : int
            Two-dot (2) or one-dot (1) scheme.
        bond_dims : list(int) or int
            Bond dimension for each sweep.
    """
    def __init__(self, mpo, mps, ket_mps, bond_dims, noise, ket_canonical_form=None, contractor=None):
        self.n_sites = len(mpo)
        self.dot = mps.dot
        self.center = mps.center
        assert mps.dot == ket_mps.dot
        assert mps.center == ket_mps.center
        self.bond_dims = bond_dims if isinstance(bond_dims, list) else [bond_dims]
        self.noise = noise if isinstance(noise, list) else [noise]
        
        self.mps = mps.deep_copy()
        
        self._k = ket_mps.deep_copy().add_tags({'_KET'}) # const MPS
        self._b = mps.deep_copy().add_tags({'_BRA'}) # target MPS
        self._h = mpo.copy().add_tags({'_HAM'})
        
        self._h.set_contractor(contractor)
        
        self.bra_canonical_form  = ['L'] * min(self.center, self.n_sites)
        self.bra_canonical_form += ['C'] * self.dot
        self.bra_canonical_form += ['R'] * max(self.n_sites - self.center - self.dot, 0)
        
        if ket_canonical_form is not None:
            self.ket_canonical_form = ket_canonical_form
        else:
            self.ket_canonical_form = self.bra_canonical_form.copy()

        self.energies = []
        
        self._pre_sweep = contractor.pre_sweep if contractor is not None else lambda: None
        self._post_sweep = contractor.post_sweep if contractor is not None else lambda: None
        
        self.beta = 0

        self.rebuild = contractor.rebuild
        
        if not self.rebuild:
            self.construct_envs()
    
    def set_mps(self, tags, wfn):
        self.mps = self._b.deep_copy()
        self.mps.center = self.center
        self.mps.dot = self.dot
        self.mps.replace(tags, wfn.deep_copy().set_tags(tags), which='any')
        self.mps.remove_tags({'_BRA'})
        self.mps.form = self.bra_canonical_form.copy()
        self.mps.set_contractor(None)
    
    def update_one_dot(self, i, forward, bond_dim, noise, beta):
        """
        Update local sites in one-dot scheme.
        
        Args:
            i : int
                Site index of left dot
            forward : bool
                Direction of current sweep. If True, sweep is performed from left to right.
            bond_dim : int
                Bond dimension of current sweep.
            beta : float
                Not used.
        
        Returns:
            norm : float
                Norm of compressed state.
            error : float
                Sum of discarded weights.
            nexpos : (int, int)
                Number of operator multiplication steps.
        """
        ctr = self._h[{i, '_HAM'}].contractor
        
        if ctr is None:
            raise CompressionError('Need a contractor for updating local site!')
        
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
        psi_ket = self._k[{i, '_KET'}]
        energy, psi_bra, nexpo = ctr.apply(h_eff, psi_ket)

        if not fuse_left and forward:
            psi_bra = ctr.unfuse_right(i, psi_bra.add_tags({'_BRA'})).add_tags({'_BRA'})
            self.set_mps({i}, psi_bra)
            ctr.fuse_left(i, psi_bra, self.bra_canonical_form[i])
            psi_ket = ctr.unfuse_right(i, psi_ket.add_tags({'_KET'})).add_tags({'_KET'})
            ctr.fuse_left(i, psi_ket, self.ket_canonical_form[i])
        elif fuse_left and not forward:
            psi_bra = ctr.unfuse_left(i, psi_bra.add_tags({'_BRA'})).add_tags({'_BRA'})
            self.set_mps({i}, psi_bra)
            ctr.fuse_right(i, psi_bra, self.bra_canonical_form[i])
            psi_ket = ctr.unfuse_left(i, psi_ket.add_tags({'_KET'})).add_tags({'_KET'})
            ctr.fuse_right(i, psi_ket, self.ket_canonical_form[i])
        
        if forward:
            ket_limit = ctr.bond_left({'_KET'})[i]
            bra_limit = ctr.bond_upper_limit_left({'_BRA'})[i]
        else:
            ket_limit = ctr.bond_right({'_KET'})[i]
            bra_limit = ctr.bond_upper_limit_right({'_BRA'})[i]
        
        dm_ket = psi_ket.get_diag_density_matrix(trace_right=forward)
        l_fused_ket, r_fused_ket, error_ket = \
            psi_ket.split_using_density_matrix(dm_ket, absorb_right=forward, k=-1, limit=ket_limit)
        assert np.isclose(error_ket, 0.0)
        
        dm_bra = psi_bra.get_diag_density_matrix(trace_right=forward, noise=noise)
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
        
        return energy, error_bra, nexpo
    
    def update_two_dot(self, i, forward, bond_dim, noise, beta):
        """
        Update local sites in two-dot scheme.
        
        Args:
            i : int
                Site index of left dot
            forward : bool
                Direction of current sweep. If True, sweep is performed from left to right.
            bond_dim : int
                Bond dimension of current sweep.
            beta : float
                Not used.
        
        Returns:
            norm : float
                Norm of compressed state.
            error : float
                Sum of discarded weights.
            nexpos : (int, int)
                Number of operator multiplication steps.
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
        psi_ket = self.eff_ham()[{i, i + 1, '_KET'}]
        energy, psi_bra, nexpo = ctr.apply(h_eff, psi_ket)
        
        self.set_mps({i, i + 1}, psi_bra)
        
        if forward:
            ket_limit = ctr.bond_left({'_KET'})[i]
            bra_limit = ctr.bond_upper_limit_left({'_BRA'})[i]
        else:
            ket_limit = ctr.bond_right({'_KET'})[i + 1]
            bra_limit = ctr.bond_upper_limit_right({'_BRA'})[i + 1]
        
        dm_ket = psi_ket.get_diag_density_matrix(trace_right=forward)
        l_fused_ket, r_fused_ket, error_ket = \
            psi_ket.split_using_density_matrix(dm_ket, absorb_right=forward, k=-1, limit=ket_limit)
        assert np.isclose(error_ket, 0.0)
        
        dm_bra = psi_bra.get_diag_density_matrix(trace_right=forward, noise=noise)
        l_fused_bra, r_fused_bra, error_bra = \
            psi_bra.split_using_density_matrix(dm_bra, absorb_right=forward, k=bond_dim, limit=bra_limit)
        
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

        self.bra_canonical_form[i] = self.ket_canonical_form[i] = 'L' if forward else 'C'
        self.bra_canonical_form[i + 1] = self.ket_canonical_form[i + 1] = 'R' if not forward else 'C'
        
        tn_lr = TensorNetwork(tensors=[l_ket, r_ket])
        tn_lr_ket = tn_lr.copy().add_tags({'_KET'})
        tn_lr = TensorNetwork(tensors=[l_bra, r_bra])
        tn_lr_bra = tn_lr.copy().add_tags({'_BRA'})
        
        self._k.replace({i, i + 1}, tn_lr_ket)
        self._b.replace({i, i + 1}, tn_lr_bra)
        self.eff_ham().replace({i, i + 1}, tn_lr_ket | tn_lr_bra)
        
        return energy, error_bra, nexpo
    
    def construct_envs(self):
        t = time.perf_counter()
        pprint(" Constructing environment .. ", end='', flush=True)
        self.eff_ham = MovingEnvironment(self.n_sites, self.center, self.dot, self._b | self._h | self._k)
        pprint("T = %4.2f" % (time.perf_counter() - t))

    def blocking(self, i, forward, bond_dim, noise, beta):
        """
        Perform one blocking iteration.
        
        Args:
            i : int
                Site index of left dot
            forward : bool
                Direction of current sweep. If True, sweep is performed from left to right.
            bond_dim : int
                Bond dimension of current sweep.
            noise : float
                Noise prefactor of current sweep.
            beta : float
                Not used.
        
        Returns:
            norm : float
                Norm of compressed state.
            error : float
                Sum of discarded weights.
            nexpo : int
                Number of operator multiplication steps.
        """
        self.eff_ham.move_to(i)
        self.center = i

        if self.dot == 1:
            return self.update_one_dot(i, forward, bond_dim, noise, beta)
        else:
            return self.update_two_dot(i, forward, bond_dim, noise, beta)

    def sweep(self, forward, bond_dim, noise, beta):
        """
        Perform one sweep iteration.
        
        Args:
            forward : bool
                Direction of current sweep. If True, sweep is performed from left to right.
            bond_dims : int
                Bond dimension of current sweep.
            noise : float
                Noise prefactor of current sweep.
            beta : float
                Not used.
        
        Returns:
            norm : float
                Norm of compressed state.
        """
        self._pre_sweep()
        
        if self.rebuild:
            self.construct_envs()
        else:
            self.eff_ham.prepare_sweep(self.dot, self.center)

        # if forward/backward, i is the first dot site

        if forward:
            sweep_range = range(self.center, self.n_sites - self.dot + 1)
        else:
            sweep_range = range(self.center, -1, -1)
        
        sweep_results = []

        for i in sweep_range:
            if self.dot == 2:
                pprint(" %3s Site = %4d-%4d .. " % ('-->' if forward else '<--', i, i + 1), end='', flush=True)
            else:
                pprint(" %3s Site = %4d .. " % ('-->' if forward else '<--', i), end='', flush=True)
            t = time.perf_counter()
            result, error, nmult = self.blocking(i, forward=forward, bond_dim=bond_dim, noise=noise, beta=beta)
            pprint("Nmult = %4d N = %15.8f Error = %15.8f T = %4.2f" % (nmult, result, error, time.perf_counter() - t))
            sweep_results.append(result)
        
        self._post_sweep()

        return sweep_results[-1]

    def solve(self, n_sweeps, tol, forward=True, two_dot_to_one_dot=-1):
        """
        Perform Compression algorithm.
        
        Args:
            n_sweeps : int
                Maximum number of sweeps.
            tol : float
                Norm convergence threshold.
            forward : bool
                Direction of first sweep. If True, sweep is performed from left to right.
            two_dot_to_one_dot : int or -1
                Indicating when to switch to one-dot scheme. If -1, no switching.
        
        Returns:
            nrom : float
                Final compressed stae norm.
        """
        converged = False

        if len(self.bond_dims) < n_sweeps:
            self.bond_dims.extend([self.bond_dims[-1]] * (n_sweeps - len(self.bond_dims)))
            
        if len(self.noise) < n_sweeps:
            self.noise.extend([self.noise[-1]] * (n_sweeps - len(self.noise)))
        
        start = time.perf_counter()
        
        for iw in range(n_sweeps):

            pprint("Sweep = %4d | Direction = %8s | Bond dimension = %4d | Noise = %9.2g | Beta = %9.2g"
                % (iw, "forward" if forward else "backward", self.bond_dims[iw], self.noise[iw], self.beta))
            
            if two_dot_to_one_dot == iw:
                assert self.dot == 2
                self.dot = 1
                if self.center != 0 and self.center == self.n_sites - 2:
                    self.center = self.n_sites - 1

            energy = self.sweep(forward=forward, bond_dim=self.bond_dims[iw], noise=self.noise[iw], beta=self.beta)
            self.energies.append(energy)

            converged = (len(self.energies) >= 2 and tol is not None
                and abs(self.energies[-1] - self.energies[-2]) < tol
                and self.noise[iw] == self.noise[-1] and self.bond_dims[iw] == self.bond_dims[-1])

            forward = not forward
            
            pprint('Time elapsed = %10.2f' % (time.perf_counter() - start))
            
            if converged:
                break
        
        self.forward = forward
        return energy
