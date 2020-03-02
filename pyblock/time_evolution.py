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
Imaginary time evolution algorithm.
"""

from .tensor.tensor import Tensor, TensorNetwork
from .dmrg import MovingEnvironment
import time
from mpi4py import MPI

class TEError(Exception):
    pass

def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)

class ExpoApply:
    """
    Apply exp(beta H) to MPS.
    
    Attributes:
        n_sites : int
            Number of sites/orbitals
        dot : int
            Two-dot (2) or one-dot (1) scheme.
        bond_dims : list(int) or int
            Bond dimension for each sweep.
        energies : list(float)
            Energies collected for all sweeps.
    """
    def __init__(self, mpo, mps, beta, bond_dim, contractor=None):
        self.n_sites = len(mpo)
        self.dot = mps.dot
        self.center = mps.center
        self.bond_dims = bond_dim if isinstance(bond_dim, list) else [bond_dim]

        self._k = mps.deep_copy().add_tags({'_KET'})
        self._b = mps.deep_copy().add_tags({'_BRA'})
        self._h = mpo.copy().add_tags({'_HAM'})
        
        self._h.set_contractor(contractor)
        self._k.set_contractor(contractor)
        
        self.canonical_form  = ['L'] * min(self.center, self.n_sites)
        self.canonical_form += ['C'] * self.dot
        self.canonical_form += ['R'] * max(self.n_sites - self.center - self.dot, 0)

        self.energies = []
        
        self._pre_sweep = contractor.pre_sweep if contractor is not None else lambda: None
        self._post_sweep = contractor.post_sweep if contractor is not None else lambda: None
        
        self.beta = beta

        self.rebuild = contractor.rebuild
        
        if not self.rebuild:
            self.construct_envs()
    
    def update_one_dot(self, i, forward, bond_dim, beta):
        """
        Update local site in one-dot scheme.
        
        Args:
            i : int
                Site index of left dot
            forward : bool
                Direction of current sweep. If True, sweep is performed from left to right.
            bond_dim : int
                Bond dimension of current sweep.
            beta : float
                Time step.
        
        Returns:
            energy : float
                Ground state energy.
            error : float
                Sum of discarded weights.
            nexpos : (int, int)
                Number of Expokit iterations.
        """
        
        ctr = self._h[{i, '_HAM'}].contractor
        
        if ctr is None:
            raise DMRGError('Need a contractor for updating local site!')
        
        fuse_left = i <= self.n_sites // 2
            
        self._k[{i, '_KET'}].contractor = ctr
        fuse_tags = {'_FUSE_L'} if fuse_left else {'_FUSE_R'}
        
        if fuse_left:
            ctr.fuse_left(i, self._k[{i, '_KET'}], self.canonical_form[i])
        else:
            ctr.fuse_right(i, self._k[{i, '_KET'}], self.canonical_form[i])
        
        self.eff_ham()[{i, '_HAM'}].tags |= fuse_tags
        h_eff = (self.eff_ham() ^ '_HAM')['_HAM']
        h_eff.tags |= fuse_tags
        gs_old = self._k[{i, '_KET'}]
        energy, gs, nexpo = ctr.expo(h_eff, gs_old, beta)
        
        if not fuse_left and forward:
            gs = ctr.unfuse_right(i, gs)
            ctr.fuse_left(i, gs, self.canonical_form[i])
        elif fuse_left and not forward:
            gs = ctr.unfuse_left(i, gs)
            ctr.fuse_right(i, gs, self.canonical_form[i])
        
        if forward:
            limit = ctr.bond_upper_limit_left[i]
        else:
            limit = ctr.bond_upper_limit_right[i]
        
        dm = gs.get_diag_density_matrix(trace_right=forward)
        
        l_fused, r_fused, error = \
            gs.split_using_density_matrix(dm, absorb_right=forward, k=bond_dim, limit=limit)
        
        if forward:
            ctr.update_local_left_mps_info(i, l_fused)
            gs_new = ctr.unfuse_left(i, l_fused)
        else:
            ctr.update_local_right_mps_info(i, r_fused)
            gs_new = ctr.unfuse_right(i, r_fused)
        
        self.eff_ham()[{i, '_HAM'} | fuse_tags].tags -= fuse_tags
        self._k[{i, '_KET'}].modify(gs_new)
        self._b[{i, '_BRA'}].modify(gs_new)
        self.eff_ham()[{i, '_KET'}].modify(gs_new)
        self.eff_ham()[{i, '_BRA'}].modify(gs_new)
 
        nexpok = 0
        if forward:
            if i + 1 < self.n_sites:
                self.canonical_form[i:i + 2] = "LC"
                k_tn = self.eff_ham.move_to(i + 1).add_tags({'_NO_FUSE'})
                k_eff = (k_tn ^ '_HAM')['_HAM']
                k_eff.tags |= {'_NO_FUSE'}
                r_fused.tags |= {i}
                _, r_back, nexpok = ctr.expo(k_eff, r_fused, -beta)
                k_tn.remove_tags({'_NO_FUSE'})
                adj_new = Tensor.contract(r_back, self._k[{i + 1, '_KET'}], [1], [0])
                self._k[{i + 1, '_KET'}].modify(adj_new)
                self._b[{i + 1, '_BRA'}].modify(adj_new)
                self.eff_ham()[{i + 1, '_KET'}].modify(adj_new)
                self.eff_ham()[{i + 1, '_BRA'}].modify(adj_new)
            else:
                self.canonical_form[i] = 'L'
        else:
            if i - 1 >= 0:
                self.canonical_form[i - 1:i + 1] = "CR"
                k_tn = self.eff_ham.move_to(i - 1).add_tags({'_NO_FUSE'})
                k_eff = (k_tn ^ '_HAM')['_HAM']
                k_eff.tags |= {'_NO_FUSE'}
                l_fused.tags |= {i - 1}
                _, l_back, nexpok = ctr.expo(k_eff, l_fused, -beta)
                k_tn.remove_tags({'_NO_FUSE'})
                adj_new = Tensor.contract(self._k[{i - 1, '_KET'}], l_back, [2], [0])
                self._k[{i - 1, '_KET'}].modify(adj_new)
                self._b[{i - 1, '_BRA'}].modify(adj_new)
                self.eff_ham()[{i - 1, '_KET'}].modify(adj_new)
                self.eff_ham()[{i - 1, '_BRA'}].modify(adj_new)
            else:
                self.canonical_form[i] = 'R'
        
        return energy, error, (nexpo, nexpok)
    
    def update_two_dot(self, i, forward, bond_dim, beta):
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
                Temperature step.
        
        Returns:
            energy : float
                Ground state energy.
            error : float
                Sum of discarded weights.
            nexpos : (int, int)
                Number of Expokit iterations.
        """
        
        ctr = self._h[{i, '_HAM'}].contractor
        
        if ctr is None:
            raise DMRGError('Need a contractor for updating local site!')
        
        if len(self._k.select({i, i + 1, '_KET'}, which='exact')) == 0:
            
            self._k[{i, '_KET'}].contractor = ctr
            ctr.fuse_left(i, self._k[{i, '_KET'}], self.canonical_form[i])
            ctr.fuse_right(i + 1, self._k[{i + 1, '_KET'}], self.canonical_form[i + 1])
            twod_ket = self._k.select({i, i + 1}, which='any') ^ ('_KET', i, i + 1)
            twod_bra = twod_ket.copy().remove_tags({'_KET'}).add_tags({'_BRA'})
            self._k.replace({i, i + 1}, twod_ket, which='any')
            self._b.replace({i, i + 1}, twod_bra, which='any')
            [self.eff_ham().remove({j, t}, in_place=True) for j in [i, i + 1] for t in ['_KET', '_BRA']]
            self.eff_ham().add(twod_ket | twod_bra)
        
        h_eff = (self.eff_ham() ^ '_HAM')['_HAM']
        gs_old = self.eff_ham()[{i, i + 1, '_KET'}]
        energy, gs, nexpo = ctr.expo(h_eff, gs_old, beta)
        
        if forward:
            limit = ctr.bond_upper_limit_left[i]
        else:
            limit = ctr.bond_upper_limit_right[i + 1]
        
        dm = gs.get_diag_density_matrix(trace_right=forward)
        
        l_fused, r_fused, error = \
            gs.split_using_density_matrix(dm, absorb_right=forward, k=bond_dim, limit=limit)
        
        if forward:
            ctr.update_local_left_mps_info(i, l_fused)
        else:
            ctr.update_local_right_mps_info(i + 1, r_fused)
        
        l = ctr.unfuse_left(i, l_fused)
        r = ctr.unfuse_right(i + 1, r_fused)

        self.canonical_form[i] = 'L' if forward else 'C'
        self.canonical_form[i + 1] = 'R' if not forward else 'C'

        tn_lr = TensorNetwork(tensors=[l, r])
        tn_lr_ket = tn_lr.copy().add_tags({'_KET'})
        tn_lr_bra = tn_lr.copy().add_tags({'_BRA'})
        
        self._k.replace({i, i + 1}, tn_lr_ket)
        self._b.replace({i, i + 1}, tn_lr_bra)
        self.eff_ham().replace({i, i + 1}, tn_lr_ket | tn_lr_bra)

        nexpok = 0
        if forward:
            if i + 1 != self.n_sites - 1:
                k_tn = self.eff_ham.move_to(i + 1).add_tags({'_FUSE_R'})
                k_eff = (k_tn ^ '_HAM')['_HAM']
                k_eff.tags |= {'_FUSE_R'}
                r_fused.tags |= {i + 1}
                _, r_fused, nexpok = ctr.expo(k_eff, r_fused, -beta)
                k_tn.remove_tags({'_FUSE_R'})
                r = ctr.unfuse_right(i + 1, r_fused)
                self._k[{i + 1, '_KET'}].modify(r)
                self._b[{i + 1, '_BRA'}].modify(r)
                self.eff_ham()[{i + 1, '_KET'}].modify(r)
        else:
            if i != 0:
                k_tn = self.eff_ham.move_to(i - 1).add_tags({'_FUSE_L'})
                k_eff = (k_tn ^ '_HAM')['_HAM']
                k_eff.tags |= {'_FUSE_L'}
                l_fused.tags |= {i}
                _, l_fused, nexpok = ctr.expo(k_eff, l_fused, -beta)
                k_tn.remove_tags({'_FUSE_L'})
                l = ctr.unfuse_left(i, l_fused)
                self._k[{i, '_KET'}].modify(l)
                self._b[{i, '_BRA'}].modify(l)
                self.eff_ham()[{i, '_KET'}].modify(l)
        
        return energy, error, (nexpo, nexpok)

    def construct_envs(self):
        t = time.perf_counter()
        pprint(" Constructing environment .. ", end='', flush=True)
        self.eff_ham = MovingEnvironment(self.n_sites, self.center, self.dot, self._b | self._h | self._k)
        pprint("T = %4.2f" % (time.perf_counter() - t))

    def blocking(self, i, forward, bond_dim, beta):
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
        
        Returns:
            energy : float
                Ground state energy.
            error : float
                Sum of discarded weights.
            ndav : int
                Number of Davidson iterations.
        """
        self.eff_ham.move_to(i)
        self.center = i

        if self.dot == 1:
            return self.update_one_dot(i, forward, bond_dim, beta)
        else:
            return self.update_two_dot(i, forward, bond_dim, beta)

    def sweep(self, forward, bond_dim, beta):
        """
        Perform one sweep iteration.
        
        Args:
            forward : bool
                Direction of current sweep. If True, sweep is performed from left to right.
            bond_dims : int
                Bond dimension of current sweep.
            noise : float
                Noise prefactor of current sweep.
        
        Returns:
            energy : float
                Ground state energy.
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
        
        sweep_energies = []

        for i in sweep_range:
            if self.dot == 2:
                pprint(" %3s Site = %4d-%4d .. " % ('-->' if forward else '<--', i, i + 1), end='', flush=True)
            else:
                pprint(" %3s Site = %4d .. " % ('-->' if forward else '<--', i), end='', flush=True)
            t = time.perf_counter()
            energy, error, nexpos = self.blocking(i, forward=forward, bond_dim=bond_dim, beta=beta)
            pprint("Nexpo = %4d/%4d E = %15.8f Error = %15.8f T = %4.2f" % (nexpos[0], nexpos[1], energy, error, time.perf_counter() - t))
            sweep_energies.append(energy)
        
        self._post_sweep()

        return sweep_energies[-1]

    def solve(self, n_sweeps, forward=True, two_dot_to_one_dot=-1):
        """
        Perform TDVP time evolution algorithm.
        
        Args:
            n_sweeps : int
                Maximum number of sweeps.
            tol : float
                Energy convergence threshold.
            forward : bool
                Direction of first sweep. If True, sweep is performed from left to right.
            two_dot_to_one_dot : int or -1
                Indicating when to switch to one-dot scheme. If -1, no switching.
        
        Returns:
            energy : float
                Final ground state energy.
        """
        if len(self.bond_dims) < n_sweeps:
            self.bond_dims.extend([self.bond_dims[-1]] * (n_sweeps - len(self.bond_dims)))
        
        start = time.perf_counter()
        
        current_beta = 0.0

        for iw in range(n_sweeps):

            current_beta += self.beta / 2
            pprint("Sweep = %4d | Direction = %8s | Bond dimension = %4d | Beta = %9.2g"
                % (iw, "forward" if forward else "backward", self.bond_dims[iw], current_beta))
            
            if two_dot_to_one_dot == iw:
                assert self.dot == 2
                self.dot = 1
                if self.center != 0 and self.center == self.n_sites - 2:
                    self.center = self.n_sites - 1

            energy = self.sweep(forward=forward, bond_dim=self.bond_dims[iw], beta=self.beta / 2)
            self.energies.append(energy)

            forward = not forward
            
            pprint('Time elapsed = %10.2f' % (time.perf_counter() - start))
            if iw % 2 == 1:
                pprint('Beta = %10.5f Energy = %15.8f' % (current_beta, energy))
        
        return energy
