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
DMRG algorithm.
"""

from ..tensor.tensor import Tensor, TensorNetwork
import time
from mpi4py import MPI

class DMRGError(Exception):
    pass

def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)

class MovingEnvironment:
    """
    Environment blocks in DMRG.
    
    Attributes:
        n_sites : int
            Number of sites/orbitals.
        dot : int
            Two-dot (2) or one-dot (1) scheme.
        tnc : TensorNetwork
            The tensor network <bra|H|ket> before contraction.
        pos : int
            Current site position of left dot.
        envs : dict(int -> TensorNetwork)
            DMRG Environment for different positions of left dot.
    """
    def __init__(self, n_sites, center, dot, tn, iprint=True):
        self.pos = center
        self.dot = dot
        self.n_sites = n_sites
        self.tnc = tn.copy()
        self.iprint = iprint
        self.init_environments()
    
    def init_environments(self):
        """Initialize DMRG Environment blocks by contraction."""

        self.tnc |= Tensor(blocks=None, tags={'_LEFT', '_HAM'})
        self.tnc |= Tensor(blocks=None, tags={'_RIGHT', '_HAM'})

        self.envs = {}
        for i in range(1, self.dot + 1):
            tags_initial = {'_RIGHT'} | set(range(self.n_sites - i, self.n_sites))
            self.envs[self.n_sites - i] = self.tnc.select(tags_initial, which='any')

        # sites inside [env=] are contracted. there is extra one dot site not contracted
        # i = 0, dot = 1 :: [sys=][sdot=0][env=1,2..]
        # i = 0, dot = 2 :: [sys=][sdot=0][edot=1][env=2,3..]
        for i in range(self.n_sites - self.dot - 1, self.pos - 1, -1):
            if self.iprint:
                pprint("\r Constructing environment .. %3d%% " %
                       ((self.n_sites - self.dot - i) * 100 // self.n_sites),
                       end="", flush=True)
            # add a new site to previous env, and contract one site
            self.envs[i] = self.envs[i + 1].copy()
            self.envs[i].remove({i}, in_place=True)
            self.envs[i] |= self.tnc.select(i)
            self.envs[i] ^= ('_RIGHT', i + self.dot)

        tags_initial = {'_LEFT'} | set(range(0, self.dot - 1))
        self.envs[-1] = self.tnc.select(tags_initial, which='any')

        # i = n - 1, dot = 1 :: [env=..n-2][sdot=n-1][sys=]
        # i = n - 2, dot = 2 :: [env=..n-3][edot=n-2][sdot=n-1][sys=]
        for i in range(0, self.pos):
            if self.iprint:
                pprint("\r Constructing environment .. %3d%% " %
                       ((self.n_sites - self.dot - self.pos + i + 1) * 100 // self.n_sites),
                       end="", flush=True)
            # add a new site to previous env, and contract one site
            self.envs[i] = self.envs[i - 1].copy()
            self.envs[i].remove({i + self.dot - 1}, in_place=True)
            self.envs[i] |= self.tnc.select(i + self.dot - 1)
            self.envs[i] ^= ('_LEFT', i - 1)
        
        self.envs[self.pos] |= (self.envs[self.pos - 1].copy() ^ ('_LEFT', self.pos - 1)).select('_LEFT')
    
    def prepare_sweep(self, dot, pos):
        """Prepare environment for next sweep."""
        
        for i in range(self.n_sites - 1, self.pos, -1):
            self.envs[i].remove({'_LEFT'}, in_place=True)
        
        for i in range(-1, self.pos):
            self.envs[i].remove({'_RIGHT'}, in_place=True)
        
        if dot != self.dot:
            assert dot == 1 and self.dot == 2
            self.dot = 1
            for i in range(self.n_sites - 1, self.pos, -1):
                self.envs[i] = self.envs[i - 1].copy()
                self.envs[i].remove({'_LEFT'}, in_place=True)
                self.envs[i].remove({i - 1}, in_place=True)
            
            for i in range(-1, self.pos):
                self.envs[i].remove({i + 1}, in_place=True)
            
            if pos != self.pos:
                assert pos == self.n_sites - 1 and self.pos == self.n_sites - 2
                self.envs[self.n_sites - 1] = self.envs[self.n_sites - 2].copy()
                self.envs[self.n_sites - 1] ^= ('_LEFT', self.n_sites - 2)
                self.envs[self.pos].remove({'_RIGHT'}, in_place=True)
                self.envs[self.pos].remove({self.n_sites - 1}, in_place=True)
                self.pos = self.n_sites - 1
            else:
                self.envs[self.pos] ^= ('_RIGHT', self.pos + self.dot)
    
    def move_to(self, i):
        """
        Change the current left dot site to ``i`` (by zero or one site).
        """
        if i > self.pos:
            # move right
            new_sys = self.envs[self.pos].select({'_LEFT', self.pos}, which='any')
            self.envs[self.pos + 1] |= new_sys.copy() ^ ('_LEFT', self.pos)
            if self.dot == 2:
                self.envs[self.pos + 1].remove({self.pos + 1}, in_place=True)
                self.envs[self.pos + 1] |= self.envs[self.pos].select({self.pos + 1})
            self.pos += 1
            if self.dot == 1:
                return self.envs[self.pos].select({'_LEFT'}) | self.envs[self.pos - 1].select({'_RIGHT'})
            else:
                return self.envs[self.pos].select({'_LEFT', self.pos}, which='any') | self.envs[self.pos - 1].select({'_RIGHT'})
        elif i < self.pos:
            # move left
            new_sys = self.envs[self.pos].select({'_RIGHT', self.pos + self.dot - 1}, which='any')
            self.envs[self.pos - 1] |= new_sys.copy() ^ ('_RIGHT', self.pos + self.dot - 1)
            if self.dot == 2:
                self.envs[self.pos - 1].remove({self.pos}, in_place=True)
                self.envs[self.pos - 1] |= self.envs[self.pos].select({self.pos})
            self.pos -= 1
            if self.dot == 1:
                return self.envs[self.pos].select({'_RIGHT'}) | self.envs[self.pos + 1].select({'_LEFT'})
            else:
                return self.envs[self.pos].select({'_RIGHT', self.pos + 1}, which='any') | self.envs[self.pos + 1].select({'_LEFT'})
    
    def __call__(self):
        return self.envs[self.pos]

class DMRG:
    """
    DMRG algorithm.
    
    Attributes:
        n_sites : int
            Number of sites/orbitals
        dot : int
            Two-dot (2) or one-dot (1) scheme.
        bond_dims : list(int) or int
            Bond dimension for each sweep.
        noise : list(float) or float
            Noise prefactor for each sweep.
        energies : list(float)
            Energies collected for all sweeps.
    """
    def __init__(self, mpo, mps, bond_dims, noise=0.0, contractor=None):
        self.n_sites = len(mpo)
        self.dot = mps.dot
        self.center = mps.center
        self.bond_dims = bond_dims if isinstance(bond_dims, list) else [bond_dims]
        self.noise = noise if isinstance(noise, list) else [noise]
        
        self.mps = mps.deep_copy()
        
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
        
        self.rebuild = contractor.rebuild
        
        if not self.rebuild:
            self.construct_envs()
    
    def set_mps(self, tags, wfn):
        self.mps = self._k.deep_copy()
        self.mps.center = self.center
        self.mps.dot = self.dot
        self.mps.replace(tags, wfn.deep_copy().set_tags(tags), which='any')
        self.mps.remove_tags({'_KET'})
        self.mps.form = self.canonical_form.copy()
        self.mps.set_contractor(None)
    
    def update_one_dot(self, i, forward, bond_dim, noise):
        """
        Update local site in one-dot scheme.
        
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
        energy, gs, ndav = ctr.eigs(h_eff, gs_old)
        
        if not fuse_left and forward:
            gs = ctr.unfuse_right(i, gs)
            self.set_mps({i}, gs)
            ctr.fuse_left(i, gs, self.canonical_form[i])
        elif fuse_left and not forward:
            gs = ctr.unfuse_left(i, gs)
            self.set_mps({i}, gs)
            ctr.fuse_right(i, gs, self.canonical_form[i])
        
        dm = gs.get_diag_density_matrix(trace_right=forward, noise=noise)
        
        l_fused, r_fused, error = \
            gs.split_using_density_matrix(dm, absorb_right=forward, k=bond_dim, limit=None)
        
        if forward:
            ctr.update_local_left_mps_info(i, l_fused)
            gs_new = ctr.unfuse_left(i, l_fused)
            if i + 1 < self.n_sites:
                self.canonical_form[i:i + 2] = "LC"
                adj_new = Tensor.contract(r_fused, self._k[{i + 1, '_KET'}], [1], [0])
                self._k[{i + 1, '_KET'}].modify(adj_new)
                self._b[{i + 1, '_BRA'}].modify(adj_new)
                self.eff_ham.envs[self.eff_ham.pos + 1][{i + 1, '_KET'}].modify(adj_new)
                self.eff_ham.envs[self.eff_ham.pos + 1][{i + 1, '_BRA'}].modify(adj_new)
            else:
                gs_new.right_multiply(r_fused.to_dict(0))
                ctr.update_local_left_mps_info(i, gs_new)
                self.canonical_form[i] = 'K'
        else:
            ctr.update_local_right_mps_info(i, r_fused)
            gs_new = ctr.unfuse_right(i, r_fused)
            if i - 1 >= 0:
                self.canonical_form[i - 1:i + 1] = "CR"
                adj_new = Tensor.contract(self._k[{i - 1, '_KET'}], l_fused, [2], [0])
                self._k[{i - 1, '_KET'}].modify(adj_new)
                self._b[{i - 1, '_BRA'}].modify(adj_new)
                self.eff_ham.envs[self.eff_ham.pos - 1][{i - 1, '_KET'}].modify(adj_new)
                self.eff_ham.envs[self.eff_ham.pos - 1][{i - 1, '_BRA'}].modify(adj_new)
            else:
                gs_new.left_multiply(l_fused.to_dict(1))
                ctr.update_local_right_mps_info(i, gs_new)
                self.canonical_form[i] = 'S'
        
        self.eff_ham()[{i, '_HAM'} | fuse_tags].tags -= fuse_tags
        self._k[{i, '_KET'}].modify(gs_new)
        self._b[{i, '_BRA'}].modify(gs_new)
        self.eff_ham()[{i, '_KET'}].modify(gs_new)
        self.eff_ham()[{i, '_BRA'}].modify(gs_new)
        
        return energy, error, ndav
    
    def update_two_dot(self, i, forward, bond_dim, noise):
        """
        Update local sites in two-dot scheme.
        
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
        energy, gs, ndav = ctr.eigs(h_eff, gs_old)
        
        self.set_mps({i, i + 1}, gs)
        
        dm = gs.get_diag_density_matrix(trace_right=forward, noise=noise)
        
        l_fused, r_fused, error = \
            gs.split_using_density_matrix(dm, absorb_right=forward, k=bond_dim, limit=None)
        
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
        
        return energy, error, ndav

    def construct_envs(self):
        t = time.perf_counter()
        pprint(" Constructing environment .. ", end='', flush=True)
        self.eff_ham = MovingEnvironment(self.n_sites, self.center, self.dot, self._b | self._h | self._k)
        pprint("T = %4.2f" % (time.perf_counter() - t))

    def blocking(self, i, forward, bond_dim, noise):
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
            return self.update_one_dot(i, forward, bond_dim, noise)
        else:
            return self.update_two_dot(i, forward, bond_dim, noise)

    def sweep(self, forward, bond_dim, noise):
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
            energy, error, ndav = self.blocking(i, forward=forward, bond_dim=bond_dim, noise=noise)
            pprint("Ndav = %4d E = %15.8f Error = %15.8f T = %4.2f" % (ndav, energy, error, time.perf_counter() - t))
            sweep_energies.append(energy)
        
        self._post_sweep()

        return sorted(sweep_energies)[0]

    def solve(self, n_sweeps, tol, forward=True, two_dot_to_one_dot=-1):
        """
        Perform DMRG algorithm.
        
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
        converged = False

        if len(self.bond_dims) < n_sweeps:
            self.bond_dims.extend([self.bond_dims[-1]] * (n_sweeps - len(self.bond_dims)))
            
        if len(self.noise) < n_sweeps:
            self.noise.extend([self.noise[-1]] * (n_sweeps - len(self.noise)))
        
        start = time.perf_counter()
        
        for iw in range(n_sweeps):

            pprint("Sweep = %4d | Direction = %8s | Bond dimension = %4d | Noise = %9.2g"
                % (iw, "forward" if forward else "backward", self.bond_dims[iw], self.noise[iw]))
            
            if two_dot_to_one_dot == iw:
                assert self.dot == 2
                self.dot = 1
                if self.center != 0 and self.center == self.n_sites - 2:
                    self.center = self.n_sites - 1

            energy = self.sweep(forward=forward, bond_dim=self.bond_dims[iw], noise=self.noise[iw])
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
