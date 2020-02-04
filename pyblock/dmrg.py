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

from .tensor.tensor import Tensor

class DMRGError(Exception):
    pass

class MovingEnvironment:
    """
    Environment blocks in DMRG.
    
    Attributes:
        n_sites : int
            Number of sites/orbitals.
        dot : int
            Two-dot (2) or one-dot (1) scheme.
        forward : bool
            Direction of current sweep. If True, sweep is performed from left to right.
        tnc : TensorNetwork
            The tensor network <bra|H|ket> before contraction.
        pos : int
            Current site position of left dot.
        envs : dict(int -> TensorNetwork)
            DMRG Environment for different positions of left dot.
    """
    def __init__(self, n_sites, dot, tn, forward, complete=False):
        self.dot = dot
        self.n_sites = n_sites
        self.forward = forward
        # contracted tensor network
        self.tnc = tn.copy()

        self.init_environments(forward, complete)
    
    def init_environments(self, forward, complete):
        """
        Initialize DMRG Environment blocks by contraction.
        
        Args:
            forward : bool
                Direction of current sweep. If True, sweep is performed from left to right.
            complete : bool
                Whether extra edge environments (not used in DMRG algorithm) should be generated.
        """

        self.tnc |= Tensor(blocks=None, tags={'_LEFT', '_HAM'})
        self.tnc |= Tensor(blocks=None, tags={'_RIGHT', '_HAM'})

        if forward:
            
            tags_initial = {'_RIGHT'} | set(range(self.n_sites - self.dot, self.n_sites))
            self.envs = {self.n_sites - self.dot: self.tnc.select(tags_initial, which='any')}

            # sites inside [env=] are contracted. there is extra one dot site not contracted
            # i = 0, dot = 1 :: [sys=][sdot=0][env=1,2..]
            # i = 0, dot = 2 :: [sys=][sdot=0][edot=1][env=2,3..]
            for i in range(self.n_sites - self.dot - 1, -1, -1):
                # add a new site to previous env, and contract one site
                self.envs[i] = self.envs[i + 1].copy()
                self.envs[i] |= self.tnc.select(i)
                self.envs[i] ^= ('_RIGHT', i + self.dot)
            
            if complete:
                for i in range(1, self.dot + 1):
                    self.envs[-i] = self.envs[-i + 1].copy()
                    self.envs[-i] ^= ('_RIGHT', -i + self.dot)
            
            self.envs[0] |= self.tnc[{'_LEFT', '_HAM'}]
            self.pos = 0
        
        else:

            tags_initial = {'_LEFT'} | set(range(0, self.dot))
            self.envs = {0: self.tnc.select(tags_initial, which='any')}

            # i = n - 1, dot = 1 :: [env=..n-2][sdot=n-1][sys=]
            # i = n - 2, dot = 2 :: [env=..n-3][edot=n-2][sdot=n-1][sys=]
            for i in range(1, self.n_sites - (self.dot - 1)):
                # add a new site to previous env, and contract one site
                self.envs[i] = self.envs[i - 1].copy()
                self.envs[i] |= self.tnc.select(i + self.dot - 1)
                self.envs[i] ^= ('_LEFT', i - 1)
                
            if complete:
                for i in range(1, self.dot + 1):
                    self.envs[self.n_sites - self.dot + i] = self.envs[self.n_sites - self.dot + i - 1].copy()
                    self.envs[self.n_sites - self.dot + i] ^= ('_LEFT', self.n_sites - self.dot + i - 1)
            
            self.envs[self.n_sites - self.dot] |= self.tnc[{'_RIGHT', '_HAM'}]
            self.pos = self.n_sites - self.dot

    def move_to(self, i):
        """
        Change the current left dot site to ``i`` (by zero or one site).
        """
        if i > self.pos:
            # move right
            new_sys = self.envs[self.pos].select({'_LEFT', self.pos}, which='any')
            self.envs[self.pos + 1] |= new_sys ^ ('_LEFT', self.pos)
            self.pos += 1
        elif i < self.pos:
            # move left
            new_sys = self.envs[self.pos].select({'_RIGHT', self.pos + self.dot - 1}, which='any')
            self.envs[self.pos - 1] |= new_sys ^ ('_RIGHT', self.pos + self.dot - 1)
            self.pos -= 1
    
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
        tn : TensorNetwork
            The tensor network <bra|H|ket> before contraction.
        energies : list(float)
            Energies collected for all sweeps.
    """
    def __init__(self, mpo, bond_dims, dot=2, mps=None):
        self.n_sites = mpo.n_sites
        self.dot = dot
        self.bond_dims = bond_dims if isinstance(
            bond_dims, list) else [bond_dims]

        if mps is not None:
            self._k = mps.deep_copy()
        else:
            self._k = mpo.rand_state()

        self._b = self._k.deep_copy()
        self._h = mpo.copy()

        self._k.add_tags({'_KET'})
        self._b.add_tags({'_BRA'})
        self._h.add_tags({'_HAM'})

        self.tn = self._b | self._h | self._k

        self.energies = []
    
    def update_one_dot(self, i, forward, bond_dim):
        """Update local site in one-dot scheme. Not implemented."""
        
        h_eff = (self.eff_ham() ^ '_HAM')['_HAM']

        gs_old = self._k[{i, '_KET'}]

        if h_eff.contractor is not None:
            energy, gs = h_eff.contractor.eigs(h_eff, gs_old)
        else:
            raise DMRGError('general eigenvalue solver is not defined!')
        
        self._k[i].modify(gs)
        self._b[i].modify(gs)
    
    def update_two_dot(self, i, forward, bond_dim):
        """
        Update local site in two-dot scheme.
        
        Args:
            i : int
                Site index of left dot
            forward : bool
                Direction of current sweep. If True, sweep is performed from left to right.
            bond_dims : int
                Bond dimension of current sweep.
        
        Returns:
            energy : float
                Ground state energy.
            error : float
                Sum of discarded weights.
        """
        h_eff = (self.eff_ham() ^ '_HAM')['_HAM']
        
        gs_old = self._k.select({i, i + 1}, which='any')
        
        if h_eff.contractor is not None:
            energy, gs = h_eff.contractor.eigs(h_eff, gs_old)
            
            l_fused, r_fused, error = gs.split(absorb_right=forward, k=bond_dim)

            h_eff.contractor.update_local_mps_info(i, l_fused)
            l = h_eff.contractor.unfuse_left(i, l_fused)
            r = h_eff.contractor.unfuse_right(i + 1, r_fused)

        else:
            raise DMRGError('general eigenvalue solver is not implemented!')
        
        self._k[i].modify(l)
        self._b[i].modify(l)
        self._k[i + 1].modify(r)
        self._b[i + 1].modify(r)
        
        return energy, error

    def blocking(self, i, forward, bond_dim):
        """
        Perform one blocking iteration.
        
        Args:
            i : int
                Site index of left dot
            forward : bool
                Direction of current sweep. If True, sweep is performed from left to right.
            bond_dims : int
                Bond dimension of current sweep.
        
        Returns:
            energy : float
                Ground state energy.
            error : float
                Sum of discarded weights.
        """
        self.eff_ham.move_to(i)

        if self.dot == 1:
            return self.update_one_dot(i, forward, bond_dim)
        else:
            return self.update_two_dot(i, forward, bond_dim)

    def sweep(self, forward, bond_dim):
        """
        Perform one sweep iteration.
        
        Args:
            forward : bool
                Direction of current sweep. If True, sweep is performed from left to right.
            bond_dims : int
                Bond dimension of current sweep.
        
        Returns:
            energy : float
                Ground state energy.
        """
        self.eff_ham = MovingEnvironment(self.n_sites, self.dot, self.tn, forward)

        # if forward/backward, i is the first dot site

        if forward:
            sweep_range = range(0, self.n_sites - self.dot + 1)
        else:
            sweep_range = range(self.n_sites - self.dot, -1, -1)
        
        sweep_energies = []

        for i in sweep_range:
            print("\t Iteration = %4d %s ... " % (abs(i - sweep_range[0]), 'F' if forward else 'B'))
            energy, error = self.blocking(i, forward=forward, bond_dim=bond_dim)
            print("\t\t\t Energy = %15.8f Error = %15.8f" % (energy, error))
            sweep_energies.append(energy)

        return sorted(sweep_energies)[0]

    def solve(self, n_sweeps, tol):
        """
        Perform DMRG algorithm.
        
        Args:
            n_sweeps : int
                Maximum number of sweeps.
            tol : float
                Energy convergence threshold.
        
        Returns:
            energy : float
                Final ground state energy.
        """
        forward = False
        converged = False

        if len(self.bond_dims) < n_sweeps:
            self.bond_dims.extend([self.bond_dims[-1]] * (n_sweeps - len(self.bond_dims)))

        for iw in range(n_sweeps):

            print("Sweep = %4d | Direction = %8s | Bond dimension = %4d"
                % (iw, "forward" if forward else "backward", self.bond_dims[iw]))

            energy = self.sweep(forward=forward, bond_dim=self.bond_dims[iw])
            self.energies.append(energy)

            converged = (len(self.energies) >= 2 and tol is not None and abs(
                self.energies[-1] - self.energies[-2]) < tol)

            if converged:
                break

            forward = not forward
        
        return energy
