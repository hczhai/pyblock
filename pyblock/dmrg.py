
from .tensor.tensor import Tensor

class DMRGError(Exception):
    pass

class MovingEnvironment:
    def __init__(self, n_sites, dot, tn, forward):
        self.dot = dot
        self.n_sites = n_sites
        self.forward = forward
        # contracted tensor network
        self.tnc = tn.copy()

        self.init_environments(forward)
    
    def init_environments(self, forward):

        self.tnc |= Tensor(blocks=None, tags='_LEFT')
        self.tnc |= Tensor(blocks=None, tags='_RIGHT')

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
            
            self.envs[0] |= self.tnc['_LEFT']
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
            
            self.envs[self.n_sites - self.dot] |= self.tnc['_RIGHT']
            self.pos = self.n_sites - self.dot

    def move_to(self, i):
        if i > self.pos:
            # move right
            new_sys = self.envs[self.pos].select({'_LEFT', self.pos}, which='any')
            self.envs[self.pos + 1] |= new_sys ^ ('_LEFT', self.pos)
        elif i < self.pos:
            # move left
            new_sys = self.envs[self.pos].select({'_RIGHT', self.pos + self.dot - 1}, which='any')
            self.envs[self.pos - 1] |= new_sys ^ ('_RIGHT', self.pos)
    
    def __call__(self):
        return self.envs[self.pos]

class DMRG:
    def __init__(self, mpo, bond_dims, cut_offs=1E-9, dot=2, mps=None):
        self.n_sites = mpo.n_sites
        self.dot = dot
        self.bond_dims = bond_dims if isinstance(
            bond_dims, list) else [bond_dims]

        if mps is not None:
            self._k = mps.copy()
        else:
            self._k = mpo.rand_state(self.bond_dims[0])

        self._b = self._k.copy()
        self._h = mpo.copy()

        self._k.add_tags({'_KET'})
        self._b.add_tags({'_BRA'})
        self._h.add_tags({'_HAM'})

        self.tn = self._b | self._h | self._k

        self.energies = []
    
    def update_one_dot(self, i, bond_dim):
        
        h_eff = (self.eff_ham() ^ '_HAM')['_HAM']

        gs_old = self._k[{i, '_KET'}]

        if h_eff.contractor is not None:
            energy, gs = h_eff.contractor.eigs(h_eff, gs_old)
        else:
            raise DMRGError('general eigenvalue solver is not defined!')
        
        self._k[i].modify(data=gs)
        self._b[i].modify(data=gs)

    def blocking(self, i, forward, bond_dim):

        self.eff_ham.move_to(i)

        if self.dot == 1:
            return self.update_one_dot(i, bond_dim)
        else:
            return self.update_two_dot(i, bond_dim)

    def sweep(self, forward, bond_dim):
        
        self.eff_ham = MovingEnvironment(self.n_sites, self.dot, self.tn, forward)

        # if forward/backward, i is the first dot site

        if forward:
            sweep_range = range(0, self.n_sites - self.dot + 1)
        else:
            sweep_range = range(self.n_sites - self.dot, -1, -1)
        
        sweep_energies = []

        for i in sweep_range:
            print("\t Iteration = %4d ... " % i)
            energy = self.blocking(i, forward=forward, bond_dim=bond_dim)
            print("\t\t\t Energy = %15.8f " % energy)
            sweep_energies.append(energy)

        return sweep_energies[-1]

        # TODO:: not implemented !!!

    def solve(self, n_sweeps, tol):

        forward = False
        converged = False

        if len(self.bond_dims) < n_sweeps:
            self.bond_dims.extend([self.bond_dims[-1]] * (n_sweeps - len(self.bond_dims)))

        for iw in range(n_sweeps):

            print("Sweep = %4d | Direction = %8s | Bond dimension = %4d"
                % (iw, "forward" if forward else "backward", self.bond_dims[iw]))

            energy = self.sweep(forward=forward, bond_dim=self.bond_dims[iw])
            self.energies.append(energy)

            converged = (len(self.energies) >= 2 and abs(
                self.energies[-1] - self.energies[-2]) < tol)

            if converged:
                break

            forward = not forward
