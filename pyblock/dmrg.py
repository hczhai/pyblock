
from .tensor.tensor import Tensor

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
            
            tags_initial = {'_RIGHT'} | set(range(n_sites - self.dot, n_sites))
            self.envs = {self.n_sites - self.dot: self.tnc.select(tags_initial, which='any')}

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

            for i in range(1, self.n_sites - (self.dot - 1)):
                # add a new site to previous env, and contract one site
                self.envs[i] = self.envs[i - 1].copy()
                self.envs[i] |= self.tnc.select(i + self.dot - 1)
                self.envs[i] ^= ('_LEFT', i - 1)
            
            self.envs[self.n_sites - self.dot] |= self.tnc['_RIGHT']
            self.pos = self.n_sites - self.dot


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
    
    def sweep(self, forward, bond_dim):
        
        self.eff_ham = MovingEnvironment(self.n_sites, self.dot, self.tn, forward)

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
