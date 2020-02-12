
# PYTHONPATH=../..:../../build python dmrg.py
# E = -6.225634098701

from pyblock.qchem import *
from pyblock.dmrg import DMRG

bond_dim = 200

with BlockHamiltonian.get(fcidump='HUBBARD-L8.FCIDUMP', pg='c1', su2=True, output_level=-1) as hamil:
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    mps = MPS(lcp, center=0, dot=2)
    mps.randomize()
    mps.canonicalize()
    mpo = MPO(hamil)
    ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil))
    dmrg = DMRG(mpo, mps, bond_dim=bond_dim, contractor=ctr)
    ener = dmrg.solve(10, 1E-6)
    print('final energy = ', ener)
