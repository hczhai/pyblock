
# PYTHONPATH=../..:../../build python dmrg-no-save.py
# E(dmrg) = -75.728386566258


from pyblock.qchem import *
from pyblock.dmrg import DMRG
import numpy as np

np.random.seed(1234)

bond_dim = 200

page = None
memory = 16000

with BlockHamiltonian.get(fcidump='C2.BLOCK.FCIDUMP', pg='d2h', su2=True, output_level=0, memory=memory, page=page) as hamil:
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    lcp.set_bond_dimension(50)
    mps = MPS(lcp, center=0, dot=2)
    mps.randomize()
    mps.canonicalize()
    print('mps ok')
    mpo = MPO(hamil)
    print('mpo ok')
    ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil))
    dmrg = DMRG(mpo, mps, bond_dims=[50, 80, 100, 200, 250], noise=[1E-3, 1E-4, 1E-5, 1E-5, 1E-5, 0], contractor=ctr)
    ener = dmrg.solve(20, 1E-6)
    print('final energy = ', ener)
