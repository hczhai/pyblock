
# PYTHONPATH=../..:../../build python dmrg.py
# E(rand init) = -170.186268309451 (without noise)
# E(rand init) = -171.27683359988384 (with noise)
# E(FCI) = -171.276833655391


from pyblock.qchem import *
from pyblock.algorithm import DMRG

bond_dim = 200

with BlockHamiltonian.get(fcidump='N2-1E.FCIDUMP', pg='d2h', su2=True, output_level=-1) as hamil:
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    mps = MPS(lcp, center=0, dot=2)
    mps.randomize()
    mps.canonicalize()
    mpo = MPO(hamil)
    ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil))
    dmrg = DMRG(mpo, mps, bond_dims=[50, 80, 100, 200, 500], noise=[1E-3, 1E-4, 1E-5, 0], contractor=ctr)
    ener = dmrg.solve(10, 1E-6)
    print('final energy = ', ener)
