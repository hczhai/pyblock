
# PYTHONPATH=../..:../../build python dmrg.py
# E = -6.225634098701 (L8)
# E = -12.966715281726584 (L16)

from pyblock.qchem import *
from pyblock.dmrg import DMRG

page = DMRGDataPage(save_dir='node0')
bond_dim = 100

with BlockHamiltonian.get(fcidump='HUBBARD-L16.FCIDUMP', pg='c1', su2=True, output_level=-1, page=page, memory=4000) as hamil:
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    lcp.set_bond_dimension(50)
    mps = MPS(lcp, center=0, dot=2, iprint=True)
    mps.randomize()
    mps.canonicalize()
    mpo = MPO(hamil, iprint=True)
    ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), Simplifier(AllRules()))
    dmrg = DMRG(mpo, mps, bond_dims=bond_dim, contractor=ctr)
    ener = dmrg.solve(10, 1E-6)
    print('final energy = ', ener)
