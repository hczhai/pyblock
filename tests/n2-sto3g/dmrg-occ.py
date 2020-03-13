
# PYTHONPATH=../..:../../build python dmrg-occ.py
# E(DMRG) = -107.6541224070486
# E(HF) = -107.496500511798
# E(CCSD) = -107.6501974086188
# E(FCI) =-107.65412244752243


from pyblock.qchem import *
from pyblock.algorithm import DMRG

page = DMRGDataPage('node0')

with BlockHamiltonian.get(fcidump='N2.STO3G-OCC.FCIDUMP', pg='d2h', su2=True, output_level=-1, page=page) as hamil:
    occ = [float(x) for x in open('N2.STO3G-OCC.OCC', 'r').read().split()]
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    lcp.set_bond_dimension_using_occ(50, occ, bias=10000)
    mps = MPS(lcp, center=0, dot=2)
    mps.randomize()
    mps.canonicalize()
    mpo = MPO(hamil)
    ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), Simplifier(AllRules()))
    dmrg = DMRG(mpo, mps, bond_dims=[50, 80, 100, 200, 500], noise=[1E-3, 1E-5, 1E-6, 0], contractor=ctr)
    ener = dmrg.solve(10, 1E-6)
    print('final energy = ', ener)

