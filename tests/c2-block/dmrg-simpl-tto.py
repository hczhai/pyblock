
# PYTHONPATH=../..:../../build python dmrg-simpl-tto.py
# E(dmrg) = -75.728386566258


from pyblock.qchem import *
from pyblock.algorithm import DMRG
import numpy as np
import time
t = time.perf_counter()

np.random.seed(4)

page = DMRGDataPage(save_dir='node0')
memory = 2000

with BlockHamiltonian.get(fcidump='C2.BLOCK.FCIDUMP', pg='d2h', su2=True, output_level=0,
                          memory=memory, page=page, omp_threads=2) as hamil:
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    lcp.set_bond_dimension(50)
    mps = MPS(lcp, center=0, dot=2, iprint=True)
    mps.randomize()
    mps.canonicalize()
    print('mps ok')
    mpo = MPO(hamil, iprint=True)
    print('mpo ok')
    ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), Simplifier(AllRules()))
    dmrg = DMRG(mpo, mps, bond_dims=[50, 80, 100, 200, 250], noise=[1E-5, 1E-6, 1E-7, 1E-7, 1E-7, 0], contractor=ctr)
    print('init time = ', time.perf_counter() - t)
    ener = dmrg.solve(20, 1E-6, two_dot_to_one_dot=8)
    print('final energy = ', ener)
