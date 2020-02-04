
# PYTHONPATH=../..:../../build python dmrg.py
# E(rand init) = -170.186268309451 (without noise)
# E(rand init) = -171.27683359988384 (with noise)
# E(FCI) = -171.276833655391

from pyblock.dmrg import DMRG
from pyblock.hamiltonian.block import BlockHamiltonian
from pyblock.tensor.mpo import BlockMPO

ham = BlockHamiltonian(fcidump='N2-1E.FCIDUMP', point_group='d2h', dot=2, output_level=0)
mpo = BlockMPO(ham)
mpo.init_site_operators()
mpo.init_mpo_tensors()

bond_dim = 500

lcp = mpo.get_line_coupling(bond_dim)
mpo.set_line_coupling(lcp)
# mps = mpo.identity_state()
mps = mpo.rand_state()
mps.left_normalize()

dmrg = DMRG(mpo, bond_dim=[50, 80, 100, 200, 500], noise=[1E-3, 1E-4, 1E-5, 0], mps=mps)
ener = dmrg.solve(5, 1E-6)
print('final energy = ', ener)
