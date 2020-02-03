
# PYTHONPATH=../..:../../build python dmrg.py
# E(rand init) = -170.186268309451
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

dmrg = DMRG(mpo, bond_dim, mps=mps)
ener = dmrg.solve(10, 1E-6)
print('final energy = ', ener)
