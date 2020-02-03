
# PYTHONPATH=../..:../../build python dmrg_block.py
# E(rand init) = -170.1862673967
# E(FCI) = -171.276833655391

from pyblock.hamiltonian.block import BlockHamiltonian
from pyblock.tensor.mpo import BlockMPO
from pyblock.block_dmrg import DMRG as BLOCK_DMRG

ham = BlockHamiltonian(fcidump='N2-1E.FCIDUMP', point_group='d2h', dot=2, output_level=0)
mpo = BlockMPO(ham)
mpo.init_site_operators()
mpo.init_mpo_tensors()

bond_dim = 500

lcp = mpo.get_line_coupling(bond_dim)
mpo.set_line_coupling(lcp)
mps = mpo.identity_state()
# mps = mpo.rand_state()
mps.left_normalize()

rot_mats = {tuple(range(0, i + 1)): mpo.info.get_left_rotation_matrix(i, mps[i]) for i in range(0, ham.n_sites)}
dmrg_bl = BLOCK_DMRG("")
dmrg_bl.dmrg(gen_block=True, rot_mats=rot_mats)
dmrg_bl.finalize()
