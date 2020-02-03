
# Example:
# Block SU(2) DMRG with random initial MPS genereated by python code
# Energy = -107.6482490830

from pyblock.hamiltonian.block import BlockHamiltonian, BlockSymmetry
from pyblock.symmetry.symmetry import PGD2H
from pyblock.tensor.mpo import BlockMPO
from pyblock.block_dmrg import DMRG as BLOCK_DMRG
from block import load_rotation_matrix
from block import VectorInt, VectorMatrix
from block.io import Global

bond_dim = 30

ham = BlockHamiltonian(fcidump='N2.STO3G.FCIDUMP', point_group='d2h', dot=2, output_level=0)
mpo = BlockMPO(ham)

print(ham.n_electrons, ham.n_sites, ham.target_s, ham.target_spatial_sym)
print('spatial symmetries = ', [mpo.PG.IrrepNames[x] for x in ham.spatial_syms])

lcp = mpo.get_line_coupling(bond_dim)
mpo.set_line_coupling(lcp)
mps = mpo.rand_state()
mps.left_normalize()

# translate MPS to rotation matrices
rot_mats = {}
for i in range(0, ham.n_sites):
    rot_mats[tuple(range(0, i + 1))] = mpo.info.get_left_rotation_matrix(i, mps[i])

dmrg_block = BLOCK_DMRG("")
dmrg_block.dmrg(gen_block=True, rot_mats=rot_mats)
dmrg_block.finalize()
