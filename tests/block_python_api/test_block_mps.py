
# Example:
# Block SU(2) DMRG with initial MPS genereated from one HF determinant
# this is simply a wrapper for the "MPS" code defined inside Block code
# Energy = -107.6482508192

from pyblock.block_dmrg import DMRG as BLOCK_DMRG
from block.dmrg import MPS as BLOCK_MPS, SweepParams, MPS_init
from block import VectorBool
from block.io import Global
import block

# call block functions to write rotation matrices into disk
dmrg = BLOCK_DMRG('input.txt', output_level=0)
MPS_init(True)
occ = VectorBool(Global.dmrginp.hf_occupancy)
print(occ)
print(BLOCK_MPS.n_sweep_iters)
mps = BLOCK_MPS(occ)
mps.write_to_disk(0, True)
swp = SweepParams()
swp.save_state(False, 1)

# fullrestart calculation
dmrg.dmrg(gen_block=True, rot_mats=None)
dmrg.finalize()
