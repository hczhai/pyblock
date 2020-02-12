
# PYTHONPATH=../..:../../build python dmrg_block.py
# E(rand init) = -170.1862673967
# E(FCI) = -171.276833655391

from pyblock.qchem import BlockHamiltonian, MPS, MPSInfo, LineCoupling
from pyblock.legacy.block_dmrg import DMRG as BLOCK_DMRG

with BlockHamiltonian.get(fcidump='N2-1E.FCIDUMP', pg='d2h', su2=True) as hamil:
    
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    mps = MPS(lcp, center=hamil.n_sites, dot=0)
    mps.randomize()
    mps.canonicalize()
    mps_info = MPSInfo(lcp)
    
    rot_mats = {}
    for i in range(0, hamil.n_sites):
        rot_mats[tuple(range(0, i + 1))] = mps_info.get_left_rotation_matrix(i, mps[i])
    
    dmrg_block = BLOCK_DMRG("")
    dmrg_block.dmrg(gen_block=True, rot_mats=rot_mats)
