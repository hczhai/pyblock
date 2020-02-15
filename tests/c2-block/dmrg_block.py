
# PYTHONPATH=../..:../../build python dmrg_block.py
# E = -75.7283698579
# E(block-dmrg) = -75.728386566258

from pyblock.qchem import BlockHamiltonian, MPS, MPSInfo, LineCoupling
from pyblock.legacy.block_dmrg import DMRG as BLOCK_DMRG

with BlockHamiltonian.get(fcidump='C2.BLOCK.FCIDUMP', pg='d2h', su2=True) as hamil:
    
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    lcp.set_bond_dimension(100)
    mps = MPS(lcp, center=hamil.n_sites, dot=0)
    mps.randomize()
    mps.canonicalize()
    mps_info = MPSInfo(lcp)
    
    rot_mats = {}
    for i in range(0, hamil.n_sites):
        rot_mats[tuple(range(0, i + 1))] = mps_info.get_left_rotation_matrix(i, mps[i])
    
    dmrg_block = BLOCK_DMRG("")
    dmrg_block.dmrg(gen_block=True, rot_mats=rot_mats)
