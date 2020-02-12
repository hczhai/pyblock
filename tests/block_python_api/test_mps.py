
# PYTHONPATH=../..:../../build python test_mps.py
# Example:
# Block SU(2) DMRG with random initial MPS genereated by python code
# Energy = -107.6482490830

from pyblock.qchem import BlockHamiltonian, MPS, MPSInfo, LineCoupling
from pyblock.legacy.block_dmrg import DMRG as BLOCK_DMRG

with BlockHamiltonian.get(fcidump='N2.STO3G.FCIDUMP', pg='d2h', su2=True) as hamil:
    
    print(hamil.n_electrons, hamil.n_sites, hamil.target_s, hamil.target_spatial_sym)
    print('spatial symmetries = ', [hamil.PG.IrrepNames[x] for x in hamil.spatial_syms])
    
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
