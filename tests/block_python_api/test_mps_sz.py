
# PYTHONPATH=../..:../../build python test_mps_sz.py
# This code is used to generate random initial MPS
# for starting block code in non-spin-adapted mode
# After finishing this code, run block code with fullrestart
# to get the final energy.
# E = -107.648250907901

from pyblock.qchem import BlockHamiltonian, MPS, MPSInfo, LineCoupling
from pyblock.qchem.core import BlockSymmetry
from pyblock.legacy.block_dmrg import DMRG as BLOCK_DMRG

from block import save_rotation_matrix
from block import VectorInt
from block.dmrg import SweepParams

init_bond_dim = 30
spin_adapted = False

with BlockHamiltonian.get(fcidump='N2.STO3G.FCIDUMP.C1', pg='c1', su2=spin_adapted) as hamil:
    
    print(hamil.n_electrons, hamil.n_sites, hamil.target_s, hamil.target_spatial_sym)
    print('spatial symmetries = ', [hamil.PG.IrrepNames[x] for x in hamil.spatial_syms])
    
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    if init_bond_dim != -1:
        lcp.set_bond_dimension(init_bond_dim)
        
    # print bond dimensions
    for i in range(0, len(lcp.left_dims)):
        print("==== site %d ====" % i, " M = %d " % sum(lcp.left_dims[i].values()))
        for k, v in lcp.left_dims[i].items():
            print(k, v)
    
    print("block site basis = ", BlockSymmetry.initial_state_info(0))
    
    mps = MPS(lcp, center=hamil.n_sites, dot=0)
    mps.randomize()
    mps.canonicalize()
    mps_info = MPSInfo(lcp)
    
    rot_mats = {}
    for i in range(0, hamil.n_sites):
        rot_mats[tuple(range(0, i + 1))] = mps_info.get_left_rotation_matrix(i, mps[i])
    
    for k, v in rot_mats.items():
        # translate the list of spatial orbitals to list of spin-orbitals
        if not spin_adapted:
            k = k + tuple(k[-1] + kk + 1 for kk in k)
        print(k)
        # rotation matrix for current state (0) and averaged state (-1)
        save_rotation_matrix(VectorInt(k), v, 0)
        save_rotation_matrix(VectorInt(k), v, -1)

    # tell the block where to restart
    # start with a backward sweep
    swp = SweepParams()
    if not spin_adapted:
        swp.save_state(False, 2)  # 2 is number of sites (spin-orbital) in system block
    else:
        swp.save_state(False, 1)

    # now the rotation matrices are saved into disk
    # using fullrestart parameter in block input file
