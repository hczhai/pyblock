
from pyblock.hamiltonian.block import BlockHamiltonian, BlockSymmetry
from pyblock.symmetry.symmetry import PGD2H
from pyblock.tensor.mpo import BlockMPO
from pyblock.block_dmrg import DMRG as BLOCK_DMRG
from block import load_rotation_matrix
from block import VectorInt, VectorMatrix
from block.io import Global

def translate_rotation_matrices(mps):
    rot_mats = {}
    infos = {}
    # current state_info
    cur = BlockSymmetry.initial_state_info(0)
    t0 = mps[0]
    for i in range(1, ham.n_sites):
        rot, cur = BlockSymmetry.to_rotation_matrix(cur, mps[i], i, t0)
        t0 = None
        rot_mats[tuple(range(0, i + 1))] = rot
        infos[tuple(range(0, i + 1))] = cur
    return rot_mats, infos

def translate_wavefunction(ham, mpo, mps, infos, one_dot=False):
    if one_dot:
        cur = infos[tuple(range(0, ham.n_sites - 2 + 1))]
        wfn, info = BlockSymmetry.to_wavefunction(
            True, cur, ham.n_sites, mps, mpo.target)
        sites = VectorInt(range(0, ham.n_sites - 2 + 1))
    else:
        cur = infos[tuple(range(0, ham.n_sites - 3 + 1))]
        wfn, info = BlockSymmetry.to_wavefunction(
            False, cur, ham.n_sites, mps, mpo.target)
        sites = VectorInt(range(0, ham.n_sites - 3 + 1))
    
    return wfn, info, sites

bond_dim = 30

ham = BlockHamiltonian(fcidump='N2.STO3G.FCIDUMP', point_group='d2h', dot=2, output_level=0)
mpo = BlockMPO(ham)

print(ham.n_electrons, ham.n_sites, ham.target_s, ham.target_spatial_sym)
print('spatial symmetries = ', [mpo.PG.IrrepNames[x] for x in ham.spatial_syms])

mps = mpo.rand_state(bond_dim)
mps.left_normalize()

# translate mps to rotaion matrices
# wavefunction has to be stored into disk to let block use it
rot_mats, infos = translate_rotation_matrices(mps)
wfn, info, wfn_sites = translate_wavefunction(ham, mpo, mps, infos, one_dot=True)
Global.dmrginp.output_level = 1
wfn.save_wavefunction_info(info, wfn_sites, 0)
wfn.deallocate()

dmrg = BLOCK_DMRG("")
dmrg.dmrg(gen_block=True, rot_mats=rot_mats)
dmrg.finalize()
