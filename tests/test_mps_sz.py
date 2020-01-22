
from pyblock.hamiltonian.block import BlockHamiltonian, BlockSymmetry
from pyblock.symmetry.symmetry import LineCoupling, ParticleN, SZ, PGC1, SU2
from pyblock.tensor.mps import MPS
from block import save_rotation_matrix
from block import VectorInt
from block.io import Global
from block.dmrg import SweepParams, MPS_init, MPS as BLOCK_MPS
from fractions import Fraction

# n_sites is number of spatial orbitals
n_sites = 10
n_electrons = 14
init_bond_dim = 1
point_group = "c1"
fcidump = 'N2.STO3G.FCIDUMP.C1'
dot = 2
spin_adapted = False

if not spin_adapted:
    empty = ParticleN(0) * SZ(0) * PGC1(0)
    basis = [{
        ParticleN(0) * SZ(0) * PGC1(0): 1,
        ParticleN(1) * SZ(-Fraction(1, 2)) * PGC1(0): 1,
        ParticleN(1) * SZ(Fraction(1, 2)) * PGC1(0): 1,
        ParticleN(2) * SZ(0) * PGC1(0): 1
    } for _ in range(n_sites)]
    target = ParticleN(n_electrons) * SZ(0) * PGC1(0)
else:
    empty = ParticleN(0) * SU2(0) * PGC1(0)
    basis = [{
        ParticleN(0) * SU2(0) * PGC1(0): 1,
        ParticleN(1) * SU2(Fraction(1, 2)) * PGC1(0): 1,
        ParticleN(2) * SU2(0) * PGC1(0): 1
    } for _ in range(n_sites)]
    target = ParticleN(14) * SU2(0) * PGC1(0)

# print q-numbers and dimensions at each site
lc = LineCoupling(n_sites, basis, target, empty, both_dir=True)

# now one can do some changes on LineCoupling ``lc``
# lc.dims[i] is a dict mapping q-numbers to dimensions at site i
if init_bond_dim != -1:
    lc.set_bond_dim(init_bond_dim)

# print bond dimensions
for i in range(0, len(lc.dims)):
    print("==== site %d ====" % i, " M = %d " % sum(lc.dims[i].values()))
    for k, v in lc.dims[i].items():
        print(k, v)

mps = MPS.from_line_coupling(lc)
mps.randomize()
mps.left_normalize()

# initialize block program using some random parameters
def block_init():
    ham = BlockHamiltonian(fcidump=fcidump, point_group=point_group, dot=dot, spin_adapted=spin_adapted, output_level=0)
    print("block site basis = ", BlockSymmetry.initial_state_info(0))

def translate_rotation_matrices(mps):
    rot_mats = {}
    infos = {}
    # current state_info
    cur = BlockSymmetry.initial_state_info(0)
    t0 = mps[0]
    for i in range(1, n_sites):
        rot, cur = BlockSymmetry.to_rotation_matrix(cur, mps[i], i, t0)
        t0 = None
        rot_mats[tuple(range(0, i + 1))] = rot
        infos[tuple(range(0, i + 1))] = cur
    return rot_mats, infos

block_init()
rot_mats, _ = translate_rotation_matrices(mps)
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
