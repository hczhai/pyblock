
# This code is used to generate random initial MPS
# for starting block code in non-spin-adapted mode
# After finishing this code, run block code with fullrestart
# to get the final energy.
# E = -107.648250907901

from pyblock.hamiltonian.block import BlockHamiltonian, BlockSymmetry, MPSInfo
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

# initialize block program
ham = BlockHamiltonian(fcidump=fcidump, point_group=point_group, dot=dot, spin_adapted=spin_adapted, output_level=0)
print("block site basis = ", BlockSymmetry.initial_state_info(0))

info = MPSInfo.from_line_coupling(lc)
info.init_state_info()

rot_mats = {}
for i in range(0, ham.n_sites):
    rot_mats[tuple(range(0, i + 1))] = info.get_left_rotation_matrix(i, mps[i])

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
