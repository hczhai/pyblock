
# PYTHONPATH=../..:../../build mpirun -n 4 python dmrg-simpl-tto-mpi.py
# E(dmrg) = -75.728386566258
# E(dmrg) = -75.72850291326984 (M=750)


from pyblock.qchem import *
from pyblock.dmrg import DMRG
from mpi4py import MPI
import numpy as np
import time
t = time.perf_counter()

np.random.seed(4)

scratch = '.'

pprint = lambda *args: print(*args) if MPI.COMM_WORLD.Get_rank() == 0 else None

if MPI.COMM_WORLD.Get_size() > 1:
    par = Parallelizer(ParaRule())
    page = DMRGDataPage(save_dir=scratch + '/node%d' % MPI.COMM_WORLD.Get_rank())
else:
    par = None
    page = DMRGDataPage(save_dir=scratch + '/node0')

is_root = MPI.COMM_WORLD.Get_rank() == 0

opts = dict(
    fcidump='C2.BLOCK.FCIDUMP',
    pg='d2h',
    su2=True,
    output_level=-1,
    memory=8000,
    omp_threads=2,
    page=page
)

bdims = [50, 80, 100, 200, 250]
noise = [1E-5, 1E-6, 1E-7, 1E-7, 1E-7, 0]
simpl = Simplifier(AllRules())

with BlockHamiltonian.get(**opts) as hamil:
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    pprint('left min dims = ', [len(p) for p in lcp.left_dims_fci])
    pprint('right min dims = ', [len(p) for p in lcp.right_dims_fci])
    lcp.set_bond_dimension(bdims[0])
    mps = MPS(lcp, center=0, dot=2, iprint=is_root)
    mps.randomize()
    mps.canonicalize()
    pprint('mps ok')
    mpo = MPO(hamil, iprint=is_root)
    pprint('mpo ok')
    ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), simpl)
    dmrg = DMRG(mpo, mps, bond_dim=bdims, noise=noise, contractor=ctr)
    pprint('init time = ', time.perf_counter() - t)
    ener = dmrg.solve(20, 1E-6, two_dot_to_one_dot=8)
    pprint('final energy = ', ener)
