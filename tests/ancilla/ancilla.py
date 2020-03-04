
# PYTHONPATH=../..:../../build python ancilla.py

from pyblock.qchem import *
from pyblock.qchem.ancilla import *
from pyblock.algorithm import ExpoApply
import numpy as np

np.random.seed(3)

opts = dict(
    fcidump='HUBBARD-L8-U2.FCIDUMP',
    pg='c1',
    su2=True,
    output_level=-1,
    memory=8000,
    omp_threads=1,
    page=DMRGDataPage(save_dir='node0'),
    nelec=16
)

bdims = 200
beta = 0.02
simpl = Simplifier(AllRules())
# simpl = None

with BlockHamiltonian.get(**opts) as hamil:
    print('target = ', hamil.target)
    assert hamil.n_electrons == hamil.n_sites * 2
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    lcp.set_thermal_limit()
    print('left dims = ', [len(p) for p in lcp.left_dims])
    print('right dims = ', [len(p) for p in lcp.right_dims])
    print('left min dims = ', [len(p) for p in lcp.left_dims_fci])
    print('right min dims = ', [len(p) for p in lcp.right_dims_fci])
    mps = MPS(lcp, center=0, dot=2, iprint=True)
    mps.fill_thermal_limit()
    mps.canonicalize()
    print('mps ok')
    mpo = MPO(hamil, iprint=True)
    print('mpo ok')
    ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), simpl)
    te = ExpoApply(mpo, mps, bond_dims=bdims, beta=beta, contractor=ctr)
    ener = te.solve(400)
    print('final energy = ', ener)

