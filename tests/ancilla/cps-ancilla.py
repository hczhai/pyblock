
# PYTHONPATH=../..:../../build python cps-ancilla.py

from pyblock.qchem import *
from pyblock.qchem.ancilla import *
from pyblock.time_evolution import ExpoApply
from pyblock.compress import Compress
import numpy as np

np.random.seed(3)

opts = dict(
    fcidump='HUBBARD-L8-U2-NNN.FCIDUMP',
    pg='c1',
    su2=True,
    output_level=-1,
    memory=8000,
    omp_threads=1,
    page=DMRGDataPage(save_dir='node0', n_frames=2),
    nelec=16
)

bdims = 50
beta = 0.02
simpl = Simplifier(AllRules())
# simpl = None

with BlockHamiltonian.get(**opts) as hamil:
    print('target = ', hamil.target)
    assert hamil.n_electrons == hamil.n_sites * 2
    
    # Line Coupling
    lcp_thermal = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    lcp_thermal.set_thermal_limit()
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    lcp.set_bond_dimension(bdims)
    print('left dims = ', [sum(p.values()) for p in lcp.left_dims])
    print('right dims = ', [sum(p.values()) for p in lcp.right_dims])
    print('left min dims = ', [len(p) for p in lcp.left_dims_fci])
    print('right min dims = ', [len(p) for p in lcp.right_dims_fci])
    
    # MPS
    mps_thermal = MPS(lcp_thermal, center=0, dot=2, iprint=True)
    mps_thermal.fill_thermal_limit()
    mps_thermal.canonicalize()
    mps = MPS(lcp, center=0, dot=2, iprint=True)
    mps.randomize()
    mps.canonicalize()
    mps_info_thermal = MPSInfo(lcp_thermal)
    mps_info = MPSInfo(lcp)
    mps_info_d = { '_BRA': mps_info, '_KET': mps_info_thermal }
    
    # MPO
    mpo = MPO(hamil)
    mpo_info = MPOInfo(hamil)
    
    # Identity MPO
    impo = IdentityMPO(mpo)
    impo_info = IdentityMPOInfo(mpo_info)
    
    # Compression
    ctr = DMRGContractor(mps_info_d, impo_info, Simplifier(NoTransposeRules()))
    cps = Compress(impo, mps, mps_thermal, bond_dims=bdims, contractor=ctr, noise=1E-4)
    nrom = cps.solve(10, 1E-6)
    print('final nrom = ', nrom)
    
    # Time Evolution
    mps0 = MPS.from_tensor_network(cps._b, mps_info, center=cps.center, dot=cps.dot)
    mps0.set_contractor(None)
    mps0_form = cps.bra_canonical_form
    ctr = DMRGContractor(mps_info, mpo_info, Simplifier(AllRules()))
    te = ExpoApply(mpo, mps0, bond_dims=bdims, beta=beta, contractor=ctr, canonical_form=mps0_form)
    ener = te.solve(400, forward=cps.forward)
    print('final energy = ', ener)

