
# PYTHONPATH=../..:../../build python compress.py
# E = -6.225634098701 (L8)
# E = -12.966715281726584 (L16)

from pyblock.qchem import *
from pyblock.algorithm import DMRG, Compress

page = DMRGDataPage(save_dir='node0', n_frames=2)
bond_dim = 100

with BlockHamiltonian.get(fcidump='HUBBARD-L8.FCIDUMP', pg='c1', su2=True, output_level=-1, page=page, memory=4000) as hamil:
    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    lcp.set_bond_dimension(50)
    mps = MPS(lcp, center=0, dot=2, iprint=True)
    mps.randomize()
    mps.canonicalize()
    mpo = MPO(hamil, iprint=True)
    mps_info = MPSInfo(lcp)
    mpo_info = MPOInfo(hamil)
    ctr = DMRGContractor(mps_info, mpo_info, Simplifier(AllRules()))
    dmrg = DMRG(mpo, mps, bond_dims=bond_dim, contractor=ctr)
    ener = dmrg.solve(10, 1E-6)
    print('final energy = ', ener)
    mps0 = MPS.from_tensor_network(dmrg._k, mps_info, center=dmrg.center, dot=dmrg.dot)
    mps0_form = dmrg.canonical_form

    bond_dim = 150

    lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
    print('left dims = ', [sum(p.values()) for p in lcp.left_dims])
    print('right dims = ', [sum(p.values()) for p in lcp.right_dims])
    print('left min dims = ', [len(p) for p in lcp.left_dims_fci])
    print('right min dims = ', [len(p) for p in lcp.right_dims_fci])
    lcp.set_bond_dimension(bond_dim)
    mps = MPS(lcp, center=0, dot=2, iprint=True)
    mps.randomize()
    mps.canonicalize()
    mps_info = { '_BRA': MPSInfo(lcp), '_KET': ctr.mps_info }
    impo = IdentityMPO(mpo)
    impo_info = IdentityMPOInfo(mpo_info)
    ctr = DMRGContractor(mps_info, impo_info, Simplifier(NoTransposeRules()))
    mps0.set_contractor(ctr)
    cps = Compress(impo, mps, mps0, bond_dims=bond_dim, contractor=ctr, noise=1E-4, ket_canonical_form=mps0_form)
    ener = cps.solve(10, 1E-6)
    print('final energy = ', ener)

