
from pyblock.qchem import BlockHamiltonian, DMRGContractor
from pyblock.qchem import MPSInfo, IdentityMPOInfo, IdentityMPO
from pyblock.qchem import DMRGDataPage, Simplifier, AllRules, NoTransposeRules
from pyblock.qchem.ancilla import LineCoupling, MPOInfo, MPS, MPO
from pyblock.qchem.thermal import FreeEnergy
from pyblock.algorithm import ExpoApply, Compress, Expect

import numpy as np
import pytest
import fractions
import os

@pytest.fixture
def data_dir(request):
    filename = request.module.__file__
    return os.path.join(os.path.dirname(filename), 'data')

@pytest.fixture(scope="module", params=[1, 2, 3, 4])
def dot_scheme(request):
    return request.param

class TestDMRGOneSite:
    def test_hubbard_ancilla(self, data_dir, tmp_path):
        fcidump = 'HUBBARD-L8-U2.FCIDUMP'
        pg = 'c1'
        page = DMRGDataPage(tmp_path / 'node0')
        simpl = Simplifier(AllRules())
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=2000, page=page, nelec=16) as hamil:
            assert hamil.n_electrons == hamil.n_sites * 2
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_thermal_limit()
            mps = MPS(lcp, center=0, dot=2)
            mps.fill_thermal_limit()
            mps.canonicalize()
            fe_hamil = FreeEnergy(hamil)
            fe_hamil.set_free_energy(mu=1.0)
            mpo = MPO(hamil)
            ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), simpl)
            te = ExpoApply(mpo, mps, bond_dims=50, beta=0.02, contractor=ctr)
            ener = te.solve(10)
            assert abs(ener - (-5.76826262)) < 1E-3
        page.clean()
    
    def test_hubbard_nnn_ancilla(self, data_dir, tmp_path, dot_scheme):
        fcidump = 'HUBBARD-L8-U2-NNN.FCIDUMP'
        pg = 'c1'
        page = DMRGDataPage(tmp_path / 'node0', n_frames=2)
        simpl = Simplifier(AllRules())
        bdims = 50
        beta = 0.02
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=2000, page=page, nelec=16) as hamil:
            assert hamil.n_electrons == hamil.n_sites * 2
            # Line coupling
            lcp_thermal = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp_thermal.set_thermal_limit()
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(bdims)
            # MPS
            mps_thermal = MPS(lcp_thermal, center=0, dot=1 if dot_scheme == 1 else 2, iprint=True)
            mps_thermal.fill_thermal_limit()
            mps_thermal.canonicalize()
            mps = MPS(lcp, center=0, dot=1 if dot_scheme == 1 else 2, iprint=True)
            mps.randomize()
            mps.canonicalize()
            mps_info_thermal = MPSInfo(lcp_thermal)
            mps_info = MPSInfo(lcp)
            mps_info_d = { '_BRA': mps_info, '_KET': mps_info_thermal }
            # MPO
            fe_hamil = FreeEnergy(hamil)
            fe_hamil.set_free_energy(mu=1.0)
            mpo = MPO(hamil)
            mpo_info = MPOInfo(hamil)
            # Identity MPO
            impo = IdentityMPO(mpo)
            impo_info = IdentityMPOInfo(mpo_info)
            # Compression
            ctr = DMRGContractor(mps_info_d, impo_info, Simplifier(NoTransposeRules()))
            cps = Compress(impo, mps, mps_thermal, bond_dims=bdims, contractor=ctr, noise=1E-4)
            norm = cps.solve(10, 1E-6)
            assert abs(norm - 1) <= 1E-6
            # Time Evolution
            mps0 = MPS.from_tensor_network(cps._b, mps_info, center=cps.center, dot=cps.dot)
            mps0.set_contractor(None)
            mps0_form = cps.bra_canonical_form
            ctr = DMRGContractor(mps_info, mpo_info, Simplifier(AllRules()))
            tto = dot_scheme if dot_scheme >= 3 else -1
            te = ExpoApply(mpo, mps0, bond_dims=bdims, beta=beta, contractor=ctr, canonical_form=mps0_form)
            ener = te.solve(10, forward=cps.forward, two_dot_to_one_dot=tto)
            assert abs(ener - (-8.30251649)) <= 8E-3
            # Expectation
            mps0 = te.mps
            normsq = te.normsqs[-1]
            fener = Expect(mpo, mps0, mps0, mps0.form, None, contractor=ctr).solve() / normsq
            assert abs(ener - fener) <= 1E-6
        page.clean()
