
from pyblock.qchem import BlockHamiltonian, DMRGContractor
from pyblock.qchem import MPSInfo
from pyblock.qchem.npdm import PDM1MPOInfo, PDM1MPO
from pyblock.qchem import DMRGDataPage, Simplifier, AllRules, NoTransposeRules, PDM1Rules
from pyblock.qchem.ancilla import LineCoupling, MPS, MPOInfo, MPO, Ancilla
from pyblock.qchem.ancilla import IdentityMPOInfo as AIMPOInfo, IdentityMPO as AIMPO
from pyblock.qchem.thermal import FreeEnergy
from pyblock.algorithm import ExpoApply, Compress, Expect

import numpy as np
import pytest
import fractions
import os
import copy

@pytest.fixture
def data_dir(request):
    filename = request.module.__file__
    return os.path.join(os.path.dirname(filename), 'data')

@pytest.fixture(scope="module", params=[1, 2, 3, 4])
def dot_scheme(request):
    return request.param

@pytest.fixture(scope="module", params=[True, False])
def use_su2(request):
    return request.param

class TestAncilla:
    def test_hubbard_ancilla(self, data_dir, tmp_path, use_su2):
        fcidump = 'HUBBARD-L8-U2.FCIDUMP'
        pg = 'c1'
        page = DMRGDataPage(tmp_path / 'node0')
        simpl = Simplifier(AllRules(su2=use_su2))
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=use_su2, output_level=-1,
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
    
    def test_hubbard_nnn_ancilla(self, data_dir, tmp_path, dot_scheme, use_su2):
        fcidump = 'HUBBARD-L8-U2-NNN.FCIDUMP'
        pg = 'c1'
        page = DMRGDataPage(tmp_path / 'node0', n_frames=3)
        bdims = 100
        beta = 0.02
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=use_su2, output_level=-1,
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
            mpo_info = MPOInfo(hamil)
            ctr = DMRGContractor(mps_info, mpo_info, Simplifier(AllRules(su2=use_su2)))
            ctr.page.activate({'_BASE'})
            mpo = MPO(hamil)

            pmpo_info = Ancilla.NPDM(PDM1MPOInfo)(hamil)
            pctr = DMRGContractor(mps_info, pmpo_info, Simplifier(PDM1Rules(su2=use_su2)))
            pctr.page.activate({'_BASE'})
            pmpo = Ancilla.NPDM(PDM1MPO)(hamil)

            impo_info = AIMPOInfo(hamil)
            ictr = DMRGContractor(mps_info_d, impo_info, Simplifier(NoTransposeRules(su2=use_su2)))
            ictr.page.activate({'_BASE'})
            impo = AIMPO(hamil)

            # Compression
            cps = Compress(impo, mps, mps_thermal, bond_dims=bdims, ket_bond_dim=10, contractor=ictr, noise=1E-4)
            norm = cps.solve(10, 1E-6)
            mps0 = cps.mps
            assert abs(norm - 1) <= 1E-6

            # Time Evolution
            tto = dot_scheme if dot_scheme >= 3 else -1
            te = ExpoApply(mpo, mps0, canonical_form=mps0.form, bond_dims=bdims, beta=beta, contractor=ctr)
            ener = te.solve(10, forward=cps.forward, two_dot_to_one_dot=tto)
            if use_su2:
                assert abs(ener - (-8.30251649)) <= 8E-3
            else:
                assert abs(ener - (-8.30251649)) <= 0.05 # bond dimension is worse for sz

            # Expectation
            mps0 = te.mps
            normsq = te.normsqs[-1]
            fener = Expect(mpo, mps0, mps0, mps0.form, None, contractor=ctr).solve() / normsq
            assert abs(ener - fener) <= 1E-6

            # 1-PDM
            pctr.mps_info = copy.deepcopy(mps_info)
            mps0p = copy.deepcopy(mps0)
            ex = Expect(pmpo, mps0p, mps0p, mps0p.form, None, contractor=pctr)
            ex.solve(forward=te.forward, bond_dim=bdims)

            if use_su2:
                dm = ex.get_1pdm_spatial(normsq=normsq)
            else:
                dm_spin = ex.get_1pdm(normsq=normsq)
                dm = dm_spin[:, :, 0, 0] + dm_spin[:, :, 1, 1]
                assert np.allclose(dm_spin[:, :, 0, 0], dm_spin[:, :, 1, 1], atol=1E-3)
                assert np.allclose(dm_spin[:, :, 0, 1], 0.0, atol=1E-3)
                assert np.allclose(dm_spin[:, :, 1, 0], 0.0, atol=1E-3)

            dm_std = np.array([
                [ 0.99917239,  0.09663464,  0.14507410, -0.00166227, -0.00134697, -0.00202504, -0.00095471,  0.00003930],
                [ 0.09663464,  0.99834462,  0.09566393,  0.14416750, -0.00233483, -0.00131371, -0.00203368, -0.00095530],
                [ 0.14507410,  0.09566393,  0.99753901,  0.09430095,  0.14315198, -0.00231163, -0.00131514, -0.00202803],
                [-0.00166227,  0.14416750,  0.09430095,  0.99757270,  0.09431342,  0.14315278, -0.00233495, -0.00134784],
                [-0.00134697, -0.00233483,  0.14315198,  0.09431342,  0.99757291,  0.09430054,  0.14416787, -0.00166256],
                [-0.00202504, -0.00131371, -0.00231163,  0.14315278,  0.09430054,  0.99753883,  0.09566390,  0.14507305],
                [-0.00095471, -0.00203368, -0.00131514, -0.00233495,  0.14416787,  0.09566390,  0.99834417,  0.09663385],
                [ 0.00003930, -0.00095530, -0.00202803, -0.00134784, -0.00166256,  0.14507305,  0.09663385,  0.99917258]
            ])

            assert np.allclose(dm, dm_std, atol=1E-3)

        page.clean()
