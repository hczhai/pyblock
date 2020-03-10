
from pyblock.qchem import BlockHamiltonian, DMRGContractor
from pyblock.qchem import MPSInfo, IdentityMPOInfo, IdentityMPO
from pyblock.qchem import DMRGDataPage, Simplifier, AllRules, NoTransposeRules
from pyblock.qchem.ancilla import LineCoupling as ALineCoupling, MPOInfo as AMPOInfo, MPS as AMPS, MPO as AMPO
from pyblock.qchem import LocalMPOInfo, LocalMPO, SquareMPOInfo, SquareMPO, LineCoupling, MPOInfo, MPS, MPO
from pyblock.qchem.thermal import FreeEnergy
from pyblock.qchem.operator import OpNames
from pyblock.algorithm import ExpoApply, Compress, Expect, DMRG

import numpy as np
import pytest
import fractions
import os
import copy

@pytest.fixture
def data_dir(request):
    filename = request.module.__file__
    return os.path.join(os.path.dirname(filename), 'data')

@pytest.fixture(scope="module", params=[1, 2])
def dot_scheme(request):
    return request.param

class TestExpect:
    
    def test_hubbard_expect(self, data_dir, tmp_path, dot_scheme):
        fcidump = 'HUBBARD-L8-U2.FCIDUMP'
        pg = 'c1'
        page = DMRGDataPage(tmp_path / 'node0', n_frames=6)
        simpl = Simplifier(AllRules())
        bdims = 50
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=2000, page=page) as hamil:
            assert hamil.n_electrons == hamil.n_sites
            
            # Line coupling
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(bdims)
            
            # MPS
            mps = MPS(lcp, center=0, dot=dot_scheme, iprint=True)
            mps.randomize()
            mps.canonicalize(random=True)
            mps_info = MPSInfo(lcp)
            
            # MPOInfo
            mpo_info = MPOInfo(hamil)
            impo_info = IdentityMPOInfo(mpo_info)
            nmpo_info = MPOInfo(hamil)
            lnmpo_info = LocalMPOInfo(mpo_info, OpNames.N)
            nnmpo_info = SquareMPOInfo(mpo_info, OpNames.N, OpNames.NN)
            
            # MPO
            fe_hamil = FreeEnergy(hamil)
            
            fe_hamil.set_energy()
            ctr = DMRGContractor(mps_info, mpo_info, Simplifier(AllRules()))
            ctr.page.activate({'_BASE'})
            mpo = MPO(hamil)
            
            ictr = DMRGContractor(mps_info, impo_info, Simplifier(AllRules()))
            ictr.page.activate({'_BASE'})
            impo = IdentityMPO(mpo)
            
            fe_hamil.set_particle_number()
            nctr = DMRGContractor(mps_info, nmpo_info, Simplifier(AllRules()))
            nctr.page.activate({'_BASE'})
            nmpo = MPO(hamil)
            
            lnctr = DMRGContractor(mps_info, lnmpo_info, Simplifier(AllRules()))
            lnctr.page.activate({'_BASE'})
            lnmpo = LocalMPO(mpo, OpNames.N)
            
            nnctr = DMRGContractor(mps_info, nnmpo_info, Simplifier(AllRules()))
            nnctr.page.activate({'_BASE'})
            nnmpo = SquareMPO(mpo, OpNames.N, OpNames.NN)
            
            dmrg = DMRG(mpo, mps, bond_dims=bdims, contractor=ctr)
            ener = dmrg.solve(10, 1E-6)
            mps00 = dmrg.mps
            assert abs(ener - (-6.22563379)) <= 1E-6
            
            ctrs = [ ctr, ictr, nctr, lnctr, nnctr ]
            mpos = [ mpo, impo, nmpo, lnmpo, nnmpo ]
            ress = [ -6.22563379, 1.0, 8.0, 8.0, 64.0 ]
            for xctr, xmpo, xstd in zip(ctrs, mpos, ress):
                xctr.mps_info = copy.deepcopy(mps_info)
                mps0 = copy.deepcopy(mps00)
                xr = Expect(xmpo, mps0, mps0, mps0.form, None, contractor=xctr).solve()
                assert abs(xr - xstd) <= 1E-6
                ex = Expect(xmpo, mps0, mps0, mps0.form, None, contractor=xctr)
                ex.solve(forward=dmrg.forward, bond_dim=bdims)
                assert np.allclose(ex.results, xstd, atol=1E-6)
        page.clean()
    
    def test_hubbard_ancilla_expect(self, data_dir, tmp_path, dot_scheme):
        fcidump = 'HUBBARD-L8-U2.FCIDUMP'
        pg = 'c1'
        page = DMRGDataPage(tmp_path / 'node0', n_frames=6)
        simpl = Simplifier(AllRules())
        bdims = 50
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=2000, page=page, nelec=16) as hamil:
            assert hamil.n_electrons == hamil.n_sites * 2
            
            # Line coupling
            lcp_thermal = ALineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp_thermal.set_thermal_limit()
            lcp = ALineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(bdims)
            
            # MPS
            mps_thermal = AMPS(lcp_thermal, center=0, dot=dot_scheme, iprint=True)
            mps_thermal.fill_thermal_limit()
            mps_thermal.canonicalize()
            mps = AMPS(lcp, center=0, dot=dot_scheme, iprint=True)
            mps.randomize()
            mps.canonicalize()
            mps_info_thermal = MPSInfo(lcp_thermal)
            mps_info = MPSInfo(lcp)
            mps_info_d = { '_BRA': mps_info, '_KET': mps_info_thermal }
            
            # MPOInfo
            mpo_info = AMPOInfo(hamil)
            impo_info = IdentityMPOInfo(mpo_info)
            empo_info = AMPOInfo(hamil)
            nmpo_info = AMPOInfo(hamil)
            lnmpo_info = LocalMPOInfo(mpo_info, OpNames.N)
            nnmpo_info = SquareMPOInfo(mpo_info, OpNames.N, OpNames.NN)
            
            # MPO
            fe_hamil = FreeEnergy(hamil)
            
            fe_hamil.set_free_energy(mu=1.0)
            ctr = DMRGContractor(mps_info, mpo_info, Simplifier(AllRules()))
            ctr.page.activate({'_BASE'})
            mpo = AMPO(hamil)
            
            ictr = DMRGContractor(mps_info_d, impo_info, Simplifier(NoTransposeRules()))
            ictr.page.activate({'_BASE'})
            impo = IdentityMPO(mpo)
            
            fe_hamil.set_energy()
            ectr = DMRGContractor(mps_info, empo_info, Simplifier(AllRules()))
            ectr.page.activate({'_BASE'})
            empo = AMPO(hamil)
            
            fe_hamil.set_particle_number()
            nctr = DMRGContractor(mps_info, nmpo_info, Simplifier(AllRules()))
            nctr.page.activate({'_BASE'})
            nmpo = AMPO(hamil)
            
            lnctr = DMRGContractor(mps_info, lnmpo_info, Simplifier(AllRules()))
            lnctr.page.activate({'_BASE'})
            lnmpo = LocalMPO(mpo, OpNames.N)
            
            nnctr = DMRGContractor(mps_info, nnmpo_info, Simplifier(AllRules()))
            nnctr.page.activate({'_BASE'})
            nnmpo = SquareMPO(mpo, OpNames.N, OpNames.NN)
            
            # Compression
            cps = Compress(impo, mps, mps_thermal, bond_dims=bdims, contractor=ictr, noise=1E-4)
            norm = cps.solve(10, 1E-6)
            mps00 = cps.mps
            assert abs(norm - 1) <= 1E-6
            
            ictr.mps_info = copy.deepcopy(ictr.mps_info)
            mps0 = copy.deepcopy(mps00)
            normsq = Expect(impo, mps0, mps_thermal, mps0.form, None, contractor=ictr).solve()
            assert abs(normsq - 1) <= 1E-6
            ex = Expect(impo, mps0, mps_thermal, mps0.form, None, contractor=ictr)
            ex.solve(forward=cps.forward, bond_dim=bdims)
            assert np.allclose(ex.results, 1, atol=1E-6)
            
            ctrs = [ ctr, ectr, nctr, lnctr, nnctr ]
            mpos = [ mpo, empo, nmpo, lnmpo, nnmpo ]
            ress = [ -4.0, 4.0, 8.0, 8.0, 68.0 ]
            for xctr, xmpo, xstd in zip(ctrs, mpos, ress):
                xctr.mps_info = copy.deepcopy(mps_info)
                mps0 = copy.deepcopy(mps00)
                xr = Expect(xmpo, mps0, mps0, mps0.form, None, contractor=xctr).solve() / normsq
                assert abs(xr - xstd) <= 1E-6
                ex = Expect(xmpo, mps0, mps0, mps0.form, None, contractor=xctr)
                ex.solve(forward=cps.forward, bond_dim=bdims)
                assert np.allclose(ex.results, xstd * normsq, atol=1E-6)

        page.clean()
