
from pyblock.qchem import BlockHamiltonian, LineCoupling, DMRGContractor
from pyblock.qchem import MPSInfo, MPOInfo, MPS, MPO
from pyblock.qchem import DMRGDataPage, Simplifier, AllRules
from pyblock.dmrg import DMRG

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

@pytest.fixture(scope="module", params=[1, 2, 4])
def openmp_scheme(request):
    return request.param

@pytest.fixture(scope="module", params=[3, 4, 7])
def starting(request):
    return request.param

class TestDMRG:
    def test_hubbard(self, data_dir):
        fcidump = 'HUBBARD-L8.FCIDUMP'
        pg = 'c1'
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1, memory=200) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(50)
            mps = MPS(lcp, center=0, dot=2)
            mps.randomize()
            mps.canonicalize()
            mpo = MPO(hamil)
            ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil))
            dmrg = DMRG(mpo, mps, bond_dim=[50, 100], contractor=ctr)
            ener = dmrg.solve(10, 1E-6)
            assert abs(ener - (-6.225634098701)) < 5E-6
    
    def test_n2_sto3g(self, data_dir):
        fcidump = 'N2.STO3G.FCIDUMP'
        pg = 'd2h'
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1, memory=1200) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(100)
            mps = MPS(lcp, center=0, dot=2)
            mps.randomize()
            mps.canonicalize()
            mpo = MPO(hamil)
            ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil))
            dmrg = DMRG(mpo, mps, bond_dim=[100, 150, 200, 400, 500], noise=[1E-3, 1E-4, 1E-4, 1E-5, 0], contractor=ctr)
            ener = dmrg.solve(10, 1E-6)
            assert abs(ener - (-107.648250974014)) < 5E-6
    
    def test_n2_sto3g_data_page(self, data_dir, tmp_path):
        fcidump = 'N2.STO3G.FCIDUMP'
        pg = 'd2h'
        page = DMRGDataPage(tmp_path / 'node0')
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=1200, page=page) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(100)
            mps = MPS(lcp, center=0, dot=2)
            mps.randomize()
            mps.canonicalize()
            mpo = MPO(hamil)
            ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil))
            dmrg = DMRG(mpo, mps, bond_dim=[100, 150, 200, 400, 500], noise=[1E-3, 1E-4, 1E-4, 1E-5, 0], contractor=ctr)
            ener = dmrg.solve(10, 1E-6)
            assert abs(ener - (-107.648250974014)) < 5E-6
        page.clean()
    
    def test_n2_sto3g_simpl_dot(self, data_dir, tmp_path, dot_scheme):
        fcidump = 'N2.STO3G.FCIDUMP'
        pg = 'd2h'
        page = DMRGDataPage(tmp_path / 'node0')
        simpl = Simplifier(AllRules())
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=1200, page=page) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(100)
            mps = MPS(lcp, center=0, dot=1 if dot_scheme == 1 else 2)
            mps.randomize()
            mps.canonicalize()
            mpo = MPO(hamil)
            ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), simpl)
            tto = dot_scheme if dot_scheme >= 3 else -1
            dmrg = DMRG(mpo, mps, bond_dim=[100, 150, 200, 400, 500],
                        noise=[1E-3, 1E-4, 1E-4, 1E-5, 0], contractor=ctr)
            ener = dmrg.solve(10, 1E-6, two_dot_to_one_dot=tto)
            assert abs(ener - (-107.648250974014)) < 5E-6
        page.clean()
        
    def test_n2_sto3g_simpl_omp(self, data_dir, tmp_path, openmp_scheme):
        fcidump = 'N2.STO3G.FCIDUMP'
        pg = 'd2h'
        page = DMRGDataPage(tmp_path / 'node0')
        simpl = Simplifier(AllRules())
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=1200, page=page, omp_threads=openmp_scheme) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(100)
            mps = MPS(lcp, center=0, dot=1)
            mps.randomize()
            mps.canonicalize()
            mpo = MPO(hamil)
            ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), simpl)
            dmrg = DMRG(mpo, mps, bond_dim=[100, 150, 200, 400, 500],
                        noise=[1E-3, 1E-4, 1E-4, 1E-5, 0], contractor=ctr)
            ener = dmrg.solve(10, 1E-6)
            assert abs(ener - (-107.648250974014)) < 5E-6
        page.clean()
        
    def test_n2_sto3g_simpl_mid_tto(self, data_dir, tmp_path, starting):
        fcidump = 'N2.STO3G.FCIDUMP'
        pg = 'd2h'
        page = DMRGDataPage(tmp_path / 'node0')
        simpl = Simplifier(AllRules())
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=1200, page=page, omp_threads=2) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(100)
            mps = MPS(lcp, center=starting, dot=2)
            mps.randomize()
            mps.canonicalize()
            mpo = MPO(hamil)
            ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), simpl)
            dmrg = DMRG(mpo, mps, bond_dim=[100, 150, 200, 400, 500],
                        noise=[1E-3, 1E-4, 1E-4, 1E-5, 0], contractor=ctr)
            ener = dmrg.solve(10, 1E-6, two_dot_to_one_dot=3)
            assert abs(ener - (-107.648250974014)) < 5E-6
        page.clean()
    
    def test_n2_sto3g_simpl_mid_one(self, data_dir, tmp_path, starting):
        fcidump = 'N2.STO3G.FCIDUMP'
        pg = 'd2h'
        page = DMRGDataPage(tmp_path / 'node0')
        simpl = Simplifier(AllRules())
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=1200, page=page, omp_threads=2) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(100)
            mps = MPS(lcp, center=starting, dot=1)
            mps.randomize()
            mps.canonicalize()
            mpo = MPO(hamil)
            ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), simpl)
            dmrg = DMRG(mpo, mps, bond_dim=[100, 150, 200, 400, 500],
                        noise=[1E-3, 1E-4, 1E-4, 1E-5, 0], contractor=ctr)
            ener = dmrg.solve(10, 1E-6)
            assert abs(ener - (-107.648250974014)) < 5E-6
        page.clean()
