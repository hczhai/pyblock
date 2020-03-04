
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

class TestDMRGOneSite:
    def test_n2_sto3g_simpl_dot(self, data_dir, tmp_path, dot_scheme):
        fcidump = 'N2.STO3G.FCIDUMP'
        pg = 'd2h'
        page = DMRGDataPage(tmp_path / 'node0')
        simpl = Simplifier(AllRules())
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=2000, page=page) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(100)
            mps = MPS(lcp, center=0, dot=1 if dot_scheme == 1 else 2)
            mps.randomize()
            mps.canonicalize()
            mpo = MPO(hamil)
            ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), simpl)
            tto = dot_scheme if dot_scheme >= 3 else -1
            dmrg = DMRG(mpo, mps, bond_dims=[100, 150, 200, 400, 500],
                        noise=[1E-3, 1E-4, 1E-4, 1E-5, 0], contractor=ctr)
            ener = dmrg.solve(10, 1E-6, two_dot_to_one_dot=tto)
            assert abs(ener - (-107.648250974014)) < 5E-6
        page.clean()
    
    def test_n2_sto3g_simpl_exact(self, data_dir, tmp_path):
        fcidump = 'N2.STO3G.FCIDUMP'
        pg = 'd2h'
        page = DMRGDataPage(tmp_path / 'node0')
        simpl = Simplifier(AllRules())
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=2000, page=page) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(50, exact=True)
            mps = MPS(lcp, center=0, dot=2)
            mps.randomize()
            mps.canonicalize()
            mpo = MPO(hamil)
            ctr = DMRGContractor(MPSInfo(lcp), MPOInfo(hamil), simpl)
            dmrg = DMRG(mpo, mps, bond_dims=[50, 100, 150, 200, 400, 500],
                        noise=[1E-3, 1E-4, 1E-4, 1E-4, 1E-5, 0], contractor=ctr)
            ener = dmrg.solve(10, 1E-6, two_dot_to_one_dot=-1)
            assert abs(ener - (-107.648250974014)) < 5E-6
        page.clean()
