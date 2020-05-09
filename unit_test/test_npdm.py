
from pyblock.qchem import BlockHamiltonian, DMRGContractor
from pyblock.qchem import DMRGDataPage, Simplifier, AllRules, PDM1Rules
from pyblock.qchem import LineCoupling, MPSInfo, MPOInfo, MPS, MPO
from pyblock.qchem.npdm import PDM1MPOInfo, PDM1MPO
from pyblock.qchem.operator import OpNames
from pyblock.algorithm import Expect, DMRG

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


class TestNPDM:
    
    def test_n2_sto3g_1pdm(self, data_dir, tmp_path, dot_scheme):
        fcidump = 'N2.STO3G.FCIDUMP'
        pg = 'd2h'
        page = DMRGDataPage(tmp_path / 'node0', n_frames=2)
        bdims = 200
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=2000, page=page) as hamil:

            # Line coupling
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(bdims)
            
            # MPS
            mps = MPS(lcp, center=0, dot=dot_scheme, iprint=True)
            mps.randomize()
            mps.canonicalize(random=True)
            mps_info = MPSInfo(lcp)
            
            mpo_info = MPOInfo(hamil)
            ctr = DMRGContractor(mps_info, mpo_info, Simplifier(AllRules()))
            ctr.page.activate({'_BASE'})
            mpo = MPO(hamil)

            pmpo_info = PDM1MPOInfo(hamil)
            pctr = DMRGContractor(mps_info, pmpo_info, Simplifier(PDM1Rules()))
            pctr.page.activate({'_BASE'})
            pmpo = PDM1MPO(hamil)
            
            dmrg = DMRG(mpo, mps, bond_dims=bdims, noise=[1E-3, 1E-5, 1E-6, 0], contractor=ctr)
            ener = dmrg.solve(20, 1E-6)
            mps00 = dmrg.mps
            assert abs(ener - (-107.648250974014)) <= 5E-6
            
            dm_std = dict([
                ((0, 0),  1.99998913158986e+00),
                ((0, 1),  2.27456443442150e-05),
                ((0, 2),  2.39379512051723e-04),
                ((1, 0),  2.27456443442150e-05),
                ((1, 1),  1.99161589208801e+00),
                ((1, 2),  5.40347140948177e-03),
                ((2, 0),  2.39379512051723e-04),
                ((2, 1),  5.40347140948177e-03),
                ((2, 2),  1.98594977433493e+00),
                ((3, 3),  7.48715437741265e-02),
                ((4, 4),  7.48715443946934e-02),
                ((5, 5),  1.99999269427220e+00),
                ((5, 6),  2.36301723983430e-04),
                ((5, 7), -1.69189037164526e-04),
                ((6, 5),  2.36301723983430e-04),
                ((6, 6),  1.98642280564933e+00),
                ((6, 7),  1.79482656862581e-02),
                ((7, 5), -1.69189037164526e-04),
                ((7, 6),  1.79482656862581e-02),
                ((7, 7),  1.89717069904255e-02),
                ((8, 8),  1.93365745553765e+00),
                ((9, 9),  1.93365745136878e+00)
            ])
            
            pctr.mps_info = copy.deepcopy(mps_info)
            mps0 = copy.deepcopy(mps00)
            ex = Expect(pmpo, mps0, mps0, mps0.form, None, contractor=pctr)
            ex.solve(forward=dmrg.forward, bond_dim=bdims)
            dm = ex.get_1pdm_spatial()
            
            for i in range(hamil.n_sites):
                for j in range(hamil.n_sites):
                    if (i, j) not in dm_std:
                        assert abs(dm[i, j]) < 1E-10
                    else:
                        assert abs(dm[i, j] - dm_std[(i, j)]) < 5E-4
            
        page.clean()
