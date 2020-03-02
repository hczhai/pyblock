
from pyblock.qchem import BlockHamiltonian, TEContractor
from pyblock.qchem import MPSInfo
from pyblock.qchem import DMRGDataPage, Simplifier, AllRules
from pyblock.qchem.ancilla import LineCoupling, MPOInfo, MPS, MPO
from pyblock.time_evolution import ExpoApply

import numpy as np
import pytest
import fractions
import os

@pytest.fixture
def data_dir(request):
    filename = request.module.__file__
    return os.path.join(os.path.dirname(filename), 'data')

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
            lcp.set_bond_dimension(100)
            mps = MPS(lcp, center=0, dot=2)
            mps.fill_thermal_limit()
            mps.canonicalize()
            mpo = MPO(hamil)
            ctr = TEContractor(MPSInfo(lcp), MPOInfo(hamil), simpl)
            te = ExpoApply(mpo, mps, bond_dim=50, beta=0.02, contractor=ctr)
            ener = te.solve(10)
            assert abs(ener - (-5.76826262)) < 1E-3
        page.clean()
    
