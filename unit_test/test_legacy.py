
from pyblock.qchem import BlockHamiltonian, MPS, MPSInfo, LineCoupling
from pyblock.legacy.block_dmrg import DMRG as BLOCK_DMRG

import pytest
import os
import shutil

@pytest.fixture
def data_dir(request):
    filename = request.module.__file__
    return os.path.join(os.path.dirname(filename), 'data')

@pytest.fixture(scope="module", params=[True, False])
def use_su2(request):
    return request.param

@pytest.fixture(scope="module", params=[True, False])
def forward(request):
    return request.param

class TestBlockDMRG:
    fcidump = 'N2.STO3G.FCIDUMP'
    pg = 'd2h'
    
    def test_block_warmup(self, data_dir, tmp_path):
        os.chdir(tmp_path)
        with BlockHamiltonian.get(os.path.join(data_dir, self.fcidump), self.pg, su2=True) as hamil:
            dmrg = BLOCK_DMRG("")
            ener = dmrg.dmrg(gen_block=False, forward=True)
            assert abs(ener - (-107.648250974014)) < 5E-6
        if os.path.isdir('node0'):
            shutil.rmtree('node0')
    
    def test_block_dmrg(self, data_dir, tmp_path, use_su2, forward):
        os.chdir(tmp_path)
        with BlockHamiltonian.get(os.path.join(data_dir, self.fcidump), self.pg, su2=use_su2) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            if forward:
                mps = MPS(lcp, center=hamil.n_sites, dot=0)
            else:
                mps = MPS(lcp, center=0, dot=0)
            mps.randomize()
            mps.canonicalize()
            mps_info = MPSInfo(lcp)
            rot_mats = {}
            if forward:
                for i in range(0, hamil.n_sites):
                    if use_su2:
                        rot_mats[tuple(range(0, i + 1))] = mps_info.get_left_rotation_matrix(i, mps[i])
                    else:
                        rot_mats[tuple(range(0, (i + 1) * 2))] = mps_info.get_left_rotation_matrix(i, mps[i])
            else:
                for i in range(hamil.n_sites - 1, -1, -1):
                    if use_su2:
                        rot_mats[tuple(range(i, hamil.n_sites))] = mps_info.get_right_rotation_matrix(i, mps[i])
                    else:
                        rot_mats[tuple(range(i * 2, hamil.n_sites * 2))] = mps_info.get_right_rotation_matrix(i, mps[i])
            dmrg = BLOCK_DMRG("")
            ener = dmrg.dmrg(gen_block=True, rot_mats=rot_mats, forward=forward)
            assert abs(ener - (-107.648250974014)) < 5E-6
        if os.path.isdir('node0'):
            shutil.rmtree('node0')
