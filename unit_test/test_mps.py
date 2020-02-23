
from pyblock.qchem.mps import LineCoupling, MPSInfo, MPS
from pyblock.qchem.core import BlockHamiltonian
from pyblock.symmetry.symmetry import SU2, ParticleN, PGD2H

import numpy as np
import pytest
import os
from fractions import Fraction

@pytest.fixture
def data_dir(request):
    filename = request.module.__file__
    return os.path.join(os.path.dirname(filename), 'data')

@pytest.fixture(scope="module", params=[True, False])
def use_identity(request):
    return request.param

@pytest.fixture(scope="module", params=['L', 'R'])
def canon(request):
    return request.param

class TestMPS:
    
    def test_mps(self):
        n = 26
        spatial = """Ag  Ag  Ag  Ag  Ag  Ag  B3u  B3u  B3u  B2u  B2u  B2u
            B1g  B1u  B1u  B1u  B1u  B1u  B1u  B2g  B2g
            B2g  B3g  B3g  B3g  Au""".split()
        empty = ParticleN(0) * (SU2(0) * PGD2H(0))
        assert empty.sub_group([1, 2]) == SU2(0) * PGD2H(0)
        assert empty.sub_group([0, 1]) == ParticleN(0) * SU2(0)
        basis = [{
            ParticleN(0) * SU2(0) * PGD2H(0): 1,
            ParticleN(1) * SU2(Fraction(1, 2)) * PGD2H(sp): 1,
            ParticleN(2) * SU2(0) * PGD2H(0): 1
        } for sp in spatial]
        target = ParticleN(8) * SU2(0) * PGD2H('Ag')
        assert not (target < target)
        lcp = LineCoupling(n, basis, empty, target)
        lcp.set_bond_dimension(-1)
        for i in range(n):
            assert lcp.left_dims_fci[i] == lcp.left_dims[i]
            assert lcp.right_dims_fci[i] == lcp.right_dims[i]
        lcp.set_bond_dimension(29)
        for i in range(n):
            for k, v in lcp.left_dims[i].items():
                assert v <= lcp.left_dims_fci[i][k]
            for k, v in lcp.right_dims[i].items():
                assert v <= lcp.right_dims_fci[i][k]
    
    def test_rotation_matrix(self, data_dir, use_identity, canon):
        fcidump = 'C2.BLOCK.FCIDUMP'
        pg = 'd2h'
        with BlockHamiltonian.get(os.path.join(data_dir, fcidump), pg, su2=True, output_level=-1,
                                  memory=1200) as hamil:
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(22)
            mps = MPS(lcp, center=hamil.n_sites - 2 if canon == 'L' else 0, dot=2)
            mps.randomize()
            if use_identity:
                mps.fill_identity()
            mps.canonicalize()
            info = MPSInfo(lcp)
            if canon == 'L':
                for i in range(0, hamil.n_sites - 2):
                    rot_mat = info.get_left_rotation_matrix(i, mps[i])
                    mpsx = info.from_left_rotation_matrix(i, rot_mat)
                    assert mpsx == mps[i]
            else:
                for i in range(mps.dot, hamil.n_sites):
                    rot_mat = info.get_right_rotation_matrix(i, mps[i])
                    mpsx = info.from_right_rotation_matrix(i, rot_mat)
                    assert mpsx == mps[i]
