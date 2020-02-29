
from pyblock.symmetry.basis import slater_basis, su2_basis, basis_transform
from pyblock.symmetry.symmetry import ParticleN, SU2Proj, SU2

import numpy as np
import pytest
import os
from fractions import Fraction

class TestSU2CG:
    # <j1 m1 j2 m2 |0 0> = \delta_{j1,j2} \delta_{m1,-m2} (-1)^{j1-m1}/\sqrt{2j1+1}
    def test_su2_J0(self):
        for _ in range(50):
            J = SU2(0)
            j1 = SU2(Fraction(np.random.randint(0, 20), 2))
            j2 = (J + (-j1))[0]
            m1 = SU2Proj.random_from_multi(j1)
            if np.random.random() > 0.25:
                m2 = -m1
            else:
                m2 = SU2Proj.random_from_multi(j2)
            cg = SU2.clebsch_gordan(j1, j2, J)[int(m1.jz + j1.j), int(m2.jz + j2.j), 0]
            x = (-1) ** int(j1.j - m1.jz) / np.sqrt(int(2 * j1.j) + 1)
            x *= (0 if j1 != j2 else 1) * (0 if m1 != -m2 else 1)
            assert np.isclose(cg, x)
    
    # <j1 j1 j2 j2 | (j1+j2) (j1+j2)> = 1
    def test_su2_m_eq_j(self):
        for _ in range(50):
            j1 = SU2(Fraction(np.random.randint(0, 10), 2))
            j2 = SU2(Fraction(np.random.randint(0, 10), 2))
            J = (j1 + j2)[-1]
            cg = SU2.clebsch_gordan(j1, j2, J)[-1, -1, -1]
            assert np.isclose(cg, 1)
    
    # j1=j2=J/2 and m1 = -m2
    def test_su2_eq_j1(self):
        for _ in range(50):
            j1 = SU2(Fraction(np.random.randint(0, 10), 2))
            J = SU2(j1.ir * 2)
            m1 = SU2Proj.random_from_multi(j1)
            m2 = -m1
            cg = SU2.clebsch_gordan(j1, j1, J)[int(m1.jz + j1.j), int(m2.jz + j1.j), int(0 + J.j)]
            x = np.math.factorial(int(2 * j1.j)) ** 2 / np.sqrt(np.math.factorial(int(4 * j1.j)))
            x /= np.math.factorial(int(j1.j - m1.jz)) * np.math.factorial(int(j1.j + m1.jz))
            assert np.isclose(cg, x)
    
    # <j1 (M-1/2) 1/2  1/2|(j1 \pm 1/2) M> = \pm\sqrt{1/2 (1 \pm M/(j1 + 1/2))}
    # <j1 (M+1/2) 1/2 -1/2|(j1 \pm 1/2) M> =    \sqrt{1/2 (1 \mp M/(j1 + 1/2))}
    def test_su2_half(self):
        for _ in range(50):
            j1 = SU2(Fraction(np.random.randint(0, 20), 2))
            j2 = SU2(Fraction(1, 2))
            m2 = SU2Proj.random_from_multi(j2)
            Js = j1 + j2
            for ij, J in enumerate(Js[::-1]):
                M = SU2Proj.random_from_multi(J)
                m1 = SU2Proj.random_from_multi(j1)
                if abs(M.pir - 1) <= m1.ir:
                    m1 = SU2Proj(m1.ir, M.pir - 1)
                    m2 = SU2Proj(m2.ir, 1)
                    cg = SU2.clebsch_gordan(j1, j2, J)[int(m1.jz + j1.j), int(m2.jz + j2.j), int(M.jz + J.j)]
                    x = (-1) ** ij * np.sqrt(0.5 * (1 + (-1) ** ij * float(M.jz / (j1.j + 0.5))))
                    assert np.isclose(cg, x)
                if abs(M.pir + 1) <= m1.ir:
                    m1 = SU2Proj(m1.ir, M.pir + 1)
                    m2 = SU2Proj(m2.ir, -1)
                    cg = SU2.clebsch_gordan(j1, j2, J)[int(m1.jz + j1.j), int(m2.jz + j2.j), int(M.jz + J.j)]
                    x = np.sqrt(0.5 * (1 - (-1) ** ij * float(M.jz / (j1.j + 0.5))))
                    assert np.isclose(cg, x)
            

class TestBasis:
    @staticmethod
    def _assignment(a, i, j, x):
        a[i, j] = x
        return a
    _assign = _assignment.__func__
    Zero4 = np.zeros((4, 4), dtype=float)
    Identity4 = np.identity(4, dtype=float)
    Zero3 = np.zeros((3, 3), dtype=float)
    Identity3 = np.identity(3, dtype=float)
    SZ4 = np.diag(np.array([0, -0.5, 0.5, 0], dtype=float))
    NPlusNMinus4 = np.diag(np.array([0, 0, 0, 1], dtype=float))
    NTotal4 = np.diag(np.array([0, 1, 1, 2], dtype=float))
    NPlusNMinus3 = np.diag(np.array([0, 0, 1], dtype=float))
    NTotal3 = np.diag(np.array([0, 1, 2], dtype=float))
    CreAlpha = _assign(_assign(np.zeros((4, 4), dtype=float), 3, 1, 1), 2, 0, 1)
    DesAlpha = CreAlpha.T
    CreBeta = _assign(_assign(np.zeros((4, 4), dtype=float), 3, 2, -1), 1, 0, 1)
    DesBeta = CreBeta.T
    Cre3 = _assign(_assign(np.zeros((3, 3), dtype=float), 1, 0, 1), 2, 1, -np.sqrt(2))
    Des3 = _assign(_assign(np.zeros((3, 3), dtype=float), 0, 1, np.sqrt(2)), 1, 2, 1.0)
    Empty4 = ParticleN(0) * SU2Proj(0, 0)
    CreBetaQ = ParticleN(1) * SU2Proj(Fraction(1, 2), Fraction(-1, 2))
    CreAlphaQ = ParticleN(1) * SU2Proj(Fraction(1, 2), Fraction(1, 2))
    
    def test_zero(self):
        with pytest.raises(TypeError):
            basis_transform(self.Zero4, self.Empty4, slater_basis, [ParticleN(0) * ParticleN(0)])
        with pytest.raises(TypeError):
            basis_transform(self.Zero4, self.Empty4, [ParticleN(0) * ParticleN(0)], su2_basis)
        assert np.allclose(self.Zero3, basis_transform(self.Zero4, self.Empty4, slater_basis, su2_basis))
        assert np.allclose(self.Zero4, basis_transform(self.Zero3, self.Empty4, su2_basis, slater_basis))
    
    def test_identity(self):
        assert np.allclose(self.Identity3, basis_transform(self.Identity4, self.Empty4, slater_basis, su2_basis))
        assert np.allclose(self.Identity4, basis_transform(self.Identity3, self.Empty4, su2_basis, slater_basis))
    
    def test_cre_des_alpha(self):
        assert np.allclose(self.Cre3, basis_transform(self.CreAlpha, self.CreAlphaQ, slater_basis, su2_basis))
        assert np.allclose(self.CreAlpha, basis_transform(self.Cre3, self.CreAlphaQ, su2_basis, slater_basis))
        assert np.allclose(self.Des3, basis_transform(self.DesAlpha, -self.CreAlphaQ, slater_basis, su2_basis))
        assert np.allclose(self.DesAlpha, basis_transform(self.Des3, -self.CreAlphaQ, su2_basis, slater_basis))
        
    def test_cre_des_beta(self):
        assert np.allclose(self.Cre3, basis_transform(self.CreBeta, self.CreBetaQ, slater_basis, su2_basis))
        assert np.allclose(self.CreBeta, basis_transform(self.Cre3, self.CreBetaQ, su2_basis, slater_basis))
        # DesBeta needs an extra sign as component of SU2 spin tensor operator
        assert np.allclose(self.Des3, basis_transform(-self.DesBeta, -self.CreBetaQ, slater_basis, su2_basis))
        assert np.allclose(-self.DesBeta, basis_transform(self.Des3, -self.CreBetaQ, su2_basis, slater_basis))
    
    def test_particle_number(self):
        nplus = self.CreAlpha @ self.DesAlpha
        nminus = self.CreBeta @ self.DesBeta
        assert np.allclose(self.NPlusNMinus4, nplus @ nminus)
        assert np.allclose(self.NTotal4, nplus + nminus)
        assert np.allclose(self.SZ4, 0.5 * (nplus - nminus))
        assert np.allclose(self.NPlusNMinus3, basis_transform(self.NPlusNMinus4, self.Empty4, slater_basis, su2_basis))
        assert np.allclose(self.NPlusNMinus4, basis_transform(self.NPlusNMinus3, self.Empty4, su2_basis, slater_basis))
        assert np.allclose(self.NTotal3, basis_transform(self.NTotal4, self.Empty4, slater_basis, su2_basis))
        assert np.allclose(self.NTotal4, basis_transform(self.NTotal3, self.Empty4, su2_basis, slater_basis))
