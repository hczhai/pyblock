
from pyblock.davidson import Vector, Matrix, davidson

import numpy as np
import pytest

class TestDavidson:
    
    def test_7(self):
        
        for k in range(1, 5):
            
            a = np.array([[0,  2,  9,  2,  0,  4,  5],
                          [2,  0,  6,  5,  2,  5,  9],
                          [9,  6,  0,  4,  5,  8,  1],
                          [2,  5,  4,  0,  0,  3,  5],
                          [0,  2,  5,  0,  0,  2,  9],
                          [4,  5,  8,  3,  2,  0,  4],
                          [5,  9,  1,  5,  9,  4,  0]], dtype=float)

            b = [Vector(ib) for ib in np.eye(k, 7)]

            e = np.array([-16.24341216, -7.16254184, -5.12344007, -3.41825462, 0.68226548,
                          4.63978769, 26.62559552])

            ld, nb, _ = davidson(Matrix(a), b, k)
            e, v = np.linalg.eigh(a)
            
            assert np.allclose(e[:k], ld, rtol=1E-4)
            for ik in range(k):
                assert np.allclose(v[:, ik], nb[ik].data, rtol=1E-4) \
                    or np.allclose(-v[:, ik], nb[ik].data, rtol=1E-4)

    def test_random(self):
        
        for _ in range(4):
            n = np.random.randint(400, 1500)
            k = np.random.randint(1, 5)

            a = np.random.random((n, n))
            a = (a + a.T) / 2

            b = [Vector(ib) for ib in np.eye(k, n)]

            ld, nb, _ = davidson(Matrix(a), b, k, deflation_max_size=max(5, k + 10), max_iter=n * 2)
            e, v = np.linalg.eigh(a)

            assert len(ld) == k
            assert np.allclose(e[:k], ld, atol=1E-6)
            for ik in range(k):
                assert np.linalg.norm(v[:, ik] - nb[ik].data) / n < 1E-3 \
                    or np.linalg.norm(v[:, ik] + nb[ik].data) / n < 1E-3
