
from pyblock.davidson import Vector, Matrix
from pyblock.expo import expo

import numpy as np
import pytest

class TestExpo:
    
    def test_7(self):
        
        a = np.array([[0,  2,  9,  2,  0,  4,  5],
                      [2,  0,  6,  5,  2,  5,  9],
                      [9,  6,  0,  4,  5,  8,  1],
                      [2,  5,  4,  0,  0,  3,  5],
                      [0,  2,  5,  0,  0,  2,  9],
                      [4,  5,  8,  3,  2,  0,  4],
                      [5,  9,  1,  5,  9,  4,  0]], dtype=float)
        ld, h = np.linalg.eigh(a)
        for t in np.arange(1.0, 10.0, 2.0):
            
            v = np.eye(1, 7)[0]
            b = Vector(v.copy())
            bb, _ = expo(Matrix(a), b, t)
            
            uu = h.T.dot(v)
            uu = np.diag(np.exp(-t * ld)).dot(uu)
            uu = h.dot(uu)
            
            assert np.allclose(uu, bb.data, rtol=1E-6)
        
    def test_random(self):
        
        for _ in range(4):
            n = np.random.randint(400, 1600)
            t = np.random.random() * 0.1 + 0.01

            a = np.random.random((n, n))
            a = (a + a.T) / 2
            
            v = np.random.random((n, ))
            b = Vector(v.copy())
            bb, _ = expo(Matrix(a), b, t)
            
            ld, h = np.linalg.eigh(a)
            ld = np.array(ld, dtype=np.float128)
            uu = h.T.dot(v)
            uu = np.diag(np.exp(-t * ld)).dot(uu)
            uu = h.dot(uu)

            assert np.allclose(uu, bb.data, atol=1E-6)
