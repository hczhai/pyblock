
from pyblock.qchem.fcidump import TInt, VInt, read_fcidump

import numpy as np
import pytest
import os

@pytest.fixture
def data_dir(request):
    filename = request.module.__file__
    return os.path.join(os.path.dirname(filename), 'data')

class TestInt:
    
    def test_TInt(self):
        n = 10
        t = TInt(n)
        t.data = np.random.random(t.data.shape)
        for m in range(10):
            i = np.random.randint(n)
            j = np.random.randint(n)
            t[i, j] = np.random.random()
            assert np.isclose(t[j, i], t[i, j])
    
    def test_VInt(self):
        n = 10
        v = VInt(n)
        v.data = np.random.random(v.data.shape)
        for m in range(25):
            i = np.random.randint(n)
            j = np.random.randint(n)
            k = np.random.randint(n)
            l = np.random.randint(n)
            v[i, j, k, l] = np.random.random()
            assert np.isclose(v[j, i, k, l], v[i, j, k, l])
            assert np.isclose(v[i, j, l, k], v[i, j, k, l])
            assert np.isclose(v[k, l, i, j], v[i, j, k, l])
            assert np.isclose(v[k, l, j, i], v[i, j, k, l])
            assert np.isclose(v[l, k, j, i], v[i, j, k, l])

class TestFCIDUMP:
    
    def test_read_fcidump(self, data_dir):
        
        for filename in ['N2.STO3G.FCIDUMP', 'HUBBARD-L8.FCIDUMP']:
            
            cont_dict, (t, v, e) = read_fcidump(os.path.join(data_dir, filename))
            assert 'norb' in cont_dict
            assert 'nelec' in cont_dict
            assert 'orbsym' in cont_dict
            n = int(cont_dict['norb'])
            assert len(cont_dict['orbsym']) == n
            assert t.__class__ == TInt
            assert v.__class__ == VInt
            assert e.__class__ == float
            assert t.n == n
            assert v.n == n
