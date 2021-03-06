
from pyblock.qchem.fcidump import TInt, VInt, read_fcidump, write_fcidump

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
    
    def test_write_fcidump_uhf(self, data_dir, tmp_path):
        filename = 'N2.STO3G-UHF.FCIDUMP'
        cont_dict, (t, v, e) = read_fcidump(os.path.join(data_dir, filename))
        assert isinstance(t, tuple) and isinstance(v, tuple)
        assert len(t) == 2 and len(v) == 4
        nmo = int(cont_dict['norb'])
        nelec = int(cont_dict['nelec'])
        orbsym = list(map(int, cont_dict['orbsym']))
        ms2 = int(cont_dict['ms2'])
        ta = np.array([[t[0][i, j] for j in range(t[0].n)] for i in range(t[0].n)])
        tb = np.array([[t[1][i, j] for j in range(t[1].n)] for i in range(t[1].n)])
        vaa = v[0].data
        vbb = v[3].data
        vab = v[1].data.reshape((v[1].m, v[1].m))
        write_fcidump(os.path.join(tmp_path, filename), (ta, tb), (vaa, vab, vbb), nmo, nelec, e, ms2, orbsym=orbsym)
        cont_dict2, (t2, v2, e2) = read_fcidump(os.path.join(data_dir, filename))
        assert cont_dict == cont_dict2
        assert t == t2 and v == v2 and e == e2

    def test_write_fcidump_rhf(self, data_dir, tmp_path):
        filename = 'N2.STO3G.FCIDUMP'
        cont_dict, (t, v, e) = read_fcidump(os.path.join(data_dir, filename))
        assert not isinstance(t, tuple) and not isinstance(v, tuple)
        nmo = int(cont_dict['norb'])
        nelec = int(cont_dict['nelec'])
        orbsym = list(map(int, cont_dict['orbsym']))
        ms2 = int(cont_dict['ms2'])
        tx = np.array([[t[i, j] for j in range(t.n)] for i in range(t.n)])
        vx = v.data
        write_fcidump(os.path.join(tmp_path, filename), tx, vx, nmo, nelec, e, ms2, orbsym=orbsym)
        cont_dict2, (t2, v2, e2) = read_fcidump(os.path.join(data_dir, filename))
        assert cont_dict == cont_dict2
        assert t == t2 and v == v2 and e == e2

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
        
        # test alternate form and comment lines
        cont_dict_a, (ta, va, ea) = read_fcidump(os.path.join(data_dir, 'N2.STO3G.FCIDUMP'))
        cont_dict_b, (tb, vb, eb) = read_fcidump(os.path.join(data_dir, 'N2.STO3G-ALT.FCIDUMP'))
        assert cont_dict_a == cont_dict_b
        n = int(cont_dict_a['norb'])
        assert ta == tb
        assert ea == eb
        assert va != vb
        assert repr(va) != repr(vb)
        for i in range(n):
            assert va[i, i, i, i] == vb[i, i, i, i]
        for m in range(25):
            i = np.random.randint(n)
            j = np.random.randint(n)
            k = np.random.randint(n)
            l = np.random.randint(n)
            if i != j or j != k or k != l:
                assert np.isclose(vb[i, j, k, l], 0)
