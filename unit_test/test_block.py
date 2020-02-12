
from pyblock.qchem.core import BlockSymmetry, BlockHamiltonian, BlockError
from pyblock.symmetry.symmetry import DirectProdGroup, PGD2H, SZ, SU2, ParticleN

from block.symmetry import SpinQuantum, SpinSpace, IrrepSpace

import numpy as np
import pytest
import fractions
import os
import contextlib

@contextlib.contextmanager
def get_block_hamil(fcidump, pg, su2):
    ham = BlockHamiltonian(fcidump=fcidump, point_group=pg, dot=2, spin_adapted=su2, output_level=0)
    try:
        yield ham
    finally:
        BlockHamiltonian.set_current_memory(0)
        BlockHamiltonian.release_memory()

@pytest.fixture
def data_dir(request):
    filename = request.module.__file__
    return os.path.join(os.path.dirname(filename), 'data')

@pytest.fixture
def rand_su2_dpg():
    def dpg():
        n = np.random.randint(0, 20)
        s = np.random.randint(0, n + 1)
        ir = np.random.randint(0, len(PGD2H.InverseElem))
        if s % 2 != n % 2:
            n += 1
        return ParticleN(n) * SU2(fractions.Fraction(s, 2)) * PGD2H(ir)
    return dpg

@pytest.fixture
def rand_u1_dpg():
    def dpg():
        n = np.random.randint(0, 20)
        s = np.random.randint(0, n + 1)
        ir = np.random.randint(0, len(PGD2H.InverseElem))
        if s % 2 != n % 2:
            n += 1
        return ParticleN(n) * SZ(fractions.Fraction(s, 2)) * PGD2H(ir)
    return dpg

@pytest.fixture
def rand_sq():
    def sq():
        n = np.random.randint(0, 20)
        s = np.random.randint(0, n + 1)
        ir = np.random.randint(0, len(PGD2H.InverseElem))
        if s % 2 != n % 2:
            n += 1
        return SpinQuantum(n, SpinSpace(s), IrrepSpace(ir))
    return sq

class TestBlockSymmetry:
    fcidump = 'N2.STO3G.FCIDUMP'
    pg = 'd2h'
    
    def test_su2_dpg_to_spin_quantum(self, data_dir, rand_su2_dpg):
        dpg = rand_su2_dpg()
        
        with get_block_hamil(os.path.join(data_dir, self.fcidump), self.pg, su2=True) as hamil:
            assert dpg.__class__ == DirectProdGroup
            sq = BlockSymmetry.to_spin_quantum(dpg)
            assert sq.__class__ == SpinQuantum
            assert sq.n == dpg.irs[0].ir
            assert sq.s.irrep == dpg.irs[1].ir
            assert sq.symm.irrep == dpg.irs[2].ir
            
        with get_block_hamil(os.path.join(data_dir, self.fcidump), self.pg, su2=False) as hamil:
            assert dpg.__class__ == DirectProdGroup
            with pytest.raises(BlockError):
                sq = BlockSymmetry.to_spin_quantum(dpg)
            
    def test_u1_dpg_to_spin_quantum(self, data_dir, rand_u1_dpg):
        dpg = rand_u1_dpg()
        
        with get_block_hamil(os.path.join(data_dir, self.fcidump), self.pg, su2=False) as hamil:
            assert dpg.__class__ == DirectProdGroup
            sq = BlockSymmetry.to_spin_quantum(dpg)
            assert sq.__class__ == SpinQuantum
            assert sq.n == dpg.irs[0].ir
            assert sq.s.irrep == dpg.irs[1].ir
            assert sq.symm.irrep == dpg.irs[2].ir
        
        with get_block_hamil(os.path.join(data_dir, self.fcidump), self.pg, su2=True) as hamil:
            assert dpg.__class__ == DirectProdGroup
            with pytest.raises(BlockError):
                sq = BlockSymmetry.to_spin_quantum(dpg)
    
    def test_spin_quantum_to_dpg(self, data_dir, rand_sq):
        sq = rand_sq()
        
        with get_block_hamil(os.path.join(data_dir, self.fcidump), self.pg, su2=True) as hamil:
            assert sq.__class__ == SpinQuantum
            dpg = BlockSymmetry.from_spin_quantum(sq)
            assert dpg.__class__ == DirectProdGroup
            assert sq.n == dpg.irs[0].ir
            assert sq.s.irrep == dpg.irs[1].ir
            assert sq.symm.irrep == dpg.irs[2].ir
            assert dpg.irs[1].__class__ == SU2
            
        with get_block_hamil(os.path.join(data_dir, self.fcidump), self.pg, su2=False) as hamil:
            assert sq.__class__ == SpinQuantum
            dpg = BlockSymmetry.from_spin_quantum(sq)
            assert dpg.__class__ == DirectProdGroup
            assert sq.n == dpg.irs[0].ir
            assert sq.s.irrep == dpg.irs[1].ir
            assert sq.symm.irrep == dpg.irs[2].ir
            assert dpg.irs[1].__class__ == SZ
    
    def test_state_info(self, data_dir, rand_su2_dpg, rand_u1_dpg):
        
        with get_block_hamil(os.path.join(data_dir, self.fcidump), self.pg, su2=True) as hamil:
            n = 10
            basis = [(rand_su2_dpg(), np.random.randint(20)) for _ in range(n)]
            si = BlockSymmetry.to_state_info(basis)
            assert len(si.quanta) == len(si.n_states)
            assert sum(si.n_states) == si.n_total_states
            assert all([basis[i][1] == si.n_states[i] for i in range(n)])
            assert all([basis[i][0] == BlockSymmetry.from_spin_quantum(si.quanta[i])
                        for i in range(n)])
            
        with get_block_hamil(os.path.join(data_dir, self.fcidump), self.pg, su2=False) as hamil:
            n = 10
            basis = [(rand_u1_dpg(), np.random.randint(20)) for _ in range(n)]
            si = BlockSymmetry.to_state_info(basis)
            assert len(si.quanta) == len(si.n_states)
            assert sum(si.n_states) == si.n_total_states
            assert all([basis[i][1] == si.n_states[i] for i in range(n)])
            assert all([basis[i][0] == BlockSymmetry.from_spin_quantum(si.quanta[i])
                        for i in range(n)])
    
    def test_initial_state_info(self, data_dir):
        
        with get_block_hamil(os.path.join(data_dir, self.fcidump), self.pg, su2=True) as hamil:
            i = np.random.randint(hamil.n_sites)
            si = BlockSymmetry.initial_state_info(i)
            assert len(si.quanta) == len(si.n_states)
            assert sum(si.n_states) == si.n_total_states
            assert len(si.quanta) == 3
            
        with get_block_hamil(os.path.join(data_dir, self.fcidump), self.pg, su2=False) as hamil:
            i = np.random.randint(hamil.n_sites)
            si = BlockSymmetry.initial_state_info(i)
            assert len(si.quanta) == len(si.n_states)
            assert sum(si.n_states) == si.n_total_states
            assert len(si.quanta) == 4
