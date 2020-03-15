
from ..mps import LineCoupling, MPS
from ...symmetry.symmetry import ParticleN, SU2
import collections
import numpy as np


class AncillaLineCoupling(LineCoupling):
    def __init__(self, n_sites, basis, empty, target):
        self.n_physical_sites = n_sites
        self.physical_basis = basis
        basis = [basis[i // 2] for i in range(n_sites * 2)]
        super().__init__(n_sites * 2, basis, empty, target)

    def set_bond_dimension_initial(self, *args, **kwargs):
        """
        This forces one possible quantum number between physical sites, which is only true for initial state.
        For some reason currently this does not work for Hubbard NNN model.
        It is recommended that `set_bond_dimension` should be used instead in most cases.
        """
        for i, d in enumerate(self.left_dims):
            if i % 2 == 1:
                target = ParticleN(i + 1) * SU2(0)
                self.left_dims[i] = collections.Counter({ k: v for k, v in d.items() if k.sub_group([0, 1]) == target })
        for i, d in enumerate(self.right_dims):
            if i % 2 == 0:
                target = ParticleN(self.n_sites - i) * SU2(0)
                self.right_dims[i] = collections.Counter({ k: v for k, v in d.items() if k.sub_group([0, 1]) == target })
        if self.target is not None:
            self._filter()
        super().set_bond_dimension(*args, **kwargs)

    def set_thermal_limit(self):
        self.left_dims = self._fill_ancilla_from_left()
        self.right_dims = self._fill_ancilla_from_right()

    def _fill_ancilla_from_left(self):
        dim_l = [None] * self.n_sites
        ld = None
        for d in range(0, self.n_sites):
            if d % 2 == 0:
                ld = dim_l[d] = self.tensor_product(ld, self.basis[d])
            else:
                t = sorted(ld.keys())[-1] + sorted(self.basis[d].keys())[0]
                if isinstance(t, list):
                    assert len(t) == 1
                    t = t[0]
                ld = dim_l[d] = collections.Counter({ t: 1 })
        return dim_l

    def _fill_ancilla_from_right(self):
        dim_r = [None] * self.n_sites
        rd = None
        for d in range(self.n_sites - 1, -1, -1):
            if d % 2 == 1:
                rd = dim_r[d] = self.tensor_product(self.basis[d], rd)
            else:
                t = sorted(self.basis[d].keys())[0] + sorted(rd.keys())[-1]
                if isinstance(t, list):
                    assert len(t) == 1
                    t = t[0]
                rd = dim_r[d] = collections.Counter({ t: 1 })
        return dim_r


class AncillaMPS(MPS):
    def fill_thermal_limit(self):
        for ts in self.tensors:
            if ts.rank == 3:
                assert len(ts.tags) == 1
                its = list(ts.tags)[0]
                d = len(self.lcp.basis[its])
                assert len(ts.blocks) == d
                if its % 2 == 0:
                    for ib, b in enumerate(ts.blocks):
                        assert tuple(b.reduced_shape) == (1, 1, 1)
                        b.reduced = np.array([[[1.0 / np.sqrt(4) if ib != 1 else 1.0 / np.sqrt(2)]]])
                else:
                    for ib, b in enumerate(ts.blocks):
                        assert tuple(b.reduced_shape) == (1, 1, 1)
                        b.reduced = np.array([[[1.0]]])
            else:
                assert ts.rank == 2 and self.dot == 2
                its = sorted(list(ts.tags))
                assert its[0] % 2 == 0
                assert len(ts.blocks) == d
                for ib, b in enumerate(ts.blocks):
                    assert tuple(b.reduced_shape) == (1, 1)
                    b.reduced = np.array([[1.0 / np.sqrt(4) if ib != 1 else 1.0 / np.sqrt(2)]])
