#
#    pyblock: Spin-adapted quantum chemistry DMRG in MPO language (based on Block C++ code)
#    Copyright (C) 2019-2020 Huanchen Zhai
#
#    Block 1.5.3: density matrix renormalization group (DMRG) algorithm for quantum chemistry
#    Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012
#    Copyright (C) 2012 Garnet K.-L. Chan
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Matrix Product State.
"""

from .tensor import Tensor, TensorNetwork


class MPS(TensorNetwork):
    """Matrix Product State."""
    def __init__(self, tensors=None):
        super().__init__(tensors)

    @staticmethod
    def from_line_coupling(lcp):
        """Return MPS built from LineCoupling."""
        mps = MPS()
        r = {lcp.empty: 1}
        for i, post in enumerate(lcp.dims):
            mps.tensors.append(Tensor.rank3_init(r, lcp.basis[i], post))
            mps.tensors[-1].build_rank3_cg()
            mps.tensors[-1].tags = {i}
            r = post
        return mps

    def randomize(self):
        """Fill MPS reduced matrices with random numbers in [0, 1)."""
        for ts in self.tensors:
            ts.build_random()

    def build_identity(self):
        """Fill MPS reduced matrices with identity matrices (whenever possible)."""
        for ts in self.tensors:
            ts.build_zero()
            ts.build_identity()

    # at: where to divide the tensor into matrix => (0, at) x (at, n_ranks)
    def left_normalize(self):
        """Left normalization."""
        for its in range(len(self) - 1):
            rs = self.tensors[its].left_normalize()
            self.tensors[its + 1].left_multiply(rs)

    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, i):
        return self.tensors[i]

    def __setitem__(self, i, tensor):
        self.tensors[i] = tensor

    def __repr__(self):
        return "\n".join(
            "=== site %5d (NZB=) %5d ===\n%r" % (i, t.n_blocks, t)
            for i, t in enumerate(self.tensors))
