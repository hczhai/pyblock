
from .tensor import Tensor, TensorNetwork

# Matrix Product State
class MPS(TensorNetwork):
    def __init__(self, tensors=None):
        super().__init__(tensors)

    @staticmethod
    def from_line_coupling(lcp):
        mps = MPS()
        r = {lcp.empty: 1}
        for i, post in enumerate(lcp.dims):
            mps.tensors.append(Tensor.rank3_init(r, lcp.basis[i], post))
            mps.tensors[-1].build_rank3_cg()
            mps.tensors[-1].tags = {i}
            r = post
        return mps

    def randomize(self):
        for ts in self.tensors:
            ts.build_random()

    def build_identity(self):
        for ts in self.tensors:
            ts.build_zero()
            ts.build_identity()

    # at: where to divide the tensor into matrix => (0, at) x (at, n_ranks)
    def left_normalize(self):
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
