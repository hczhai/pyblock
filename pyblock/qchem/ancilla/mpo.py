
from ..mpo import MPOInfo, MPO, OperatorTensor
from ..operator import OpElement, OpNames
import numpy as np

class AncillaMPOInfo(MPOInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_physical_sites = self.n_sites
        self.n_sites *= 2
        self._init_ancilla_operator_names()

    def _init_ancilla_operator_names(self):
        lop_names = [None] * self.n_sites
        rop_names = [None] * self.n_sites
        for i, name in enumerate(self.left_operator_names):
            lop_names[i * 2] = lop_names[i * 2 + 1] = name
        for i, name in enumerate(self.right_operator_names):
            rop_names[i * 2] = name
            if i * 2 - 1 >= 0:
                rop_names[i * 2 - 1] = name
        rop_names[-1] = np.zeros((1, ), dtype=object)
        rop_names[-1][0] = OpElement(OpNames.I, (), q_label=self.hamil.empty)
        self.left_operator_names = lop_names
        self.right_operator_names = rop_names

class AncillaMPO(MPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_physical_sites = self.n_sites
        self.n_sites *= 2
        self._init_ancilla_mpo_tensors()

    def _init_ancilla_mpo_tensors(self):
        tensors = [None] * self.n_sites
        for i, ts in enumerate(self.tensors):
            tensors[i * 2] = ts
            ts.tags = {i * 2}
            rshape = ts.mat.shape[1]
            mat = np.zeros((rshape, rshape), dtype=object)
            iop = OpElement(OpNames.I, (), q_label=self.hamil.empty)
            for j in range(rshape):
                mat[j, j] = iop
            ops = { iop : ts.ops[iop] }
            tensors[i * 2 + 1] = OperatorTensor(mat=mat, tags={i * 2 + 1}, ops=ops)
        self.tensors = tensors
