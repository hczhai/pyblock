
from ..mpo import MPOInfo, MPO, OperatorTensor, DualOperatorTensor
from ..mpo import LocalMPOInfo, LocalMPO, ProdMPOInfo, ProdMPO
from ..mpo import SquareMPOInfo, SquareMPO, IdentityMPOInfo, IdentityMPO
from ..npdm.mpo import PDM1MPOInfo, PDM1MPO, NRMMPOInfo, NRMMPO
from ..operator import OpElement, OpNames
import numpy as np


class Ancilla:
    """
    MPO/MPOInfo Class decorator for adding ancilla sites.
    """
    def __init__(self, cls, npdm=False):
        self.cls = cls
        self.npdm = npdm

    def __call__(self, *args, **kwargs):
        x = self.cls(*args, **kwargs)
        x.n_physical_sites = x.n_sites
        x.n_sites *= 2
        if isinstance(x, MPOInfo):
            Ancilla._init_ancilla_operator_names(x, npdm=self.npdm)
        elif isinstance(x, MPO):
            Ancilla._init_ancilla_mpo_tensors(x)
        return x

    @staticmethod
    def _init_ancilla_operator_names(self, npdm):
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
        if self.middle_operators is not None:
            mops = [[]] * self.n_sites
            oph = OpElement(OpNames.H, ())
            for i, ops in enumerate(self.middle_operators):
                mops[i * 2] = ops
                # if NPDM, we do not need to repeat calculating expectation in ancilla sites
                if not npdm:
                    mops[i * 2 + 1] = ops
            self.middle_operators = mops

    @staticmethod
    def _init_ancilla_mpo_tensors(self):
        tensors = [None] * self.n_sites
        iop = OpElement(OpNames.I, (), q_label=self.hamil.empty)
        for i, ts in enumerate(self.tensors):
            if isinstance(ts, OperatorTensor):
                tensors[i * 2] = ts
                ts.tags = {i * 2}
                rshape = ts.mat.shape[1]
                mat = np.zeros((rshape, rshape), dtype=object)
                for j in range(rshape):
                    mat[j, j] = iop
                ops = { iop : ts.ops[iop] }
                tensors[i * 2 + 1] = OperatorTensor(mat=mat, tags={i * 2 + 1}, ops=ops)
            elif isinstance(ts, DualOperatorTensor):
                tensors[i * 2] = ts
                ts.tags = {i * 2}
                rshape = ts.lmat.shape[1]
                lmat = np.zeros((rshape, rshape), dtype=object)
                if i != self.n_physical_sites - 1:
                    lshape = self.tensors[i + 1].rmat.shape[0]
                else:
                    lshape = 1
                rmat = np.zeros((lshape, lshape), dtype=object)
                for j in range(rshape):
                    lmat[j, j] = iop
                for j in range(lshape):
                    rmat[j, j] = iop
                ops = { iop : ts.ops[iop] }
                tensors[i * 2 + 1] = DualOperatorTensor(lmat=lmat, rmat=rmat, tags={i * 2 + 1}, ops=ops)
        self.tensors = tensors

    @staticmethod
    def NPDM(cls):
        return Ancilla(cls, npdm=True)


@Ancilla
class AncillaMPOInfo(MPOInfo):
    pass


@Ancilla
class AncillaMPO(MPO):
    pass


@Ancilla
class AncillaLocalMPOInfo(LocalMPOInfo):
    pass


@Ancilla
class AncillaLocalMPO(LocalMPO):
    pass


@Ancilla
class AncillaSquareMPOInfo(SquareMPOInfo):
    pass


@Ancilla
class AncillaSquareMPO(SquareMPO):
    pass


@Ancilla
class AncillaProdMPOInfo(ProdMPOInfo):
    pass


@Ancilla
class AncillaProdMPO(ProdMPO):
    pass


@Ancilla
class AncillaIdentityMPOInfo(IdentityMPOInfo):
    pass


@Ancilla
class AncillaIdentityMPO(IdentityMPO):
    pass


@Ancilla.NPDM
class AncillaPDM1MPOInfo(PDM1MPOInfo):
    pass


@Ancilla.NPDM
class AncillaPDM1MPO(PDM1MPO):
    pass


@Ancilla.NPDM
class AncillaNRMMPOInfo(NRMMPOInfo):
    pass


@Ancilla.NPDM
class AncillaNRMMPO(NRMMPO):
    pass