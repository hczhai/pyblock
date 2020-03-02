
from block.symmetry import state_tensor_product_target

from ..expo import expo
from .contractor import DMRGContractor, BlockMultiplyH, BlockWavefunction
import numpy as np

class TEContractor(DMRGContractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def expo(self, opt, mpst, beta):
        dot = len(mpst.tags - {'_KET'})

        mpst = mpst.copy()
        i = self._tag_site(mpst)
        assert dot == 2 or '_FUSE_L' in opt.tags or '_FUSE_R' in opt.tags or '_NO_FUSE' in opt.tags

        sts = None
        if dot == 2:
            st_l = self.mps_info.left_state_info_no_trunc[i]
            st_r = self.mps_info.right_state_info_no_trunc[i + 1]
        elif '_FUSE_L' in opt.tags:
            st_l = self.mps_info.left_state_info_no_trunc[i]
            if i + 1 < self.n_sites:
                st_r = self.mps_info.right_state_info[i + 1]
            else:
                st_r = BlockSymmetry.to_state_info([(self.mps_info.lcp.empty, 1)])
            sts = (st_l, st_r)
        elif '_FUSE_R' in opt.tags:
            if i - 1 >= 0:
                st_l = self.mps_info.left_state_info[i - 1]
            else:
                st_l = BlockSymmetry.to_state_info([(self.mps_info.lcp.empty, 1)])
            st_r = self.mps_info.right_state_info_no_trunc[i]
            sts = (st_l, st_r)
        elif '_NO_FUSE' in opt.tags:
            st_l = self.mps_info.left_state_info[i]
            st_r = self.mps_info.right_state_info[i + 1]
            sts = (st_l, st_r)

        wfn = self.mps_info.get_wavefunction_fused(i, mpst, dot=dot, sts=sts)

        st = state_tensor_product_target(st_l, st_r)
        a = BlockMultiplyH(opt, st)
        b = BlockWavefunction(wfn)

        vs, nexpo = expo(a, b, beta, const_a=self.mpo_info.hamil.e)

        hvs = vs.clear_copy()
        a.apply(vs, hvs)       
        energy = vs.dot(hvs) / vs.dot(vs)
        hvs.deallocate()
        
        if dot == 2 or '_FUSE_L' in opt.tags or '_NO_FUSE' in opt.tags:
            self.page.unload({i, '_LEFT'})
            self.page.unload({i + 1, '_RIGHT'})
        else:
            self.page.unload({i - 1, '_LEFT'})
            self.page.unload({i, '_RIGHT'})

        v = self.mps_info.from_wavefunction_fused(i, vs.data, sts=sts)
        return energy, v, nexpo

