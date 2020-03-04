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
Specialized MPS/MPO operations for DMRG.
"""

from block.symmetry import state_tensor_product_target, VectorStateInfo
from block.operator import Wavefunction
from block.rev import tensor_scale, tensor_dot_product, tensor_scale_add_no_trans
from block.rev import tensor_precondition
from block.data_page import init_data_pages, release_data_pages, activate_data_page
from block.data_page import load_data_page, save_data_page
from block.data_page import set_data_page_pointer, get_data_page_pointer

from ..tensor.tensor import Tensor, SubTensor
from ..davidson import davidson
from ..expo import expo
from .core import BlockHamiltonian, BlockEvaluation, BlockSymmetry
from .simplifier import NoSimplifier
import numpy as np
import os
import shutil
import copy


class DataPage:
    def __init__(self):
        pass
    
    def get(self):
        return self
    
    def load(self, tags):
        pass
    
    def unload(self, tags):
        pass
    
    def activate(self, tags, reset=None):
        pass
    
    def save(self, tags):
        pass
    
    def initialize(self, tags):
        pass
    
    def release(self, tags):
        pass

class DMRGDataPage(DataPage):
    """Determine how to swap data between disk and memory for DMRG calculation."""
    def __init__(self, save_dir='node0', n_frames=1):
        self.current_pages = {}
        self.main_page = -1
        self.n_pages = 5
        self.save_dir = save_dir
        self.n_frames = n_frames
        self.i_frame = 0
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
    
    def get(self):
        assert self.i_frame < self.n_frames
        frame = copy.deepcopy(self)
        self.i_frame += 1
        return frame
    
    def _get_page(self, tags):
        """Return the data page id for the given data tags."""
        if tags == {'_BASE'}:
            return 0 + self.i_frame * self.n_pages
        site = [i for i in tags if isinstance(i, int)][0]
        if '_LEFT' in tags:
            return 1 + site % 2 + self.i_frame * self.n_pages
        elif '_RIGHT' in tags:
            return 3 + site % 2 + self.i_frame * self.n_pages
    
    def _get_file_name(self, tags):
        """Return the data page filename for the given data tags."""
        return os.path.join(self.save_dir, ".".join(sorted(map(str, tags))) + '.page.tmp')

    def load(self, tags):
        """Load data page from disk to memory, for reading data."""
        ip = self._get_page(tags)
        if ip not in self.current_pages:
            self.current_pages[ip] = tags
            load_data_page(ip, self._get_file_name(self.current_pages[ip]))
        else:
            assert self.current_pages[ip] == tags
    
    def unload(self, tags):
        """Unload data page in memory for reading data."""
        ip = self._get_page(tags)
        if ip in self.current_pages:
            assert self.current_pages[ip] == tags
            del self.current_pages[ip]
    
    def activate(self, tags, reset=False):
        """Activate one data page in memory for writing data."""
        ip = self._get_page(tags)
        if ip != self.main_page:
            if reset:
                assert ip not in self.current_pages
            self.current_pages[ip] = tags
            self.main_page = ip
            activate_data_page(ip)
            if reset:
                set_data_page_pointer(ip, 0)
    
    def save(self, tags):
        """Save the data page in memory to disk."""
        ip = self._get_page(tags)
        if ip in self.current_pages:
            assert self.current_pages[ip] == tags
            save_data_page(ip, self._get_file_name(self.current_pages[ip]))
        else:
            assert False
    
    def initialize(self):
        """Allocate memory for all pages."""
        init_data_pages(self.n_pages * self.n_frames)
        self.activate({'_BASE'})
    
    def release(self):
        """Deallocate memory for all pages."""
        release_data_pages()
    
    def clean(self):
        """Delete all temporary files."""
        shutil.rmtree(self.save_dir)
       
    def __repr__(self):
        return 'MAIN=%d ' % self.main_page + repr(self.current_pages)


class ContractionError(Exception):
    pass


class DMRGContractor:
    """bra_mps_info is MPSInfo of some constant MPS."""
    def __init__(self, mps_info, mpo_info, simplifier=None, parallelizer=None):
        self.mps_info = mps_info
        self.mpo_info = mpo_info
        self.n_sites = mpo_info.n_sites
        self.mem_ptr = 0
        self.rebuild = self.mpo_info.hamil.page is None
        if self.rebuild:
            self.page = DataPage().get()
        else:
            self.page = self.mpo_info.hamil.page.get()
        if simplifier is not None:
            BlockEvaluation.simplifier = simplifier
        if parallelizer is not None:
            BlockEvaluation.parallelizer = parallelizer
        self.is_parallel = parallelizer is not None
    
    def pre_sweep(self):
        """Operations performed at the beginning of each DMRG sweep."""
        if self.rebuild:
            self.mem_ptr = BlockHamiltonian.get_current_memory()
    
    def post_sweep(self):
        """Operations performed at the end of each DMRG sweep."""
        if self.rebuild:
            BlockHamiltonian.set_current_memory(self.mem_ptr)
    
    def _tag_site(self, tensor):
        tags = tensor.tags
        itag = None
        for tag in tags:
            if isinstance(tag, int):
                if itag is None or itag > tag:
                    itag = tag
        if itag is not None:
            return itag
        if '_LEFT' in tags:
            return -1
        elif '_RIGHT' in tags:
            return self.n_sites
        else:
            assert False
            return None
    
    def _get_state_info(self, dot, i, n_sites, opt, mps_info):
        sts = None
        if dot == 2:
            st_l = mps_info.left_state_info_no_trunc[i]
            st_r = mps_info.right_state_info_no_trunc[i + 1]
        elif '_FUSE_L' in opt.tags:
            st_l = mps_info.left_state_info_no_trunc[i]
            if i + 1 < n_sites:
                st_r = mps_info.right_state_info[i + 1]
            else:
                st_r = BlockSymmetry.to_state_info([(mps_info.lcp.empty, 1)])
            sts = (st_l, st_r)
        elif '_FUSE_R' in opt.tags:
            if i - 1 >= 0:
                st_l = mps_info.left_state_info[i - 1]
            else:
                st_l = BlockSymmetry.to_state_info([(mps_info.lcp.empty, 1)])
            st_r = mps_info.right_state_info_no_trunc[i]
            sts = (st_l, st_r)
        elif '_NO_FUSE' in opt.tags:
            st_l = mps_info.left_state_info[i]
            st_r = mps_info.right_state_info[i + 1]
            sts = (st_l, st_r)
        return st_l, st_r, sts

    def apply(self, opt, mpst):
        
        dot = len(mpst.tags - {'_KET', '_BRA'})

        mpst = mpst.copy()
        i = self._tag_site(mpst)
        assert dot == 2 or '_FUSE_L' in opt.tags or '_FUSE_R' in opt.tags or '_NO_FUSE' in opt.tags
        
        kst_l, kst_r, ksts = self._get_state_info(dot, i, self.n_sites, opt, self.mps_info['_KET'])
        bst_l, bst_r, bsts = self._get_state_info(dot, i, self.n_sites, opt, self.mps_info['_BRA'])
        
        super_sts = VectorStateInfo([state_tensor_product_target(bst_l, bst_r),
                                     state_tensor_product_target(kst_l, kst_r)])
        bopt = BlockMultiplyH(opt, super_sts, diag=False)

        bwfn = self.mps_info['_BRA'].get_wavefunction_fused(i, None, dot=dot, sts=bsts)
        kwfn = self.mps_info['_KET'].get_wavefunction_fused(i, mpst, dot=dot, sts=ksts)
        bkwfn = BlockWavefunction(kwfn)
        bbwfn = BlockWavefunction(bwfn)
        bopt.apply(bkwfn, bbwfn)
        
        bkwfn.deallocate()
        
        norm = np.sqrt(abs(bbwfn.dot(bbwfn)))
        
        if dot == 2 or '_FUSE_L' in opt.tags or '_NO_FUSE' in opt.tags:
            self.page.unload({i, '_LEFT'})
            self.page.unload({i + 1, '_RIGHT'})
        else:
            self.page.unload({i - 1, '_LEFT'})
            self.page.unload({i, '_RIGHT'})

        v = self.mps_info['_BRA'].from_wavefunction_fused(i, bbwfn.data, sts=bsts)
        
        bbwfn.deallocate()
        
        return norm, v, 0
    
    def expo_apply(self, opt, mpst, beta):
        
        dot = len(mpst.tags - {'_KET', '_BRA'})

        mpst = mpst.copy()
        i = self._tag_site(mpst)
        assert dot == 2 or '_FUSE_L' in opt.tags or '_FUSE_R' in opt.tags or '_NO_FUSE' in opt.tags
        
        st_l, st_r, sts = self._get_state_info(dot, i, self.n_sites, opt, self._get_mps_info({'_KET'}))
        
        super_sts = VectorStateInfo([state_tensor_product_target(st_l, st_r)])
        bopt = BlockMultiplyH(opt, super_sts, diag=True)

        kwfn = self._get_mps_info({'_KET'}).get_wavefunction_fused(i, mpst, dot=dot, sts=sts)
        bwfn = self._get_mps_info({'_BRA'}).get_wavefunction_fused(i, None, dot=dot, sts=sts)
        bkwfn = BlockWavefunction(kwfn)
        bbwfn = BlockWavefunction(bwfn)
        assert len(bkwfn.ref) == len(bbwfn.ref)
        vs, nexpo = expo(bopt, bkwfn, beta, const_a=self.mpo_info.hamil.e)
        
        bopt.apply(vs, bbwfn)
        normsq = vs.dot(vs)
        energy = vs.dot(bbwfn) / normsq
        bbwfn.deallocate()
        
        if dot == 2 or '_FUSE_L' in opt.tags or '_NO_FUSE' in opt.tags:
            self.page.unload({i, '_LEFT'})
            self.page.unload({i + 1, '_RIGHT'})
        else:
            self.page.unload({i - 1, '_LEFT'})
            self.page.unload({i, '_RIGHT'})

        v = self.mps_info.from_wavefunction_fused(i, vs.data, sts=sts)
        # v.align(mpst)
        
        vs.deallocate()
        
        return energy, normsq, v, nexpo
    
    def eigs(self, opt, mpst):
        """
        Davidson diagonalization.
        
        Args:
            opt : OperatorTensor
                Super block contracted operator tensor.
            mpst : Tensor
                Contracted MPS tensor in dot blocks.
        
        Returns:
            energy : float
                Ground state energy.
            v : class:`Tensor`
                In two-dot scheme, the rank-2 tensor representing two-dot object.
                Both left and right rank indices are fused.
                One-dot scheme is not implemented.
            ndav : int
                Number of Davidson iterations.
        """
        assert not isinstance(self.mps_info, dict)
        dot = len(mpst.tags - {'_KET'})
            
        mpst = mpst.copy()
        i = self._tag_site(mpst)
        
        assert dot == 2 or '_FUSE_L' in opt.tags or '_FUSE_R' in opt.tags
        
        st_l, st_r, sts = self._get_state_info(dot, i, self.n_sites, opt, self._get_mps_info({'_KET'}))

        wfn = self.mps_info.get_wavefunction_fused(i, mpst, dot=dot, sts=sts)

        super_sts = VectorStateInfo([state_tensor_product_target(st_l, st_r)])
        a = BlockMultiplyH(opt, super_sts)
        b = [BlockWavefunction(wfn)]
        
        es, vs, ndav = davidson(a, b, 1, mpi=self.is_parallel)
        
        if dot == 2 or '_FUSE_L' in opt.tags:
            self.page.unload({i, '_LEFT'})
            self.page.unload({i + 1, '_RIGHT'})
        else:
            self.page.unload({i - 1, '_LEFT'})
            self.page.unload({i, '_RIGHT'})
            
        if len(es) == 0:
            raise ContractionError('Davidson not converged!!')
            
        e = es[0]
        v = self.mps_info.from_wavefunction_fused(i, vs[0].data, sts=sts)
        
        vs[0].deallocate()
        
        return e + self.mpo_info.hamil.e, v, ndav
    
    def _get_mps_info(self, tags):
        if isinstance(self.mps_info, dict):
            if '_KET' in tags:
                tag = '_KET'
            elif '_BRA' in tags:
                tag = '_BRA'
            else:
                raise ContractionError('Cannot detetermine bra/ket mps info!!')
            return self.mps_info[tag]
        else:
            return self.mps_info
    
    def update_local_left_mps_info(self, i, l_fused):
        """Update :attr:`info` for site i using the left tensor from SVD."""
        block_basis = [(b.q_labels[1], b.reduced.shape[1]) for b in l_fused.blocks]
        self._get_mps_info(l_fused.tags).update_local_left_block_basis(i, block_basis)
        
    def update_local_right_mps_info(self, i, r_fused):
        """Update :attr:`info` for site i using the right tensor from SVD."""
        block_basis = [(b.q_labels[0], b.reduced.shape[0]) for b in r_fused.blocks]
        self._get_mps_info(r_fused.tags).update_local_right_block_basis(i, block_basis)
    
    def fuse_left(self, i, tensor, original_form):
        mps_info = self._get_mps_info(tensor.tags)
        st_l = mps_info.left_state_info_no_trunc[i]
        stlq = BlockSymmetry.from_state_info(st_l)
        if original_form == 'L':
            tensor.fuse_index(0, dict(stlq), equal=True)
        else:
            if original_form == 'R':
                for block in tensor.blocks:
                    assert block.q_labels[0] == mps_info.lcp.target
                    block.q_labels = (mps_info.lcp.empty, ) + block.q_labels[1:]
            tensor.fuse_index(0, dict(stlq), target=mps_info.lcp.target)
    
    def fuse_right(self, i, tensor, original_form):
        mps_info = self._get_mps_info(tensor.tags)
        st_r = mps_info.right_state_info_no_trunc[i]
        strq = BlockSymmetry.from_state_info(st_r)
        if original_form == 'R':
            tensor.fuse_index(1, dict(strq), equal=True)
        else:
            if original_form == 'L':
                for block in tensor.blocks:
                    assert block.q_labels[2] == mps_info.lcp.target
                    block.q_labels = block.q_labels[:2] + (mps_info.lcp.empty, )
            tensor.fuse_index(1, dict(strq), target=mps_info.lcp.target)
    
    def unfuse_left(self, i, tensor):
        mps_info = self._get_mps_info(tensor.tags)
        l = [(mps_info.lcp.empty, 1)] if i == 0 else mps_info.left_block_basis[i - 1]
        r = mps_info.basis[i]
        tensor = tensor.copy()
        tensor.unfuse_index(0, l, r)
        return tensor.set_tags({i}).set_contractor(self)
    
    def unfuse_right(self, i, tensor):
        mps_info = self._get_mps_info(tensor.tags)
        l = mps_info.basis[i]
        r = [(mps_info.lcp.empty, 1)] if i == self.n_sites - 1 else mps_info.right_block_basis[i + 1]
        tensor = tensor.copy()
        tensor.unfuse_index(1, l, r)
        return tensor.set_tags({i}).set_contractor(self)
    
    def bond_upper_limit_left(self, tags={}):
        return self._get_mps_info(tags).lcp.left_dims_fci
    
    def bond_upper_limit_right(self, tags={}):
        return self._get_mps_info(tags).lcp.right_dims_fci
    
    def bond_left(self, tags={}):
        return self._get_mps_info(tags).lcp.left_dims
    
    def bond_right(self, tags={}):
        return self._get_mps_info(tags).lcp.right_dims
    
    def contract(self, tn, tags):
        """
        Tensor network contraction.
        
        Args:
            tn : TensorNetwork
                Part of tensor network to be contracted.
            tags : (str, int) or (str, )
                Tags of the tensor network to be contracted.
                If tags = ('_LEFT', i), the contraction is corresponding to
                blocking and renormalizing left block at site i.
                If tags = ('_RIGHT', i), the contraction is corresponding to
                blocking and renormalizing right block at site i.
                If tags = ('_HAM'), the contraction is corresponding to
                blocking both left and right block and forming the super block hamiltonian.
        
        Returns:
            mpo : OperatorTensor
                The contracted MPO tensor.
        """
        assert isinstance(tags, tuple)
        if len(tags) == 2:
            dir, i = tags
        elif len(tags) == 3:
            dir, i, j = tags
        else:
            dir = tags[0]
        if isinstance(self.mps_info, dict):
            info_ket, info_bra = self.mps_info['_KET'], self.mps_info['_BRA']
        else:
            info_ket, info_bra = self.mps_info, None
        if dir == '_LEFT':
            ket = tn[{i, '_KET'}]
            bra = tn[{i, '_BRA'}]
            ham = tn[{i, '_HAM'}]
            self.page.activate({i, '_LEFT'}, reset=True)
            if i == 0:
                ham_rot = BlockEvaluation.left_rotate(i, ham, ket, info_ket, bra, info_bra)
                ham_rot.tags.add(dir)
            else:
                self.page.load({i - 1, '_LEFT'})
                ham_prev = tn[{i - 1, '_HAM', dir}]
                ham_ctr = BlockEvaluation.left_contract(i, ham_prev, ham, self.mpo_info, info_ket, info_bra)
                ham_rot = BlockEvaluation.left_rotate(i, ham_ctr, ket, info_ket, bra, info_bra)
                ham_rot.tags.add(dir)
                self.page.unload({i - 1, '_LEFT'})
            self.page.save({i, '_LEFT'})
            return ham_rot
        elif dir == '_RIGHT':
            ket = tn[{i, '_KET'}]
            bra = tn[{i, '_BRA'}]
            ham = tn[{i, '_HAM'}]
            self.page.activate({i, '_RIGHT'}, reset=True)
            if i == self.n_sites - 1:
                ham_rot = BlockEvaluation.right_rotate(i, ham, ket, info_ket, bra, info_bra)
                ham_rot.tags.add(dir)
            else:
                self.page.load({i + 1, '_RIGHT'})
                ham_prev = tn[{i + 1, '_HAM', dir}]
                ham_ctr = BlockEvaluation.right_contract(i, ham_prev, ham, self.mpo_info, info_ket, info_bra)
                ham_rot = BlockEvaluation.right_rotate(i, ham_ctr, ket, info_ket, bra, info_bra)
                ham_rot.tags.add(dir)
                self.page.unload({i + 1, '_RIGHT'})
            self.page.save({i, '_RIGHT'})
            return ham_rot
        elif dir == '_HAM':
            dot = len(tn) - 2
            ts = sorted(tn.tensors, key=self._tag_site)
            assert (dot == 2) ^ ('_FUSE_L' in ts[1].tags or '_FUSE_R' in ts[1].tags or '_NO_FUSE' in ts[1].tags)
            if dot == 2:
                if self._tag_site(ts[0]) != -1:
                    self.page.activate({self._tag_site(ts[1]), '_LEFT'}, reset=True)
                    self.page.load({self._tag_site(ts[0]), '_LEFT'})
                    ham_left = BlockEvaluation.left_contract(self._tag_site(ts[1]), ts[0], ts[1], self.mpo_info, info_ket, info_bra)
                    self.page.unload({self._tag_site(ts[0]), '_LEFT'})
                else:
                    ham_left = ts[1]
                if self._tag_site(ts[3]) != self.n_sites:
                    self.page.activate({self._tag_site(ts[2]), '_RIGHT'}, reset=True)
                    self.page.load({self._tag_site(ts[3]), '_RIGHT'})
                    ham_right = BlockEvaluation.right_contract(self._tag_site(ts[2]), ts[3], ts[2], self.mpo_info, info_ket, info_bra)
                    self.page.unload({self._tag_site(ts[3]), '_RIGHT'})
                else:
                    ham_right = ts[2]
            elif '_FUSE_L' in ts[1].tags:
                if self._tag_site(ts[0]) != -1:
                    self.page.activate({self._tag_site(ts[1]), '_LEFT'}, reset=True)
                    self.page.load({self._tag_site(ts[0]), '_LEFT'})
                    ham_left = BlockEvaluation.left_contract(self._tag_site(ts[1]), ts[0], ts[1], self.mpo_info, info_ket, info_bra)
                    self.page.unload({self._tag_site(ts[0]), '_LEFT'})
                else:
                    ham_left = ts[1]
                if self._tag_site(ts[2]) != self.n_sites:
                    self.page.load({self._tag_site(ts[2]), '_RIGHT'})
                    ham_right = ts[2]
                else:
                    ham_right = ts[2]
            elif '_FUSE_R' in ts[1].tags:
                if self._tag_site(ts[0]) != -1:
                    self.page.load({self._tag_site(ts[0]), '_LEFT'})
                    ham_left = ts[0]
                else:
                    ham_left = ts[0]
                if self._tag_site(ts[2]) != self.n_sites:
                    self.page.activate({self._tag_site(ts[1]), '_RIGHT'}, reset=True)
                    self.page.load({self._tag_site(ts[2]), '_RIGHT'})
                    ham_right = BlockEvaluation.right_contract(self._tag_site(ts[1]), ts[2], ts[1], self.mpo_info, info_ket, info_bra)
                    self.page.unload({self._tag_site(ts[2]), '_RIGHT'})
                else:
                    ham_right = ts[1]
            elif '_NO_FUSE' in ts[1].tags:
                self.page.load({self._tag_site(ts[0]), '_LEFT'})
                self.page.load({self._tag_site(ts[1]), '_RIGHT'})
                ham_left = ts[0]
                ham_right = ts[1]
            self.page.activate({'_BASE'}, reset=False)
            if dot == 2:
                tag = '_FUSE_LR'
            elif '_FUSE_L' in ts[1].tags:
                tag = '_FUSE_L'
            elif '_FUSE_R' in ts[1].tags:
                tag = '_FUSE_R'
            elif '_NO_FUSE' in ts[1].tags:
                tag = '_NO_FUSE'
            return BlockEvaluation.left_right_contract(self._tag_site(ts[1]), ham_left, ham_right, self.mpo_info, tag)
        elif dir == '_KET' or dir == '_BRA':
            ts = sorted(tn.tensors, key=self._tag_site)
            at = 1
            map_idx_b = {}
            for block in ts[1].blocks:
                subg = block.q_labels[0]
                if subg not in map_idx_b:
                    map_idx_b[subg] = []
                map_idx_b[subg].append(block)
            blocks = []
            for block_a in ts[0].blocks:
                subg = block_a.q_labels[at]
                if subg in map_idx_b:
                    outga = block_a.q_labels[0:at]
                    for block_b in map_idx_b[subg]:
                        outg = outga + block_b.q_labels[1:]
                        mat = np.tensordot(block_a.reduced, block_b.reduced, axes=([at], [0]))
                        blocks.append(SubTensor(q_labels=outg, reduced=mat))
            return Tensor(blocks=blocks).set_tags({dir, i, j})
        else:
            assert False


class BlockWavefunction:
    """A wrapper of Wavefunction (block code) for Davidson algorithm."""
    def __init__(self, wave, factor=1.0):
        self.data = wave
        self.factor = factor
    
    def __rmul__(self, factor):
        return BlockWavefunction(self.data, self.factor * factor)
    
    def __imul__(self, factor):
        tensor_scale(self.factor * factor, self.data)
        self.factor = 1.0
        return self
    
    def __iadd__(self, other):
        assert self.factor == 1.0
        tensor_scale_add_no_trans(other.factor, other.data, self.data)
        return self
    
    def copy(self):
        """Return a deep copy of this object."""
        mat = self.data.__class__()
        mat.deep_copy(self.data)
        return BlockWavefunction(mat, self.factor)
    
    def clear_copy(self):
        """Return a deep copy of this object, but all the matrix elements are set to zero."""
        mat = self.data.__class__()
        mat.deep_clear_copy(self.data)
        return BlockWavefunction(mat, 1.0)
    
    def copy_data(self, other):
        """Fill the matrix elements in this object with data
        from another :class:`BlockWavefunction` object."""
        self.data.copy_data(other.data)
        self.factor = other.factor
    
    def dot(self, other):
        """Return dot product of two :class:`BlockWavefunction`."""
        return tensor_dot_product(self.data, other.data) * self.factor * other.factor
    
    def precondition(self, ld, diag):
        """
        Apply precondition on this object.
        
        Args:
            ld : float
                Eigenvalue.
            diag : DiagonalMatrix
                Diagonal elements of Hamiltonian.
        """
        tensor_precondition(self.data, ld, diag)
    
    def normalize(self):
        """Normalization."""
        self.factor = 1.0
        tensor_scale(1 / np.sqrt(self.dot(self)), self.data)
    
    def deallocate(self):
        """Deallocate the memory associated with this object."""
        assert self.data is not None
        self.data.deallocate()
        self.data = None
    
    @property
    def ref(self):
        return self.data.ref
    
    def __repr__(self):
        return repr(self.factor) + " * " + repr(self.data)


class BlockMultiplyH:
    """
    A wrapper of Block.MultiplyH (block code) for Davidson algorithm.
    
    Attributes:
        opt : OperatorTensor
            The (symbolic) super block Hamiltonian.
        st : StateInfo
            StateInfo of super block.
        diag_mat : DiagonalMatrix
            Diagonal elements of super block Hamiltonian, in flatten form with no quantum labels.
    """
    def __init__(self, opt, sts, diag=True):
        self.opt = opt
        self.sts = sts
        if diag:
            self.diag_mat = BlockEvaluation.expr_diagonal_eval(opt.mat[0, 0], opt.ops[0], opt.ops[1], sts)
        else:
            self.diag_mat = None
    
    def diag(self):
        """Returns Diagonal elements (for preconditioning)."""
        return self.diag_mat
    
    def diag_norm(self):
        return np.linalg.norm(self.diag().ref)
    
    def apply(self, other, result):
        """
        Perform :math:`\\hat{H}|\\psi\\rangle`.
        
        Args:
            other : BlockWavefunction
                Input vector/wavefunction.
            result : BlockWavefunction
                Output vector/wavefunction.
        """
        assert isinstance(result, BlockWavefunction)
        result.data.clear()
        result.factor = 1.0
        BlockEvaluation.expr_multiply_eval(self.opt.mat[0, 0], self.opt.ops[0], self.opt.ops[1],
            other.data, result.data, self.sts)
