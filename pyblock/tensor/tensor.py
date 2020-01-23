
import numpy as np
from itertools import accumulate


class SubTensor:
    # -- q_labels is tuple([direct product of irreps])
    # for example, qlabels = [ParticleN(0) * SU2(0) * PGD2H(0),]
    # each element represent one rank of the reduced matrix/tensor
    # if rank == 2, then tensor = matrix
    # -- rank r = len(q_labels)
    # -- reduced is rank r dense tensor (numpy array)
    # -- cgs is [rank r dense tensor for rotating proj q numbers (Sz etc)]
    # -- ng is number of ir symmetry sub-groups
    def __init__(self, q_labels=None, reduced=None, cgs=None):
        self.q_labels = q_labels if q_labels is not None else []
        # assuming that q_labels must be set at the beginning
        self.rank = len(q_labels)
        self.ng = 0 if self.rank == 0 else q_labels[0].ng
        self.reduced = reduced
        self.reduced_shape = [0] * \
            self.rank if reduced is None else reduced.shape
        self.cgs = cgs
        if self.rank != 0:
            if reduced is not None:
                assert len(self.reduced.shape) == self.rank
            assert all(q.ng == self.ng for q in q_labels)
            if self.cgs is not None:
                assert len(self.cgs) == self.ng
                assert all(len(cg.shape) == self.rank for cg in cgs)

    def build_rank3_cg(self):
        assert self.rank == 3
        syms = [ir.__class__ for ir in self.q_labels[0].irs]
        self.cgs = [syms[ig].clebsch_gordan(*[q.irs[ig] for q in self.q_labels])
                    for ig in range(self.ng)]

    def build_random(self):
        self.reduced = np.random.random(self.reduced_shape)

    def build_zero(self):
        self.reduced = np.zeros(self.reduced_shape)

    # tranpose of operator
    # (ket, op, bra) -> (bra, -op, ket)
    @property
    def T(self):
        q_ket, q_op, q_bra = self.q_labels
        return SubTensor(q_labels=(q_bra, -q_op, q_ket),
                         reduced=np.transpose(self.reduced, (2, 1, 0)),
                         cgs=[np.transpose(cg, (2, 1, 0)) for cg in self.cgs])

    def __mul__(self, o):
        return SubTensor(q_labels=self.q_labels, reduced=o * reduced, cgs=self.cgs)

    def __repr__(self):
        return "(Q=) %r (R=) %r" % (self.q_labels, self.reduced)


class Tensor:
    # blocks: list of (non-zero) TensorBlock's
    # tags: list of tags for each rank
    def __init__(self, blocks=None, tags=None, contractor=None):
        self.blocks = blocks if blocks is not None else []
        self.tags = tags if tags is not None else set()
        if not isinstance(self.tags, set):
            self.tags = {self.tags}
        self.contractor = contractor

    def copy(self):
        return Tensor(blocks=self.blocks[:], tags=self.tags.copy(),
                      contractor=self.contractor)

    @property
    def rank(self):
        return 0 if len(self.blocks) == 0 else self.blocks[0].rank

    @property
    def ng(self):
        return 0 if len(self.blocks) == 0 else self.blocks[0].ng

    @property
    def n_blocks(self):
        return len(self.blocks)

    # build the tensor by coupling states in pre and basis into states in post
    @staticmethod
    def rank3_init(pre, basis, post):
        blocks = []
        for kp, vp in sorted(pre.items(), key=lambda x:x[0]):
            for kb, vb in sorted(basis.items(), key=lambda x:x[0]):
                rs = kp + kb
                for kr in (rs if isinstance(rs, list) else [rs]):
                    if kr in sorted(post.keys()):
                        blocks.append(SubTensor(q_labels=(kp, kb, kr)))
                        blocks[-1].reduced_shape = [vp, vb, post[kr]]
        return Tensor(blocks)

    # repr is a list of dense matrices, in different op_q_labels
    # op_q_labels is a list of operator quantum numbers
    @staticmethod
    def operator_init(basis, repr, op_q_labels):
        blocks = []
        for q in range(len(repr.shape)):
            for i in range(repr[q].shape[0]):
                for j in range(repr[q].shape[1]):
                    if not np.isclose(repr[q][i, j], 0.0):
                        q_labels = (basis[j], op_q_labels[q], basis[i])
                        reduced = np.array([[[repr[q][i, j]]]], dtype=float)
                        blocks.append(SubTensor(q_labels, reduced))
        t = Tensor(blocks)
        t.build_rank3_cg()
        return t

    def build_rank3_cg(self):
        for block in self.blocks:
            block.build_rank3_cg()

    def build_random(self):
        for block in self.blocks:
            block.build_random()

    def build_zero(self):
        for block in self.blocks:
            block.build_zero()

    # set the internal reduced matrix to identity
    # not work for general situations
    def build_identity(self):
        assert self.rank == 3
        cur_idx = {}
        for block in self.blocks:
            q_labels_r = block.q_labels[2:]
            if q_labels_r not in cur_idx:
                cur_idx[q_labels_r] = 0
            k = cur_idx[q_labels_r]
            for i in range(block.reduced.shape[0]):
                for j in range(block.reduced.shape[1]):
                    # if MPS is initialized from LineCoupling (both_dir = False)
                    # then k will never bigger than block.reduced.shape[2]
                    # otherwise there is truncation in right labels
                    # then the corresponding extra left rows will be set zero
                    if k < block.reduced.shape[2]:
                        block.reduced[i, j, k] = 1.0
                        k += 1
            cur_idx[q_labels_r] = k

    # the indices in idx_l will be combined
    # the indices in idx_r will also be combined
    # then for each entry if q_label(idx_l) == q_label(idx_r), the term will be included
    @staticmethod
    def partial_trace(ts, idx_l, idx_r, target_q_labels=None):
        out_idx = list(set(range(0, ts.rank)) - set(idx_l) - set(idx_r))

        trace_scr = list(range(0, ts.rank))
        for ia, ib in zip(idx_l, idx_r):
            trace_scr[ib] = trace_scr[ia]

        if target_q_labels is None:
            map_idx_out = {}
            for block in ts.blocks:
                sub_l = tuple(block.q_labels[id] for id in idx_l)
                sub_r = tuple(block.q_labels[id] for id in idx_r)
                if sub_l != sub_r:
                    continue
                outg = tuple(block.q_labels[id] for id in out_idx)
                mat = np.einsum(block.reduced, trace_scr)
                if outg not in map_idx_out:
                    cgs = [np.einsum(cg, trace_scr) for cg in block.cgs]
                    map_idx_out[outg] = SubTensor(
                        q_labels=outg, reduced=mat, cgs=cgs)
                else:
                    map_idx_out[outg].reduced += mat
        else:
            raise TensorNetworkError('not implemented yet!')
        
        return Tensor(tensors=map_idx_out.values(), tags=ts.tags, contractor=ts.contractor)

    # contract two tensor to form a new tensor
    # idxa, idxb are indices to be contracted in tensor a and b, respectively
    # if target_q_labels is a DirectProdGroup, then target only one specific S
    # elif target_q_labels is a list, then target multiple S
    @staticmethod
    def contract(tsa, tsb, idxa, idxb, target_q_labels=None):
        out_idx_a = list(set(range(0, tsa.rank)) - set(idxa))
        out_idx_b = list(set(range(0, tsb.rank)) - set(idxb))

        map_idx_b = {}
        for block in tsb.blocks:
            subg = tuple(block.q_labels[id] for id in idxb)
            if subg not in map_idx_b:
                map_idx_b[subg] = []
            map_idx_b[subg].append(block)

        # if target is None, assuming target is the S=0 state
        # ie, all quantum numbers should be equal in contracted indices
        # this is the abelian or S=0 non-abelian case (MPO-MPS contraction case or MPS-MPS contraction case)
        # this works because MPS can be considered as an identity operator (with the operator index implicitly zero)
        # then any S q-number operator contract with S=0 operator, the operator q-number index will not change
        # then only need to contract other state representation indices.
        # MPS indices contraction auto handled by CGC. Operator contraction need to specify target.
        if target_q_labels is None:
            map_idx_out = {}
            for block_a in tsa.blocks:
                subg = tuple(block_a.q_labels[id] for id in idxa)
                if subg in map_idx_b:
                    outga = tuple(block_a.q_labels[id] for id in out_idx_a)
                    for block_b in map_idx_b[subg]:
                        outg = outga + \
                            tuple(block_b.q_labels[id] for id in out_idx_b)
                        mat = np.tensordot(
                            block_a.reduced, block_b.reduced, axes=(idxa, idxb))
                        if outg not in map_idx_out:
                            cgs = [np.tensordot(cga, cgb, axes=(idxa, idxb))
                                   for cga, cgb in zip(block_a.cgs, block_b.cgs)]
                            map_idx_out[outg] = SubTensor(
                                q_labels=outg, reduced=mat, cgs=cgs)
                        else:
                            map_idx_out[outg].reduced += mat
        # non-abelian case (operator blocking case)
        # can only contract one index at a time
        # a rank-3 operator contracted rank-3 operator, in operator q-number index
        # will generate a rank-5 operator, the additional index is for new operator q-number index, in the middle
        else:
            assert len(idxa) == 1 and len(idxb) == 1
            if not isinstance(target_q_labels, list):
                target_q_labels = [target_q_labels]
            map_idx_out = {}
            for target in target_q_labels:
                for block_a in tsa.blocks:
                    a_rank = tuple(block_a.q_labels[id] for id in idxa)[0]
                    b_ranks = target + (-a_rank)
                    for b_rank in (b_ranks if isinstance(b_ranks, list) else [b_ranks]):
                        if (b_rank, ) in map_idx_b:
                            syms = [ir.__class__ for ir in target.irs]
                            target_cgs = [syms[ig].clebsch_gordan(a_rank.irs[ig], b_rank.irs[ig], target.irs[ig])
                                          for ig in range(target.ng)]
                            a_out_ranks = tuple(
                                block_a.q_labels[id] for id in out_idx_a)
                            for block_b in map_idx_b[(b_rank, )]:
                                outg = a_out_ranks + \
                                    (target, ) + \
                                    tuple(block_b.q_labels[id]
                                          for id in out_idx_b)
                                mat = np.tensordot(
                                    block_a.reduced, block_b.reduced, axes=(idxa, idxb))
                                mat = mat.reshape(
                                    mat.shape[:len(a_out_ranks)] + (1, ) + mat.shape[len(a_out_ranks):])
                                if outg not in map_idx_out:
                                    cgs = [np.tensordot(cga, np.tensordot(cgt, cgb, axes=([1], idxb)), axes=(idxa, [0]))
                                           for cga, cgb, cgt in zip(block_a.cgs, block_b.cgs, target_cgs)]
                                    map_idx_out[outg] = SubTensor(
                                        q_labels=outg, reduced=mat, cgs=cgs)
                                else:
                                    map_idx_out[outg].reduced += mat
        return Tensor(map_idx_out.values())

    # left normalization needs to collect all left indices for each specific right index
    # so that we will only have one R, but left dim of q is unchanged
    # at: where to divide the tensor into matrix => (0, at) x (at, n_ranks)
    def left_normalize(self):
        at = self.rank - 1
        collected_rows = {}
        for block in self.blocks:
            q_labels_r = tuple(block.q_labels[id]
                               for id in range(at, self.rank))
            if q_labels_r not in collected_rows:
                collected_rows[q_labels_r] = []
            collected_rows[q_labels_r].append(block)
        r_blocks = {}
        for q_labels_r, blocks in collected_rows.items():
            l_shapes = [np.prod([b.reduced.shape[id]
                                 for id in range(at)]) for b in blocks]
            mat = np.concatenate([b.reduced.reshape((sh, -1))
                                  for sh, b in zip(l_shapes, blocks)], axis=0)
            q, r = np.linalg.qr(mat)
            r_blocks[q_labels_r] = r
            qs = np.split(q, list(accumulate(l_shapes[:-1])), axis=0)
            assert(len(qs) == len(blocks))
            for q, b in zip(qs, blocks):
                b.reduced = q.reshape(b.reduced.shape[:at] + (r.shape[0], ))
                b.reduced_shape = b.reduced.shape
        return r_blocks

    # mats: dict {partial q_labels: matrix}
    # currently only used for multiply r from left-normalization
    def left_multiply(self, mats):
        for block in self.blocks:
            q_labels_r = (block.q_labels[0], )
            if q_labels_r in mats:
                block.reduced = np.tensordot(
                    mats[q_labels_r], block.reduced, axes=([1], [0]))
                block.reduced_shape = block.reduced.shape

    def set_tags(self, tags):
        self.tags = tags
        return self

    def set_contractor(self, contractor):
        self.contractor = contractor
        return self

    def __add__(self, o):
        assert self.rank == o.rank and self.ng == o.ng
        map_s = {tuple(b.q_labels): b for b in self.blocks}
        map_o = {tuple(b.q_labels): b for b in o.blocks}
        blocks = []
        for q, b in map_s:
            if q in map_o:
                blocks.append(
                    SubTensor(q, b.reduced + map_o[q].reduced, b.cgs))
            else:
                blocks.append(b)
        return Tensor(blocks=blocks, tags=self.tags, contractor=self.contractor)
    
    def __mul__(self, o):
        return Tensor(blocks=[b * o for b in self.blocks], tags=self.tags.copy(),
                      contractor=self.contractor)

    def __repr__(self):
        return "\n".join(
            ("%3d " % ib) + b.__repr__() for ib, b in enumerate(self.blocks))


class TensorNetworkError(Exception):
    pass


# an inefficient implementation for Quimb TensorNetwork
class TensorNetwork:
    def __init__(self, tensors=None):
        self.tensors = list(tensors) if tensors is not None else []

    def select(self, tags, which='all', inverse=False):
        r = []
        if not isinstance(tags, set):
            tags = {tags}
        for tensor in self.tensors:
            if which == 'all':
                p = tags.issubset(tensor.tags)
            elif which == 'any':
                p = len(tags.intersection(tensor.tags)) != 0
            elif which == 'exact':
                p = tags == tensor.tags
            else:
                raise TensorNetworkError('invalid which parameter.')
            if inverse ^ p:
                r.append(tensor)
        return self.__class__(tensors=r)

    def remove(self, tags, which='all', in_place=False):
        if not in_place:
            return self.select(tags, which, inverse=True)
        else:
            self.tensors = self.select(tags, which, inverse=True).tensors
            return self

    def add(self, tn):
        if isinstance(tn, Tensor):
            self.tensors.append(tn)
        elif isinstance(tn, TensorNetwork):
            for tensor in tn.tensors:
                self.tensors.append(tensor)
        else:
            raise TensorNetworkError(
                'Unable to add this object to the network.')

    def add_tags(self, tags):
        for tensor in self.tensors:
            tensor.tags |= tags

    def copy(self):
        return self.__class__(tensors=[t.copy() for t in self.tensors])

    def remove_tags(self, tags):
        for tensor in self.tensors:
            tensor.tags -= tags

    def contract(self, tags, in_place=False):
        cont_tn = self.select(set(tags), which='any')
        ctr = None
        for t in cont_tn.tensors:
            if t.contractor is not None:
                ctr = t.contractor
                break
        if ctr is None:
            raise TensorNetworkError('Regular contraction not implemented.')
        else:
            cont_ts = ctr.contract(cont_tn, tags)
            if in_place:
                self.remove(set(tags), which='any', in_place=True)
                self |= cont_ts
                return self
            else:
                return self.remove(set(tags), which='any') | cont_ts

    def __xor__(self, tags):
        return self.contract(tags)

    def __ixor__(self, tags):
        return self.contract(tags, in_place=True)

    def __ior__(self, tensors):
        self.add(tensors)
        return self

    def __len__(self):
        return len(self.tensors)

    # this may be different from quimb impl
    def __getitem__(self, tags):
        t = self.select(tags, which='exact')
        if len(t) == 0:
            raise TensorNetworkError(
                'Unable to find an item with the given tags.')
        return t if len(t) > 1 else t.tensors[0]

    def __or__(self, other):
        if isinstance(other, Tensor):
            return self.__class__(tensors=self.tensors + [other])
        elif isinstance(other, TensorNetwork):
            if self.__class__ == other.__class__:
                return self.__class__(tensors=self.tensors + other.tensors)
            else:
                return TensorNetwork(tensors=self.tensors + other.tensors)
        else:
            raise TensorNetworkError(
                'Unable to create the network using this object.')

    @property
    def tags(self):
        return [ts.tags for ts in self.tensors]
