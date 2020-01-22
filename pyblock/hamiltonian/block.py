
from block import VectorInt, VectorMatrix, Matrix
from block import save_rotation_matrix
from block.io import Global, read_input, Input, AlgorithmTypes
from block.io import init_stack_memory, release_stack_memory, AlgorithmTypes
from block.dmrg import MPS_init, MPS, get_dot_with_sys
from block.symmetry import tensor_product, SpinQuantum, VectorSpinQuantum
from block.symmetry import StateInfo, SpinSpace, IrrepSpace, tensor_product_target
from block.operator import Wavefunction, OpTypes
from block.block import Block, StorageTypes
from block.block import init_starting_block, init_new_system_block

from ..symmetry.symmetry import ParticleN, SU2, PointGroup, point_group
from ..tensor.tensor import Tensor
from .qc import read_fcidump
from fractions import Fraction
import numpy as np


class BlockError(Exception):
    pass


class BlockSymmetry:
    # translate a DirectProdGroup object to SpinQuantum (block code)
    @classmethod
    def to_spin_quantum(self, dpg):
        subg = [ParticleN, SU2, PointGroup]
        if dpg.ng != 3 or not all(isinstance(ir, c) for c, ir in zip(subg, dpg.irs)):
            raise BlockError('Representation not supported by block code.')
        return SpinQuantum(dpg.irs[0].ir, SpinSpace(dpg.irs[1].ir), IrrepSpace(dpg.irs[2].ir))

    # translate SpinQuantum (block code) to a DirectProdGroup object
    @classmethod
    def from_spin_quantum(self, sq):
        PG = point_group(Global.point_group)
        return ParticleN(sq.n) * SU2(sq.s.irrep) * PG(sq.symm.irrep)

    # translate a [(DirectProdGroup, int)] to SpinQuantum (block code)
    @classmethod
    def to_state_info(self, states):
        qs = VectorSpinQuantum()
        ns = VectorInt()
        for k, v in states:
            qs.append(self.to_spin_quantum(k))
            ns.append(v)
        return StateInfo(qs, ns)

    @classmethod
    def initial_state_info(self, i=0):
        return MPS.site_blocks[i].ket_state_info

    # Translate the last one or two mps tensors to Wavefunction (block code)
    # in two_dot scheme, wavefunction is repr'd in 2M(n-3 site) x M(n-1 site)
    # in one_dot scheme, wavefunction is repr'd in  M(n-2 site) x M(n-1 site)
    # in two_dot, left_state_info is gen'd from :func:`to_rotation_matrix`
    # with `i = n - 3`
    # in one_dot, left_state_info is gen'd with `i = n - 2`
    @classmethod
    def to_wavefunction(self, one_dot, left_state_info, n_sites, mps, target):

        # one dot case
        if one_dot:

            map_last = {
                block.q_labels[:2]: block for block in mps[n_sites - 1].blocks}

            l = left_state_info
            r = MPS.site_blocks[n_sites - 1].ket_state_info
            big = tensor_product_target(l, r)

            target_state_info = self.to_state_info([(target, 1)])
            wfn = Wavefunction()
            wfn.initialize(target_state_info.quanta, l, r, True)
            wfn.onedot = True

            for (ib, ik), mat in wfn.non_zero_blocks:
                sqs = [l.quanta[ib], r.quanta[ik]]
                q_labels = tuple(self.from_spin_quantum(sq) for sq in sqs)
                if q_labels in map_last:
                    mat.ref[:, :] = map_last[q_labels].reduced.reshape(
                        mat.ref.shape)
                else:
                    mat.ref[:, :] = 0

            big.collect_quanta()
            return wfn, big

        # two dot case
        else:

            last_two = Tensor.contract(
                mps[n_sites - 2], mps[n_sites - 1], [2], [0])
            map_last = {block.q_labels[:3]: block for block in last_two.blocks}

            l = left_state_info
            r = MPS.site_blocks[n_sites - 2].ket_state_info
            rr = MPS.site_blocks[n_sites - 1].ket_state_info
            ll = tensor_product(l, r)
            big = tensor_product_target(ll, rr)
            ll_idl = ll.left_unmap_quanta
            ll_idr = ll.right_unmap_quanta

            ll_collected = ll.copy()
            ll_collected.collect_quanta()
            otn = ll_collected.old_to_new_state

            qq = self.to_spin_quantum(target)

            target_state_info = self.to_state_info([(target, 1)])

            wfn = Wavefunction()
            wfn.initialize(target_state_info.quanta, ll_collected, rr, False)
            wfn.onedot = False
            wfn.clear()

            for (ibc, ik), mat in wfn.non_zero_blocks:
                mats = []
                for ib in otn[ibc]:
                    sqs = [l.quanta[ll_idl[ib]],
                           r.quanta[ll_idr[ib]], rr.quanta[ik]]
                    q_labels = tuple(self.from_spin_quantum(sq) for sq in sqs)
                    if q_labels in map_last:
                        rd_shape = map_last[q_labels].reduced.shape
                        shape = (rd_shape[0] * rd_shape[1], rd_shape[2])
                        mats.append(map_last[q_labels].reduced.reshape(shape))
                if mats != []:
                    all_mat = np.concatenate(mats, axis=0)
                    assert(all_mat.shape == mat.ref.shape)
                    mat.ref[:, :] = all_mat
                else:
                    mat.ref[:, :] = 0

            big.collect_quanta()
            big.left_state_info = ll_collected
            big.right_state_info.left_state_info = rr
            # TODO: this should be empty state
            big.right_state_info.right_state_info = self.to_state_info([
                                                                       (target, 1)])
            return wfn, big, (target_state_info, ll_collected)

    # Translate a site in MPS to rotation matrix (block code) for that site
    # left_state_info: state_info to the left of site i
    # i: site index [left_state_info] \otimes [site i]
    # tensor: mps tensor at site i
    # the first two sites only contributes one rotation matrix
    # this is handled by i = 1 and tensor0 = mps[0]
    @classmethod
    def to_rotation_matrix(self, left_state_info, tensor, i, tensor0=None):
        l = left_state_info
        r = MPS.site_blocks[i].ket_state_info
        lr = tensor_product(l, r)
        lr_idl = lr.left_unmap_quanta
        lr_idr = lr.right_unmap_quanta

        lr_collected = lr.copy()
        lr_collected.collect_quanta()

        map_tensor = {block.q_labels: block for block in tensor.blocks}
        if tensor0 is not None:
            map_tensor0 = {block.q_labels[2]: block for block in tensor0.blocks}

        # if there are repeated q in lr.quanta,
        # current rot Matrix should be None and
        # the reduced matrix should be concatenated to that of the first unique q
        rot = []
        collected = {}
        for i, q in enumerate(lr.quanta):
            sqs = [l.quanta[lr_idl[i]], r.quanta[lr_idr[i]], q]
            q_labels = tuple(self.from_spin_quantum(sq) for sq in sqs)
            if q_labels in map_tensor:
                block = map_tensor[q_labels]
                reduced = block.reduced.reshape(block.reduced.shape[0::2])
                if tensor0 is not None:
                    block0 = map_tensor0[q_labels[0]]
                    reduced = block0.reduced.reshape(
                        block0.reduced.shape[0::2]) @ reduced
                if q_labels[2] not in collected:
                    rot.append([reduced])
                    collected[q_labels[2]] = len(rot) - 1
                else:
                    rot[collected[q_labels[2]]].append(reduced)
                    rot.append(())
            else:
                if q_labels[2] not in collected:
                    rot.append(None)
                    collected[q_labels[2]] = len(rot) - 1
                else:
                    rot.append(())

        rot_uncollected = VectorMatrix(Matrix() if r is None or r is () else Matrix(
            np.concatenate(r, axis=0)) for r in rot)

        # order the quanta according to the order in lr_collected
        rot_collected = []
        for q in lr_collected.quanta:
            m = rot[collected[self.from_spin_quantum(q)]]
            rot_collected.append(
                Matrix() if m is None else Matrix(np.concatenate(m, axis=0)))

        rot_collected = VectorMatrix(rot_collected)

        lr_truncated = StateInfo()
        StateInfo.transform_state(rot_uncollected, lr, lr_truncated)
        lr_truncated.left_state_info = lr_collected.left_state_info
        lr_truncated.right_state_info = lr_collected.right_state_info

        return rot_collected, lr_truncated


class BlockHamiltonian:
    memory_initialzed = False

    def __init__(self, *args, **kwargs):

        Global.dmrginp.output_level = -1
        self.output_level = -1

        if 'output_level' in kwargs:
            Global.dmrginp.output_level = kwargs['output_level']
            self.output_level = kwargs['output_level']
            del kwargs['output_level']

        file_input = False

        if len(args) == 1 and isinstance(args[0], str):
            read_input(args[0])
            self.output_level = Global.dmrginp.output_level
            file_input = True
            if len(kwargs) != 0:
                raise BlockError(
                    'Cannot accept additional arguments if initialized by input file name.')
        elif len(args) != 0:
            raise BlockError(
                'Unknown argument for initialize Block Hamiltonian.')

        if not file_input:
            input = {'noreorder': '', 'maxM': '500', 'maxiter': '30', 'sym': 'c1',
                     'hf_occ': 'integral', 'schedule': 'default'}

            if 'orbitals' in kwargs or 'fcidump' in kwargs:
                fd_name = kwargs['orbitals' if 'orbitals' in kwargs else 'fcidump']
                opts, _ = read_fcidump(fd_name)

                input['nelec'] = opts['nelec']
                input['spin'] = opts['ms2']
                input['irrep'] = opts['isym']

            for k, v in kwargs.items():
                if k in ['orbitals', 'fcidump']:
                    input['orbitals'] = v
                elif k in ['sym', 'point_group']:
                    input['sym'] = v
                elif k == 'nelec':
                    input['nelec'] = str(v)
                elif k == 'spin':
                    input['spin'] = str(v)
                elif k == 'dot':
                    if v == 2:
                        input['twodot'] = ''
                    elif v == 1:
                        input['onedot'] = ''
                elif k == 'irrep':
                    input['irrep'] = str(v)
                elif k == 'output_level':
                    input['outputlevel'] = str(v)
                elif k == 'hf_occ':
                    input['hf_occ'] = str(v)
                elif k == 'max_iter':
                    input['maxiter'] = str(v)
                elif k == 'max_m':
                    input['maxM'] = str(v)
                elif k == 'spin_adapted':
                    if v == False:
                        input['nonspinadapted'] = ''

            Global.dmrginp = Input.read_input_contents(
                '\n'.join([k + ' ' + v for k, v in input.items()]))

            read_input("")

        Global.dmrginp.output_level = self.output_level

        self.point_group = Global.point_group
        self.n_sites = Global.dmrginp.slater_size // 2
        self.n_electrons = Global.dmrginp.n_electrons
        self.target_s = Fraction(Global.dmrginp.molecule_quantum.s.irrep, 2)
        self.spatial_syms = Global.dmrginp.spin_orbs_symmetry[::2]
        self.target_spatial_sym = Global.dmrginp.molecule_quantum.symm.irrep
        if Global.dmrginp.algorithm_type == AlgorithmTypes.TwoDotToOneDot:
            raise BlockError('Currently two dot to one dot is not supported.')
        self.dot = 1 if Global.dmrginp.algorithm_type == AlgorithmTypes.OneDot else 2

        if not self.__class__.memory_initialzed:
            init_stack_memory()
            self.__class__.memory_initialzed = True

        MPS_init(True)

    def make_starting_block(self, forward):
        system = Block()
        init_starting_block(system, forward, -1, -1, 1, 1, 0, False, False, 0)
        system.store(forward, system.sites, -1, -1)
        return system

    def make_big_block(self, system):

        forward = system.sites[0] == 0

        dot_with_sys = get_dot_with_sys(system, self.dot == 1, forward)

        if forward:
            sys_dot_site = system.sites[-1] + 1
            env_dot_site = sys_dot_end + 1
        else:
            sys_dot_site = system.sites[0] - 1
            env_dot_site = sys_dot_end - 1
        
        system_dot = Block(sys_dot_site, sys_dot_site, 0, True)
        environment_dot = Block(env_dot_site, env_dot_site, 0, True)

        sys_have_norm_ops = dot_with_sys
        sys_have_comp_ops = not dot_with_sys

        env_have_norm_ops = not sys_have_norm_ops
        env_have_comp_ops = not sys_have_comp_ops

        if sys_have_comp_ops and OpTypes.CreDesComp not in system.ops:
            system.add_all_comp_ops()

        system.add_additional_ops()

        new_system = Block()

        if not self.dot == 1 or dot_with_sys:
            init_new_system_block(system, system_dot, new_system, -1, -1, 1, True, 0,
                StorageTypes.DistributedStorage, sys_have_norm_ops, sys_have_comp_ops)
        
        environment = Block()
        new_environment = Block()

        init_new_environment_block(
            environment,
            system_dot if self.dot == 1 and not dot_with_sys else environment_dot,
            new_environment, system, system_dot,
            -1, -1, 1, 1, forward, True, self.dot == 1, False, 0,
            env_have_norm_ops, env_have_comp_ops, dot_with_sys)
        
        new_system.loop_block = dot_with_sys
        system.loop_block = dot_with_sys
        new_environment.loop_block = not dot_with_sys
        environment.loop_block = not dot_with_sys

        if self.dot == 1 and not dot_with_sys:
            left_block = system
            right_block = new_environment
        else:
            left_block = new_system
            right_block = new_environment
        
        big = Block()

        init_big_block(left_block, right_block, big)

        return system, system_dot, new_system, environment, big

    def block_rotation(self, new_system, system, rot_mat):
        save_rotation_matrix(new_system.sites, rot_mat, 0)
        save_rotation_matrix(new_system.sites, rot_mat, -1)

        new_system.transform_operators(rot_mat)
        new_system.move_and_free_memory(system)

        return new_system

    def enlarge_block(self, forward, system, rot_mat):
        dot_with_sys = get_dot_with_sys(system, self.dot == 1, forward)
        if forward:
            dot_site = system.sites[-1] + 1
        else:
            dot_site = system.sites[0] - 1
        print(forward, system.sites, dot_site)
        system_dot = Block(dot_site, dot_site, 0, True)

        do_norms = dot_with_sys
        do_comp = not dot_with_sys
        if do_comp and OpTypes.CreDesComp not in system.ops:
            system.add_all_comp_ops()
        system.add_additional_ops()

        new_system = Block()
        init_new_system_block(system, system_dot, new_system, -1, -1, 1, True,
                              0, StorageTypes.DistributedStorage, do_norms, do_comp)

        return self.block_rotation(new_system, system, rot_mat)


if __name__ == '__main__':
    # BlockHamiltonian.read_fcidump('N2.STO3G.FCIDUMP')
    pass
