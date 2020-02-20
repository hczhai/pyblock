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
Python wrapper for high-level DMRG algorithm in block code.
"""

import block
from block import VectorInt, VectorDouble, VectorBool, VectorMatrix, \
    load_rotation_matrix, save_rotation_matrix, DiagonalMatrix, \
    save_rotation_matrix
from block.io import Global, AlgorithmTypes, Timer, read_input, \
    init_stack_memory, release_stack_memory
from block.dmrg import SweepParams, block_and_decimate, do_one, dmrg, calldmrg, \
    get_dot_with_sys, guess_wavefunction, MPS, MPS_init
from block.block import Block, GuessWaveTypes, StorageTypes, \
    init_starting_block, init_new_system_block, init_new_environment_block, \
    init_big_block
from block.operator import OperatorCre, OpTypes, Wavefunction, VectorWavefunction, \
    DensityMatrix, multiply_with_own_transpose

print(block.__doc__)

class DMRG(object):
    """
    Block DMRG in its original workflow.
    """

    sweep_params = None
    system = None
    integral_index = 0
    output_level = 0

    def __init__(self, input_file, output_level=0):
        if input_file != '':
            Global.dmrginp.output_level = output_level
            read_input(input_file)
            Global.dmrginp.output_level = output_level
            init_stack_memory()

        self.sweep_params = SweepParams()
        self.sweep_params.current_root = -1
        self.output_level = output_level

    def finalize(self):
        """Release stack memory."""
        release_stack_memory()

    def dmrg(self, gen_block=False, rot_mats=None):
        """Perform DMRG."""

        global_timer = Timer()

        # 0 = backward 1 = forward
        last_energies = [1E7] * 2
        old_energies = [0.0] * 2

        sweep_tol = Global.dmrginp.sweep_tol

        # warm up sweep

        if not gen_block:
            last_energies[1] = self.do_one(warm_up=True, forward=True)
        else:
            self.gen_block_do_one(rot_mats=rot_mats)

        # --- Start Sweep Loop ---

        while (abs(last_energies[1] - old_energies[1]) > sweep_tol) \
            or (abs(last_energies[0] - old_energies[0]) > sweep_tol) \
            or (Global.dmrginp.algorithm_type == AlgorithmTypes.TwoDotToOneDot
                and Global.dmrginp.twodot_to_onedot_iter + 1
                >= self.sweep_params.sweep_iter):

            # backward and then forward

            for idir in range(0, 2):

                old_energies[idir] = last_energies[idir]

                if Global.dmrginp.n_max_iters <= self.sweep_params.sweep_iter:
                    break

                last_energies[idir] = self.do_one(
                    warm_up=False, forward=idir == 1)

                if self.output_level >= 0:
                    print("\t\t\t Finished Sweep Iteration %d"
                          % self.sweep_params.sweep_iter)

        # --- End Sweep Loop ---

        cputime = global_timer.elapsed_cputime()
        walltime = global_timer.elapsed_walltime()

        if self.output_level >= 0:
            print("\n\n\t\t\t BLOCK CPU  Time (seconds): %.3f" % cputime)
            print("\t\t\t BLOCK Wall Time (seconds): %.3f" % walltime)

    def gen_block_do_one(self, rot_mats=None, forward=True, implicit_trans=True, do_norms=None, do_comp=None):
        """Perform one sweep for generating blocks from rotation matrices."""

        self.system = Block()
        cur_root = self.sweep_params.current_root
        f_size = self.sweep_params.forward_starting_size
        b_size = self.sweep_params.backward_starting_size

        init_starting_block(
            self.system, forward, cur_root, cur_root, f_size, b_size, 0, False,
            False, self.integral_index)

        dot_with_sys = True
        self.system.store(forward, self.system.sites, cur_root, cur_root)

        self.sweep_params.block_iter = 0

        # --- Start Blocking Loop ---

        while self.sweep_params.block_iter < self.sweep_params.n_block_iters:
            print("\nBlock Iteration :: %5d %10s ============="
                  % (self.sweep_params.block_iter,
                      "Forwards" if forward else "Backwards"))

            if self.sweep_params.block_iter == 0:
                self.sweep_params.guess_type = GuessWaveTypes.Basic
            else:
                self.sweep_params.guess_type = GuessWaveTypes.Transform
            
            self.system = self.gen_block_block_and_decimate(
                dot_with_sys, rot_mats=rot_mats, forward=forward,
                implicit_trans=implicit_trans, do_norms=do_norms, do_comp=do_comp)

            self.system.store(forward, self.system.sites, cur_root, cur_root)
            
            if do_norms:
                dot_with_sys = True
            else:
                dot_with_sys = get_dot_with_sys(
                    self.system, self.sweep_params.one_dot, forward)

            self.sweep_params.block_iter += 1

        self.system.deallocate()
        self.system.clear()

        self.sweep_params.sweep_iter += 1

    def do_one(self, warm_up, forward, restart=False, restart_size=0):
        """Perform one sweep."""
        sweep_timer = Timer()

        self.system = Block()
        self.sweep_params.set_sweep_parameters()

        n_roots = Global.dmrginp.n_roots(self.sweep_params.sweep_iter)
        cur_root = self.sweep_params.current_root
        f_size = self.sweep_params.forward_starting_size
        b_size = self.sweep_params.backward_starting_size

        final_energies = VectorDouble([1E10] * n_roots)
        final_error = 0.0

        init_starting_block(
            self.system, forward, cur_root, cur_root, f_size, b_size, restart_size, restart,
            warm_up, self.integral_index)

        dot_with_sys = True

        if not restart:
            self.sweep_params.block_iter = 0

        self.system.store(forward, self.system.sites, cur_root, cur_root)
        self.sweep_params.save_state(forward, len(self.system.sites))

        # --- Start Blocking Loop ---

        while self.sweep_params.block_iter < self.sweep_params.n_block_iters:
            print("\nBlock Iteration :: %5d %10s ============="
                  % (self.sweep_params.block_iter,
                      "Forwards" if forward else "Backwards"))

            if self.sweep_params.block_iter == 0 or warm_up:
                self.sweep_params.guess_type = GuessWaveTypes.Basic
            else:
                self.sweep_params.guess_type = GuessWaveTypes.Transform

            # new_system = Block()
            # block_and_decimate(self.sweep_params, self.system,
            #                    new_system, warm_up, dot_with_sys)

            new_system = self.block_and_decimate(warm_up, dot_with_sys)

            for j in range(0, n_roots):
                if self.output_level >= 0:
                    print("\t\t\t Total block energy for state [ %d ] with %d states = %20.10f" %
                          (j, self.sweep_params.n_keep_states, self.sweep_params.lowest_energy[j]))

            if sum(self.sweep_params.lowest_energy) < sum(final_energies):
                final_energies = self.sweep_params.lowest_energy

            final_error = max(self.sweep_params.lowest_error, final_error)

            self.system = new_system

            # this will only print operators when output_level >= 2
            self.system.print_operator_summary()

            self.system.store(forward, self.system.sites, cur_root, cur_root)

            if self.output_level >= 0:
                print(self.system)

            dot_with_sys = get_dot_with_sys(
                self.system, self.sweep_params.one_dot, forward)

            self.sweep_params.block_iter += 1

            self.sweep_params.save_state(forward, len(self.system.sites))

        # --- End Blocking Loop ---

        self.system.deallocate()
        self.system.clear()

        if self.output_level >= 0:
            for j in range(0, n_roots):
                print(("\n\t\t\t Finished Sweep with %d states and sweep energy for " +
                       "State [ %d ] with Spin [ %s ] = %20.10f")
                      % (self.sweep_params.n_keep_states, j, str(Global.dmrginp.molecule_quantum.s),
                          final_energies[j]))

            print("\n\t\t\t Largest Error for Sweep with %s states is %20.10f."
                  % (self.sweep_params.n_keep_states, final_error))

        self.sweep_params.largest_dw = final_error

        self.sweep_params.sweep_iter += 1

        cputime = sweep_timer.elapsed_cputime()
        walltime = sweep_timer.elapsed_walltime()

        if self.output_level >= 0:
            print("\t\t\t Elapsed Sweep CPU  Time (seconds): %f" % cputime)
            print("\t\t\t Elapsed Sweep Wall Time (seconds): %f" % walltime)

        return sum(final_energies) / len(final_energies)

    def block_and_decimate(self, warm_up, dot_with_sys):
        """Block and renormalize operators in Block."""

        if self.output_level >= 2:
            print("\t\t\t dot with system %r" % dot_with_sys)

        if self.output_level >= 1:
            print("\t\t\t Performing Blocking")

        Global.dmrginp.timer_guessgen.start()

        forward = self.system.sites[0] == 0
        su2_used = Global.dmrginp.is_spin_adapted

        # figure out the range of dot blocks and environment block - start

        if forward:
            sys_dot_start = self.system.sites[-1] + \
                1 if su2_used else self.system.sites[-1] // 2 + 1
            sys_dot_end = sys_dot_start + self.sweep_params.sys_add - \
                1  # end means the index of last site
            env_dot_start = sys_dot_end + 1
            env_dot_end = env_dot_start + self.sweep_params.env_add - 1
        else:
            sys_dot_start = self.system.sites[0] - \
                1 if su2_used else self.system.sites[0] // 2 + 1
            sys_dot_end = sys_dot_start - (self.sweep_params.sys_add - 1)
            env_dot_start = sys_dot_end - 1
            env_dot_end = env_dot_start - (self.sweep_params.env_add - 1)

        system_dot = Block(sys_dot_start, sys_dot_end,
                           self.integral_index, True)
        environment_dot = Block(
            env_dot_start, env_dot_end, self.integral_index, True)

        # make System Environment Big Blocks

        sys_have_norm_ops = dot_with_sys
        sys_have_comp_ops = not dot_with_sys

        env_have_norm_ops = not sys_have_norm_ops
        env_have_comp_ops = not sys_have_comp_ops

        if sys_have_comp_ops and OpTypes.CreDesComp not in self.system.ops:
            self.system.add_all_comp_ops()

        self.system.add_additional_ops()

        cur_root = self.sweep_params.current_root

        # case 1: one_dot and dot_with_sys : only big system
        # case 2: one_dot and not dot_with_sys : only big env
        # case 3: not one_dot : big system and big env

        new_system = Block()

        if not self.sweep_params.one_dot or dot_with_sys:
            init_new_system_block(
                self.system, system_dot, new_system, cur_root, cur_root,
                self.sweep_params.sys_add, True, self.integral_index,
                StorageTypes.DistributedStorage, sys_have_norm_ops, sys_have_comp_ops)

        environment = Block()
        new_environment = Block()
        big = Block()

        init_new_environment_block(
            environment,
            system_dot if self.sweep_params.one_dot and not dot_with_sys else environment_dot,
            new_environment, self.system, system_dot,
            cur_root, cur_root, self.sweep_params.sys_add, self.sweep_params.env_add,
            forward, True, self.sweep_params.one_dot, warm_up, self.integral_index,
            env_have_norm_ops, env_have_comp_ops, dot_with_sys)

        new_system.loop_block = dot_with_sys
        self.system.loop_block = dot_with_sys
        new_environment.loop_block = not dot_with_sys
        environment.loop_block = not dot_with_sys

        if self.sweep_params.one_dot and not dot_with_sys:
            left_block = self.system
            right_block = new_environment
        else:
            left_block = new_system
            right_block = new_environment
        
        init_big_block(left_block, right_block, big)

        if self.output_level >= 2:
            print("\t\t\t System  Block")
            left_block.print_operator_summary()
            print("\t\t\t Environment Block", str(right_block))
            right_block.print_operator_summary()

        Global.dmrginp.timer_guessgen.stop()
        Global.dmrginp.timer_multiplier.start()
        
        if self.output_level >= 1:
            print("\t\t\t Solving wavefunction")

        lower_states = VectorWavefunction()

        if cur_root >= 0:
            for i in range(cur_root):
                state = Wavefunction()
                state.initialize(
                    Global.dmrginp.effective_molecule_quantum_vec(),
                    left_block.ket_state_info, right_block.ket_state_info,
                    self.sweep_params.one_dot)
                lower_states.append(state)
        
            self.output_level = Global.dmrginp.output_level
            Global.dmrginp.output_level = -1

            e = DiagonalMatrix()

            if self.sweep_params.block_iter == 0:
                guess_type = GuessWaveTypes.Transpose
            else:
                guess_type = GuessWaveTypes.Transform
        
            for i in range(cur_root):

                overlap_big = Block()
                overlap_system = Block()
                overlap_environment = Block()
                overlap_new_system = Block()
                overlap_new_environment = Block()

                make_system_environment_big_overlap_blocks(
                    self.system.sites(), system_dot, environment_dot, overlap_system,
                    overlap_new_system, overlap_environment, overlap_new_environment,
                    overlap_big, self.sweep_params, dot_with_sys, warm_up,
                    self.integral_index, cur_root, i)
                
                lower_states[i].clear()

                temp = Wavefunction()
                temp.initialize(
                    Global.dmrginp.effective_molecule_quantum_vec(),
                    overlap_new_system.ket_state_info,
                    overlap_new_environment.ket_state_info, True)
                temp.clear()

                guess_wavefunction(
                    temp, e, overlap_big, guess_type, self.sweep_params.one_dot,
                    i, dot_with_sys, 0.0)
                
                overlap_big.multiply_overlap(temp, lower_states[i])

                overlap_new_environment.deallocate()
                overlap_new_system.deallocate()
                overlap_environment.deallocate()
                overlap_system.deallocate()
            
            Global.dmrginp.output_level = self.output_level
        
        noise = self.sweep_params.noise
        additional_noise = self.sweep_params.additional_noise

        rotate_matrix = VectorMatrix()

        self.sweep_params.lowest_error = new_system.renormalize_from(
            self.sweep_params.lowest_energy, self.sweep_params.lowest_energy_spin,
            self.sweep_params.lowest_error, rotate_matrix,
            self.sweep_params.n_keep_states, self.sweep_params.n_keep_qstates,
            self.sweep_params.davidson_tol, big, self.sweep_params.guess_type,
            noise, additional_noise, self.sweep_params.one_dot, self.system, system_dot,
            environment, dot_with_sys, warm_up, self.sweep_params.sweep_iter,
            self.sweep_params.current_root, lower_states)
        
        if self.sweep_params.current_root >= 0:
            for i in range(self.sweep_params.current_root - 1, -1, -1):
                lower_states[i].deallocate()
        
        new_environment.remove_additional_ops()
        new_environment.deallocate()

        environment.remove_additional_ops()
        environment.deallocate()

        if self.output_level >= 1:
            print("\t\t\t Performing Renormalization")
        
        if self.output_level >= 0:
            print("\n\t\t\t Total discarded weight: %f" % self.sweep_params.lowest_error)
        
        Global.dmrginp.timer_multiplier.stop()
        Global.dmrginp.timer_operrot.start()

        new_system.transform_operators(rotate_matrix)

        # here implies system.clear()
        new_system.move_and_free_memory(self.system)

        if self.sweep_params.current_root >= 0:

            self.output_level = Global.dmrginp.output_level
            Global.dmrginp.output_level = -1

            e = DiagonalMatrix()

            if self.sweep_params.block_iter == 0:
                guess_type = GuessWaveTypes.Transpose
            else:
                guess_type = GuessWaveTypes.Transform
        
            for i in range(self.sweep_params.current_root):

                overlap_big = Block()
                overlap_system = Block()
                overlap_environment = Block()
                overlap_new_system = Block()
                overlap_new_environment = Block()

                overlap_system_dot = Block(sys_dot_start, sys_dot_end, self.integral_index, True)
                overlap_environment_dot = Block(env_dot_start, env_dot_end, self.integral_index, True)

                make_system_environment_big_overlap_blocks(
                    self.system.sites, overlap_system_dot, overlap_environment_dot,
                    overlap_system, overlap_new_system,
                    overlap_environment , overlap_new_environment,
                    overlap_big, self.sweep_params, True, warm_up,
                    self.integral_index, self.sweep_params.current_root, i)

                iwave = Wavefunction()
                iwave.initialize(
                    Global.dmrginp.effective_molecule_quantum_vec(),
                    overlap_new_system.ket_state_info,
                    overlap_new_environment.ket_state_info, True)
                iwave.clear()

                guess_wavefunction(
                    iwave, e, overlap_big, guess_type, self.sweep_params.one_dot,
                    i, True, 0.0)
                
                ket_rotate_matrix = VectorMatrix()
                traced_matrix = DensityMatrix()

                traced_matrix.allocate(overlap_new_system.ket_state_info)

                multiply_with_own_transpose(iwave, traced_matrix, 1.0)

                large_number = 1000000
                make_rotate_matrix(
                    traced_matrix, ket_rotate_matrix, large_number,
                    self.sweep_params.n_keep_qstates)
                
                traced_matrix.deallocate()

                iwave.save_wavefunction_info(overlap_big.ket_state_info,
                    overlap_new_system.sites, i)
                
                save_rotation_matrix(overlap_new_system.sites, ket_rotate_matrix, i)

                iwave.deallocate()

                overlap_new_system.transform_operators_2(
                    rotate_matrix, ket_rotate_matrix, False, False)
                
                overlap_new_system.store(forward, overlap_new_system.sites, cur_root, i)

                overlap_new_environment.deallocate()
                overlap_new_system.deallocate()
                overlap_environment.deallocate()
                overlap_system.deallocate()

                overlap_environment_dot.deallocate()
                overlap_system_dot.deallocate()
            
            Global.dmrginp.output_level = self.output_level

        Global.dmrginp.timer_operrot.stop()

        return new_system

    def gen_block_block_and_decimate(self, dot_with_sys, rot_mats=None, forward=True,
                                     implicit_trans=True, do_norms=None, do_comp=None):
        """Blocking and renormalization step for generating operators from rotation matrix."""
        new_system = Block()

        su2_used = Global.dmrginp.is_spin_adapted

        # figure out the range of dot blocks

        if forward:
            sys_dot_start = self.system.sites[-1] + \
                1 if su2_used else self.system.sites[-1] // 2 + 1
            sys_dot_end = sys_dot_start + self.sweep_params.sys_add - \
                1  # end means the index of last site
        else:
            sys_dot_start = self.system.sites[0] - \
                1 if su2_used else self.system.sites[0] // 2 + 1
            sys_dot_end = sys_dot_start - (self.sweep_params.sys_add - 1)

        system_dot = Block(sys_dot_start, sys_dot_end,
                           self.integral_index, implicit_trans)
        
        if do_norms is None:
            do_norms = dot_with_sys
        if do_comp is None:
            do_comp = not dot_with_sys

        if do_comp and OpTypes.CreDesComp not in self.system.ops:
            self.system.add_all_comp_ops()

        self.system.add_additional_ops()

        init_new_system_block(self.system, system_dot, new_system, -1, -1, self.sweep_params.sys_add, True,
                              self.integral_index, StorageTypes.DistributedStorage, do_norms, do_comp)

        if rot_mats is None:
            # read rot_mat from disk
            rotation_matrix = VectorMatrix()
            load_rotation_matrix(new_system.sites, rotation_matrix, -1)
        else:
            # read rot_mat from given dictionary
            rotation_matrix = rot_mats[tuple(new_system.sites)]
            save_rotation_matrix(new_system.sites, rotation_matrix, 0)
            save_rotation_matrix(new_system.sites, rotation_matrix, -1)
        
        new_system.transform_operators(rotation_matrix)

        # here implies system.clear()
        new_system.move_and_free_memory(self.system)

        return new_system
