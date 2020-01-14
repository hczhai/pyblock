
#include "Stackspinblock.h"
#include "Stackwavefunction.h"
#include "enumerator.h"
#include "initblocks.h"
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;
using namespace SpinAdapted;

PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

PYBIND11_MAKE_OPAQUE(vector<SpinQuantum>);
PYBIND11_MAKE_OPAQUE(map<opTypes, boost::shared_ptr<StackOp_component_base>>);
PYBIND11_MAKE_OPAQUE(vector<StackSpinBlock>);

void pybind_block(py::module &m) {

    py::enum_<guessWaveTypes>(m, "GuessWaveTypes", py::arithmetic(),
                              "Types of guess wavefunction for initialize "
                              "Davidson algorithm (enumerator).")
        .value("Basic", BASIC)
        .value("Transform", TRANSFORM)
        .value("Transpose", TRANSPOSE);

    py::enum_<Storagetype>(m, "StorageTypes", py::arithmetic(),
                           "Types of storage (enumerator).")
        .value("LocalStorage", LOCAL_STORAGE)
        .value("DistributedStorage", DISTRIBUTED_STORAGE);

    py::bind_map<map<opTypes, boost::shared_ptr<StackOp_component_base>>>(
        m, "MapOperators");

    py::class_<StackSpinBlock>(m, "Block")
        .def(py::init<>())
        .def(py::init<int, int, int, bool, bool>(), py::arg("start"),
             py::arg("end"), py::arg("integral_index"),
             py::arg("implicit_transpose"), py::arg("is_complement") = false)
        .def_property_readonly("name", &StackSpinBlock::get_name,
                               "A random integer.")
        .def_property("sites", &StackSpinBlock::get_sites,
                      [](StackSpinBlock *self, const vector<int> &sites) {
                          self->set_sites() = sites;
                      },
                      "List of indices of sites contained in the block.")
        .def_property("bra_state_info", &StackSpinBlock::get_braStateInfo,
                      [](StackSpinBlock *self, const StateInfo &info) {
                          self->set_braStateInfo() = info;
                      })
        .def_property("ket_state_info", &StackSpinBlock::get_ketStateInfo,
                      [](StackSpinBlock *self, const StateInfo &info) {
                          self->set_ketStateInfo() = info;
                      })
        .def_property("loop_block", &StackSpinBlock::is_loopblock,
                      &StackSpinBlock::set_loopblock,
                      "Whether the block is loop block.")
        .def_property_readonly(
            "ops", &StackSpinBlock::get_ops,
            "Map from operator types to matrix representation of operators.")
        .def("print_operator_summary", &StackSpinBlock::printOperatorSummary,
             "Print operator summary when :attr:`block.io.Input.output_level` "
             "at least = 2.")
        .def("store",
             [](StackSpinBlock *self, bool forward, const vector<int> &sites,
                int left, int right) {
                 StackSpinBlock::store(forward, sites, *self, left, right);
             },
             "Store a :class:`Block` into disk.\n\n"
             "Args:\n"
             "    forward : bool\n"
             "        The direction of sweep.\n"
             "    sites : :class:`block.VectorInt`\n"
             "        List of indices of sites contained in the block. "
             "This is kind of redundant and can be obtained "
             "from :attr:`Block.sites`.\n"
             "    block : :class:`Block`\n"
             "        The block to store.\n"
             "    left : int\n"
             "        Bra state.\n"
             "    right : int\n"
             "        Ket state.",
             py::arg("forward"), py::arg("sites"), py::arg("left"),
             py::arg("right"))
        .def("deallocate", &StackSpinBlock::deallocate)
        .def("clear", &StackSpinBlock::clear)
        .def("transform_operators",
             (void (StackSpinBlock::*)(vector<Matrix> &)) &
                 StackSpinBlock::transform_operators)
        .def("transform_operators_2",
             (void (StackSpinBlock::*)(vector<Matrix> &, vector<Matrix> &, bool,
                                       bool)) &
                 StackSpinBlock::transform_operators,
             py::arg("left_rotate_matrix"), py::arg("right_rotate_matrix"),
             py::arg("clear_right_block") = true,
             py::arg("clear_left_block") = true)
        .def("move_and_free_memory",
             [](StackSpinBlock *self, StackSpinBlock *system) {
                 long memoryToFree = self->getdata() - system->getdata();
                 long newsysmem = self->memoryUsed();
                 self->moveToNewMemory(system->getdata());
                 Stackmem[0].deallocate(self->getdata() + newsysmem,
                                        memoryToFree);
                 system->clear();
             },
             "If the parameter ``system`` is allocated before ``this`` object, "
             "but we need to free ``system``. Then we have to move the memory "
             "of ``this`` to ``system`` then clear ``system``.")
        .def("add_additional_ops", &StackSpinBlock::addAdditionalOps)
        .def("remove_additional_ops", &StackSpinBlock::removeAdditionalOps)
        .def("add_all_comp_ops", &StackSpinBlock::addAllCompOps)
        .def("multiply_overlap", &StackSpinBlock::multiplyOverlap, py::arg("c"),
             py::arg("v"), py::arg("num_threads") = 1)
        .def("renormalize_from",
             [](StackSpinBlock *self, vector<double> &energies,
                vector<double> &spins, double error,
                vector<Matrix> &rotateMatrix, const int keptstates,
                const int keptqstates, const double tol, StackSpinBlock &big,
                const guessWaveTypes &guesswavetype, double noise,
                double additional_noise, bool onedot, StackSpinBlock &System,
                StackSpinBlock &sysDot, StackSpinBlock &environment,
                const bool &dot_with_sys, const bool &warmUp, int sweepiter,
                int currentRoot, std::vector<StackWavefunction> &lowerStates) {
                 self->RenormaliseFrom(
                     energies, spins, error, rotateMatrix, keptstates,
                     keptqstates, tol, big, guesswavetype, noise,
                     additional_noise, onedot, System, sysDot, environment,
                     dot_with_sys, warmUp, sweepiter, currentRoot, lowerStates);
                 return error;
             },
             py::arg("energies"), py::arg("spins"), py::arg("error"),
             py::arg("rotate_matrix"), py::arg("kept_states"),
             py::arg("kept_qstates"), py::arg("tol"), py::arg("big"),
             py::arg("guess_wave_type"), py::arg("noise"),
             py::arg("additional_noise"), py::arg("one_dot"), py::arg("system"),
             py::arg("system_dot"), py::arg("environment"),
             py::arg("dot_with_sys"), py::arg("warm_up"), py::arg("sweep_iter"),
             py::arg("current_root"), py::arg("lower_states"))
        .def("__repr__", [](StackSpinBlock *self) {
            stringstream ss;
            ss.precision(12);
            ss << fixed << *self;
            return ss.str();
        });

    py::bind_vector<vector<StackSpinBlock>>(m, "VectorBlock");

    m.def("init_starting_block", &InitBlocks::InitStartingBlock,
          "Initialize starting block", py::arg("starting_block"),
          py::arg("forward"), py::arg("left_state"), py::arg("right_state"),
          py::arg("forward_starting_size"), py::arg("backward_starting_size"),
          py::arg("restart_size"), py::arg("restart"), py::arg("warm_up"),
          py::arg("integral_index"),
          py::arg("bra_quanta") = vector<SpinQuantum>(),
          py::arg("ket_quanta") = vector<SpinQuantum>());

    m.def("init_big_block", &InitBlocks::InitBigBlock,
          "Initialize big (super) block.", py::arg("left_block"),
          py::arg("right_block"), py::arg("big_block"),
          py::arg("bra_quanta") = vector<SpinQuantum>(),
          py::arg("ket_quanta") = vector<SpinQuantum>());

    m.def("init_new_system_block",
          [](StackSpinBlock &system, StackSpinBlock &system_dot,
             StackSpinBlock &new_system, int left_state, int right_state,
             int sys_add, bool direct, int integral_index, Storagetype storage,
             bool have_norm_ops, bool have_comp_ops) {
              InitBlocks::InitNewSystemBlock(
                  system, system_dot, new_system, left_state, right_state,
                  sys_add, direct, integral_index, storage, have_norm_ops,
                  have_comp_ops, NO_PARTICLE_SPIN_NUMBER_CONSTRAINT);
          },
          "Initialize new system block", py::arg("system"),
          py::arg("system_dot"), py::arg("new_system"), py::arg("left_state"),
          py::arg("right_state"), py::arg("sys_add"), py::arg("direct"),
          py::arg("integral_index"), py::arg("storage"),
          py::arg("have_norm_ops"), py::arg("have_comp_ops"));

    m.def("init_new_environment_block",
          [](StackSpinBlock &environment, StackSpinBlock &environment_dot,
             StackSpinBlock &new_environment, StackSpinBlock &system,
             StackSpinBlock &system_dot, int left_state, int right_state,
             int sys_add, int env_add, bool forward, bool direct, bool one_dot,
             bool use_slater, int integral_index, bool have_norm_ops,
             bool have_comp_ops, bool dot_with_sys) {
              InitBlocks::InitNewEnvironmentBlock(
                  environment, environment_dot, new_environment, system,
                  system_dot, left_state, right_state, sys_add, env_add,
                  forward, direct, one_dot, 1, use_slater, integral_index,
                  have_norm_ops, have_comp_ops, dot_with_sys,
                  NO_PARTICLE_SPIN_NUMBER_CONSTRAINT);
          },
          "Initialize new environment block", py::arg("environment"),
          py::arg("environment_dot"), py::arg("new_environment"),
          py::arg("system"), py::arg("system_dot"), py::arg("left_state"),
          py::arg("right_state"), py::arg("sys_add"), py::arg("env_add"),
          py::arg("forward"), py::arg("direct"), py::arg("one_dot"),
          py::arg("use_slater"), py::arg("integral_index"),
          py::arg("have_norm_ops"), py::arg("have_comp_ops"),
          py::arg("dot_with_sys"));
}
