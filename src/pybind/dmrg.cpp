
#include "Stackspinblock.h"
#include "enumerator.h"
#include "fciqmchelper.h"
#include "sweep.h"
#include "sweep_params.h"
#include "stackguess_wavefunction.h"
#include "wrapper.h"
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;
using namespace SpinAdapted;

PYBIND11_MAKE_OPAQUE(vector<bool>);
PYBIND11_MAKE_OPAQUE(vector<::Matrix>);
PYBIND11_MAKE_OPAQUE(vector<StackSpinBlock>);

void dmrg(double sweep_tol);
int calldmrg(char *, char *);

void pybind_dmrg(py::module &m) {

    py::class_<SweepParams>(m, "SweepParams")
        .def(py::init<>())
        .def_property(
            "current_root",
            (const int &(SweepParams::*)() const)(&SweepParams::current_root),
            [](SweepParams *self, int root) { self->current_root() = root; })
        .def_property(
            "sweep_iter", &SweepParams::get_sweep_iter,
            [](SweepParams *self, int iter) { self->set_sweep_iter() = iter; },
            "Counter for controlling the sweep iteration (outer loop).")
        .def_property(
            "block_iter", &SweepParams::get_block_iter,
            [](SweepParams *self, int iter) { self->set_block_iter() = iter; },
            "Counter for controlling the blocking iteration (inner loop).")
        .def_property(
            "n_block_iters", &SweepParams::get_n_iters,
            [](SweepParams *self, int iter) { self->set_n_iters() = iter; },
            "The number of blocking iterations (inner loops) needed in one "
            "sweep.")
        .def_property(
            "n_keep_states", &SweepParams::get_keep_states,
            [](SweepParams *self, int n) { self->set_keep_states() = n; },
            "The bond dimension for states in current sweep.")
        .def_property(
            "n_keep_qstates", &SweepParams::get_keep_qstates,
            [](SweepParams *self, int n) { self->set_keep_qstates() = n; },
            "(May not be useful.)")
        .def_property(
            "largest_dw", &SweepParams::get_largest_dw,
            [](SweepParams *self, double dw) { self->set_largest_dw() = dw; },
            "Largest discarded weight (or largest error).")
        .def_property("lowest_energy", &SweepParams::get_lowest_energy,
                      [](SweepParams *self, const vector<double> &vd) {
                          self->set_lowest_energy() = vd;
                      })
        .def_property("lowest_energy_spin", &SweepParams::get_lowest_energy_spins,
                      [](SweepParams *self, const vector<double> &vd) {
                          self->set_lowest_energy_spins() = vd;
                      })
        .def_property("lowest_error", &SweepParams::get_lowest_error,
                      [](SweepParams *self, double error) {
                          self->set_lowest_error() = error;
                      })
        .def_property("davidson_tol", &SweepParams::get_davidson_tol,
                      [](SweepParams *self, double t) {
                          self->set_davidson_tol() = t;
                      })
        .def_property(
            "n_block_iters", &SweepParams::get_n_iters,
            [](SweepParams *self, int iter) { self->set_n_iters() = iter; },
            "The number of blocking iterations (inner loops) needed in one "
            "sweep.")
        .def_property(
            "forward_starting_size",
            &SweepParams::get_forward_starting_size,
            [](SweepParams *self, int size) {
                self->set_forward_starting_size() = size;
            },
            "Initial size of system block if in forward direction.")
        .def_property(
            "backward_starting_size",
            &SweepParams::get_backward_starting_size,
            [](SweepParams *self, int size) {
                self->set_backward_starting_size() = size;
            },
            "Initial size of system block if in backward direction.")
        .def_property(
            "sys_add", &SweepParams::get_sys_add,
            [](SweepParams *self, int size) { self->set_sys_add() = size; },
            "The dot block size near system block.")
        .def_property(
            "env_add", &SweepParams::get_env_add,
            [](SweepParams *self, int size) { self->set_env_add() = size; },
            "The dot block size near environment block.")
        .def_property(
            "one_dot", &SweepParams::get_onedot,
            [](SweepParams *self, bool od) { self->set_onedot() = od; },
            "Whether it is the one-dot scheme.")
        .def_property("guess_type", &SweepParams::get_guesstype,
                      [](SweepParams *self, guessWaveTypes t) {
                          self->set_guesstype() = t;
                      })
        .def_property("noise", &SweepParams::get_noise,
                      [](SweepParams *self, double t) {
                          self->set_noise() = t;
                      })
        .def_property("additional_noise", &SweepParams::get_additional_noise,
                      [](SweepParams *self, double t) {
                          self->set_additional_noise() = t;
                      })
        .def("set_sweep_parameters", &SweepParams::set_sweep_parameters)
        .def("restorestate", [](SweepParams *self) {
            bool forward;
            int size;
            self->restorestate(forward, size);
            return make_pair(forward, size);
        })
        .def("save_state", &SweepParams::savestate, "Save the sweep direction and "
                                                    "number of sites in system block into the disk file "
                                                    "'statefile.*.tmp'.",
             py::arg("forward"), py::arg("size"));

    m.def("block_and_decimate", &Sweep::BlockAndDecimate,
          "Block and decimate to generate the new system block.",
          py::arg("sweep_params"), py::arg("system"), py::arg("new_system"),
          py::arg("use_slater"), py::arg("dot_with_sys"));

    m.def("get_dot_with_sys",
          [](const StackSpinBlock &system, bool one_dot, bool forward) {
              bool dot_with_sys = true;
              SweepParams sp;
              sp.set_onedot() = one_dot;
              Sweep::set_dot_with_sys(dot_with_sys, system, sp, forward);
              return dot_with_sys;
          },
          "Return the `dot_with_sys` variable, determing whether the "
          "complementary operators should be defined based on system block "
          "indicies.",
          py::arg("system"), py::arg("one_dot"), py::arg("forward"));

    m.def("do_one", &Sweep::do_one, "Perform one sweep procedure.",
          py::arg("sweep_params"), py::arg("warm_up"), py::arg("forward"),
          py::arg("restart"), py::arg("restart_size"));

    m.def("dmrg", &dmrg, "Perform DMRG calculation.", py::arg("sweep_tol"));

    m.def("calldmrg",
          [](const string &conf) { calldmrg((char *)conf.c_str(), 0); },
          "Global driver.", py::arg("input_file_name"));

    m.def("make_system_environment_big_overlap_blocks",
          &Sweep::makeSystemEnvironmentBigOverlapBlocks,
          py::arg("system_sites"), py::arg("system_dot"),
          py::arg("environment_dot"), py::arg("system"), py::arg("new_system"),
          py::arg("environment"), py::arg("new_environment"), py::arg("big"),
          py::arg("sweep_params"), py::arg("dot_with_sys"),
          py::arg("use_slater"), py::arg("integral_index"),
          py::arg("bra_state"), py::arg("ket_state"));

    // second overloading
    m.def("guess_wavefunction",
          (void (*)(StackWavefunction &, DiagonalMatrix &, const StackSpinBlock &,
                  const guessWaveTypes &, const bool &, const int &,
                  const bool &, double)) & GuessWave::guess_wavefunctions,
          py::arg("solution"), py::arg("e"), py::arg("big"),
          py::arg("guess_wave_type"), py::arg("one_dot"), py::arg("state"),
          py::arg("transpose_guess_wave"), py::arg("additional_noise") = 0.0);

    py::class_<MPS>(m, "MPS")
        .def(py::init<>())
        .def(py::init<vector<bool> &>())
        .def("get_site_tensors",
             (std::vector<::Matrix> & (MPS::*)(int i)) & MPS::getSiteTensors)
        .def("get_w", &MPS::getw)
        .def_readwrite_static("site_blocks", &MPS::siteBlocks)
        .def_readwrite_static("n_sweep_iters", &MPS::sweepIters,
                              "The number of ``site_tensors``.")
        .def("write_to_disk", &MPS::writeToDiskForDMRG, py::arg("state_index"),
             py::arg("write_state_average") = false);

    m.def("MPS_init", &readMPSFromDiskAndInitializeStaticVariables,
          "Initialize the single site blocks :attr:`MPS.site_blocks`. ");
}
