
#include "SpinQuantum.h"
#include "global.h"
#include "input.h"
#include "timer.h"
#ifdef _HAS_INTEL_MKL
#include <mkl.h>
#endif
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;
using namespace SpinAdapted;

PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

void ReadInput(char *conf);

double *stackmemory = nullptr;

void pybind_io(py::module &m)
{

    m.def("read_input",
          [](const string &conf) { ReadInput((char *)conf.c_str()); });

    m.def("init_stack_memory", []() {
        dmrginp.matmultFlops.resize(max(numthrds, dmrginp.quanta_thrds()), 0.);

#ifdef _HAS_INTEL_MKL
        mkl_set_num_threads(dmrginp.mkl_thrds());
        mkl_set_dynamic(0);
#endif

        cout.precision(12);
        cout << std::fixed;

        if (dmrginp.outputlevel() >= 0)
            cout << "allocating " << dmrginp.getMemory() << " doubles " << endl;

        stackmemory = new double[dmrginp.getMemory()];
        Stackmem.resize(numthrds);
        Stackmem[0].data = stackmemory;
        Stackmem[0].size = dmrginp.getMemory();
        dmrginp.initCumulTimer();

        block2::current_page = &Stackmem[0];
    });

    m.def("get_current_stack_memory", []() { return block2::current_page->memused; });

    m.def("set_current_stack_memory", [](size_t m) { block2::current_page->memused = m; });

    m.def("release_stack_memory", []() {
        if (stackmemory != nullptr)
            delete[] stackmemory;
    });

    py::enum_<algorithmTypes>(
        m, "AlgorithmTypes", py::arithmetic(),
        "Types of algorithm: one-dot or two-dot or other types.")
        .value("OneDot", ONEDOT)
        .value("TwoDot", TWODOT)
        .value("TwoDotToOneDot", TWODOT_TO_ONEDOT)
        .value("PartialSweep", PARTIAL_SWEEP);

    py::class_<Input>(m, "Input")
        .def(py::init<>())
        .def(py::init([](const string &filename) {
                 Input input(filename);
                 return input;
             }),
             "Initialize an Input object from the input file name.")
        .def_static(
            "read_input_contents",
            [](const string &contents) {
                Input input("", contents);
                return input;
            },
            "Initialize an Input object from the input file contents.")
        .def_property("output_level", &Input::outputlevel,
                      [](Input *self, int output_level) {
                          self->setOutputlevel() = output_level;
                      })
        .def_property("load_prefix", (const string&(Input::*)() const) & Input::load_prefix,
                      [](Input *self, const string &x) { self->load_prefix() = x; })
        .def_property("save_prefix", (const string&(Input::*)() const) & Input::save_prefix,
                      [](Input *self, const string &x) { self->save_prefix() = x; })
        .def_property_readonly("sweep_tol", &Input::get_sweep_tol)
        .def_property("spin_orbs_symmetry", &Input::spin_orbs_symmetry,
                      &Input::set_spin_orbs_symmetry,
                      "Spatial symmetry (irrep number) of each spin-orbital.")
        .def_property(
            "molecule_quantum", &Input::molecule_quantum,
            [](Input *self, const SpinQuantum &q) {
                self->set_molecule_quantum() = q;
            },
            "Symmetry of target state.")
        .def("effective_molecule_quantum_vec",
             &Input::effective_molecule_quantum_vec,
             "Often this simply returns a vector containing one "
             "``molecule_quantum``. For non-interacting orbitals or Bogoliubov "
             "algorithm, this may be more than that.")
        .def_property(
            "algorithm_type", &Input::algorithm_method,
            [](Input *self, algorithmTypes t) {
                self->set_algorithm_method() = t;
            },
            "Algorithm type: one-dot or two-dot or other types.")
        .def_property_readonly("twodot_to_onedot_iter",
                               &Input::twodot_to_onedot_iter,
                               "Indicating at which sweep iteration the "
                               "switching from two-dot to one-dot will happen.")
        .def_property_readonly(
            "n_max_iters", &Input::max_iter,
            "The maximal number of sweep iterations (outer loop).")
        .def_property_readonly("slater_size", &Input::slater_size,
                               "Number of spin-orbitals")
        .def_property_readonly("n_electrons", &Input::real_particle_number,
                               "Number of electrons")
        .def("n_roots", (int (Input::*)(int) const)(&Input::nroots),
             "Get number of states to solve for given sweep iteration.",
             py::arg("sweep_iter"))
        .def_readwrite(
            "timer_guessgen", &Input::guessgenT,
            "Timer for generating or loading dot blocks and environment block.")
        .def_readwrite("timer_multiplier", &Input::multiplierT,
                       "Timer for blocking.")
        .def_readwrite("timer_operrot", &Input::operrotT,
                       "Timer for operator rotation.")
        .def_property(
            "is_spin_adapted", &Input::spinAdapted, [](Input *self, bool sa) { self->spinAdapted() = sa; },
            "Indicates whether SU(2) symmetry is utilized. If SU(2) is not "
            "used, The Abelian subgroup of SU(2) (Sz symmetry) is used.")
        .def_property_readonly("hf_occupancy", &Input::hf_occupancy);

    class Global
    {
    };

    py::class_<Global>(m, "Global", "Wrapper for global variables.")
        .def_property_static(
            "dmrginp", [](py::object) -> Input & { return dmrginp; },
            [](py::object, const Input &input) { dmrginp = input; })
        .def_property_static(
            "non_abelian_sym",
            [](py::object) -> bool { return NonabelianSym; },
            [](py::object, bool b) { NonabelianSym = b; })
        .def_property_static(
            "point_group",
            [](py::object) -> string { return sym; },
            [](py::object, string b) { sym = b; });

    py::class_<Timer>(m, "Timer")
        .def(py::init<>())
        .def(py::init<bool>(), "With a `bool` parameter indicating whether"
                               " the :class:`Timer` should start immediately.")
        .def("start", &Timer::start)
        .def("elapsed_walltime", &Timer::elapsedwalltime)
        .def("elapsed_cputime", &Timer::elapsedcputime);

    py::class_<cumulTimer, boost::shared_ptr<cumulTimer>>(m, "CumulTimer")
        .def(py::init<>())
        .def("start", &cumulTimer::start)
        .def("reset", &cumulTimer::reset)
        .def("stop", &cumulTimer::stop)
        .def("__repr__", [](cumulTimer *self) {
            stringstream ss;
            ss.precision(12);
            ss << fixed << *self;
            return ss.str();
        });
}
