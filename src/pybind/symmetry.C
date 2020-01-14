
#include "IrrepSpace.h"
#include "SpinQuantum.h"
#include "SpinSpace.h"
#include "StateInfo.h"
#include "enumerator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;
using namespace SpinAdapted;

PYBIND11_MAKE_OPAQUE(vector<SpinQuantum>);
PYBIND11_MAKE_OPAQUE(vector<int>);

void pybind_symmetry(py::module &m) {

    py::class_<SpinSpace>(m, "SpinSpace",
                          "A wrapper class for the spin irrep.\n\n"
                          "In :math:`S_z` symmetry, the irrep is :math:`2S_z`. "
                          "In SU(2) symmetry, the irrep is :math:`2S`. "
                          "The behaviour is toggled checking "
                          ":attr:`block.io.Global.dmrginp.spin_adapted`.")
        .def(py::init<>())
        .def(py::init<int>())
        .def_property_readonly("irrep", &SpinSpace::getirrep)
        .def("__repr__", [](SpinSpace *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<IrrepSpace>(
        m, "IrrepSpace",
        "A wrapper class for molecular point group symmetry irrep.")
        .def(py::init<>())
        .def(py::init<int>())
        .def_property_readonly("irrep", &IrrepSpace::getirrep)
        .def("__repr__", [](IrrepSpace *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<SpinQuantum>(
        m, "SpinQuantum",
        "A collection of quantum numbers associated with a specific state"
        " (irreducible representation). One such collection defines a specific "
        "sector in the state space.")
        .def(py::init<>())
        .def(py::init<const int, const SpinSpace, const IrrepSpace>())
        .def_readwrite("s", &SpinQuantum::totalSpin,
                       "Irreducible representation for spin symmetry "
                       "(:math:`S` or :math:`S_z`).")
        .def_readwrite("n", &SpinQuantum::particleNumber, "Particle number.")
        .def_readwrite(
            "symm", &SpinQuantum::orbitalSymmetry,
            "Irreducible representation for molecular point group symmetry.")
        .def("__repr__", [](SpinQuantum *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<SpinQuantum>>(m, "VectorSpinQuantum");

    py::class_<StateInfo>(
        m, "StateInfo",
        "A collection of symmetry sectors. Each sector can contain several "
        "internal states (which can no longer be differentiated by "
        "symmetries), the number of which is also stored.")
        .def(py::init<>())
        .def(py::init([](const vector<SpinQuantum> &qs, const vector<int> &ms) -> StateInfo {
            StateInfo si((int)qs.size(), &qs[0], &ms[0]);
            return si;
        }))
        .def_readwrite("quanta", &StateInfo::quanta,
                       "Quantum numbers for a set of sites.")
        .def_readwrite("n_states", &StateInfo::quantaStates,
                       "Number of states per (combined) quantum number.")
        .def_readwrite("left_unmap_quanta", &StateInfo::leftUnMapQuanta,
                       "Index in left StateInfo, for each combined state.")
        .def_readwrite("right_unmap_quanta", &StateInfo::rightUnMapQuanta,
                       "Index in right StateInfo, for each combined state.")
        .def_readwrite("old_to_new_state", &StateInfo::oldToNewState,
                       "old_to_new_state[i] = [k1, k2, k3, ...] where i is the "
                       "index in the collected StateInfo and k's are indices "
                       "in the uncollected StateInfo.")
        .def_readwrite("left_state_info", &StateInfo::leftStateInfo)
        .def_readwrite("right_state_info", &StateInfo::rightStateInfo)
        .def_static(
            "transform_state", &StateInfo::transform_state,
            "Truncate state space based on dimension of rotation matrix.")
        .def("collect_quanta", &StateInfo::CollectQuanta)
        .def("copy",
             [](StateInfo *self) {
                 StateInfo x = *self;
                 return x;
             })
        .def("__repr__", [](StateInfo *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    m.def("tensor_product", [](StateInfo &a, StateInfo &b) {
        StateInfo c;
        TensorProduct(a, b, c, NO_PARTICLE_SPIN_NUMBER_CONSTRAINT, 0);
        return c;
    });

    m.def("tensor_product_target", [](StateInfo &a, StateInfo &b) {
        StateInfo c;
        TensorProduct(a, b, c, PARTICLE_SPIN_NUMBER_CONSTRAINT);
        return c;
    });
}
