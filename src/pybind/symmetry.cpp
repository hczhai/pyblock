
#include "IrrepSpace.h"
#include "SpinQuantum.h"
#include "SpinSpace.h"
#include "StateInfo.h"
#include "enumerator.h"
#include "global.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;
using namespace SpinAdapted;

PYBIND11_MAKE_OPAQUE(vector<SpinQuantum>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StateInfo>>);
PYBIND11_MAKE_OPAQUE(vector<int>);

PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

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
        .def(py::self == py::self)
        .def(py::self < py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::pickle(
            [](SpinQuantum *self) {
                py::list x(3);
                x[0] = self->particleNumber;
                x[1] = self->totalSpin.getirrep();
                x[2] = self->orbitalSymmetry.getirrep();
                return py::make_tuple(x);
            },
            [](py::tuple t) {
                py::list tt(t[0].cast<py::list>());
                SpinQuantum p(tt[0].cast<int>(),
                              SpinSpace(tt[1].cast<int>()),
                              IrrepSpace(tt[2].cast<int>()));
                return p;
            }
        ))
        .def("__repr__", [](SpinQuantum *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<SpinQuantum>>(m, "VectorSpinQuantum");

    py::class_<StateInfo, boost::shared_ptr<StateInfo>>(
        m, "StateInfo",
        "A collection of symmetry sectors. Each sector can contain several "
        "internal states (which can no longer be differentiated by "
        "symmetries), the number of which is also stored.")
        .def(py::init<>())
        .def(py::init([](const vector<SpinQuantum> &qs,
                         const vector<int> &ms) -> StateInfo {
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
        .def_readwrite("uncollected_state_info",
                       &StateInfo::unCollectedStateInfo)
        .def(
            "set_left_state_info",
            [](StateInfo *self, StateInfo *other) {
                self->leftStateInfo = other;
            },
            py::keep_alive<1, 2>())
        .def(
            "set_right_state_info",
            [](StateInfo *self, StateInfo *other) {
                self->rightStateInfo = other;
            },
            py::keep_alive<1, 2>())
        .def("set_uncollected_state_info",
             [](StateInfo *self, StateInfo *other) {
                 self->AllocateUnCollectedStateInfo();
                 *(self->unCollectedStateInfo) = *other;
             })
        .def_static(
            "transform_state", &StateInfo::transform_state,
            "Truncate state space based on dimension of rotation matrix.")
        .def("load", [](StateInfo *self, bool forward, const vector<int> &sites, int left) {
            StateInfo::restore(forward, sites, *self, left);
        })
        .def("save", [](StateInfo *self, bool forward, const vector<int> &sites, int left) {
            StateInfo::store(forward, sites, *self, left);
        })
        .def("collect_quanta", &StateInfo::CollectQuanta)
        .def_readwrite("n_total_states", &StateInfo::totalStates)
        .def("copy", [](StateInfo *self) {
            StateInfo x = *self;
            return x;
        })
        .def("__repr__", [](StateInfo *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<boost::shared_ptr<StateInfo>>>(m, "VectorStateInfo");

    m.def("state_tensor_product", [](StateInfo &a, StateInfo &b) {
        StateInfo c;
        TensorProduct(a, b, c, NO_PARTICLE_SPIN_NUMBER_CONSTRAINT, 0);
        return c;
    }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>());

    m.def("state_tensor_product_target", [](StateInfo &a, StateInfo &b) {
        StateInfo c;
        TensorProduct(a, b, c, PARTICLE_SPIN_NUMBER_CONSTRAINT);
        return c;
    }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>());
    
    m.def("get_commute_parity", &getCommuteParity, py::arg("a"), py::arg("b"), py::arg("c"));
    
    m.def("wigner_9j", [](int ja, int jb, int jc, int jd, int je, int jf, int jg, int jh, int ji) {
        return dmrginp.get_ninej()(ja, jb, jc, jd, je, jf, jg, jh, ji);
    });
}
