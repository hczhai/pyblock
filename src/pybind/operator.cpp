
#include "StackBaseOperator.h"
#include "StackMatrix.h"
#include "StackOperators.h"
#include "Stack_op_components.h"
#include "Stackdensity.h"
#include "Stackwavefunction.h"
#include "enumerator.h"
#include "operatorfunctions.h"
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <string>

namespace py = pybind11;
using namespace std;
using namespace Eigen;
using namespace SpinAdapted;

py::tuple pickle_stack_sparse_matrix(StackSparseMatrix *self);
StackSparseMatrix unpickle_stack_sparse_matrix(py::tuple t);

typedef vector<pair<pair<int, int>, StackMatrix>> nz_blocks;
typedef map<pair<int, int>, int> nz_map;

PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

PYBIND11_MAKE_OPAQUE(vector<int>);
PYBIND11_MAKE_OPAQUE(vector<vector<int>>);
PYBIND11_MAKE_OPAQUE(vector<::Matrix>);
PYBIND11_MAKE_OPAQUE(vector<SpinQuantum>);
PYBIND11_MAKE_OPAQUE(vector<StackWavefunction>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StateInfo>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackOp_component_base>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackSparseMatrix>>);

PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackCre>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackDes>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackCreDes>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackDesCre>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackCreCre>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackDesDes>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackCreDesComp>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackDesCreComp>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackDesDesComp>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackCreCreComp>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackCreCreDesComp>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackCreDesDesComp>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackHam>>);
PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StackOverlap>>);

PYBIND11_MAKE_OPAQUE(nz_blocks);
PYBIND11_MAKE_OPAQUE(nz_map);

template <size_t D> struct pybind_array_op {
    template <typename PyT, typename OpT> static void m(PyT &obj);
};

template <>
template <typename PyT, typename OpT>
void pybind_array_op<0>::m(PyT &obj) {
    obj.def("has_global", [](OpT *self) { return self->has(0); },
            "Query whether the element is non-zero (in local or global "
            "storage).")
        .def("has_local", [](OpT *self) { return self->has_local_index(0); },
             "Query whether the element is non-zero in local storage.")
        .def("local_element", [](OpT *self) { return self->get_element(0); },
             "Get the array of operators (for different spin "
             "quantum numbers, in local "
             "storage).");
}

template <>
template <typename PyT, typename OpT>
void pybind_array_op<1>::m(PyT &obj) {
    obj.def("has_global", [](OpT *self, int i) { return self->has(i); },
            "Query whether the element is non-zero (in local or global "
            "storage). The parameters are site indices.")
        .def("has_local",
             [](OpT *self, int i) { return self->has_local_index(i); },
             "Query whether the element is non-zero in local storage. The "
             "parameters are site indices.")
        .def("local_element",
             [](OpT *self, int i) { return self->get_element(i); },
             "Get an array of operators (for different spin quantum numbers) "
             "defined for the given site indices (in local "
             "storage).");
}

template <>
template <typename PyT, typename OpT>
void pybind_array_op<2>::m(PyT &obj) {
    obj.def("has_global",
            [](OpT *self, int i, int j) { return self->has(i, j); },
            "Query whether the element is non-zero (in local or global "
            "storage). The parameters are site indices.")
        .def(
            "has_local",
            [](OpT *self, int i, int j) { return self->has_local_index(i, j); },
            "Query whether the element is non-zero in local storage. The "
            "parameters are site indices.")
        .def("local_element",
             [](OpT *self, int i, int j) { return self->get_element(i, j); },
             "Get an array of operators (for different spin quantum numbers) "
             "defined for the given site indices (in local "
             "storage).");
}

template <size_t D, typename T>
void pybind_stack_op_component(py::module &m, const string &name) {

    py::class_<T, boost::shared_ptr<T>, StackSparseMatrix>(
        m, ("Operator" + name).c_str())
        .def(py::init<>());

    py::bind_vector<vector<boost::shared_ptr<T>>>(m, ("Vector" + name).c_str());

    py::class_<StackOp_component<T>, boost::shared_ptr<StackOp_component<T>>,
               StackOp_component_base>
        obj(m, ("OperatorArray" + name).c_str(),
            ("An array of " + name + " operators defined at different sites.")
                .c_str());

    obj.def(py::init<>())
        .def_property_readonly(
            "op_string", &StackOp_component<T>::get_op_string,
            "Name of the type of operators contained in this array.")
        .def("local_element_linear", &StackOp_component<T>::get_local_element,
             "Get an array of operators (for different spin quantum numbers) "
             "defined for the given (flattened) linear index (in local "
             "storage).")
        .def("global_element_linear", &StackOp_component<T>::get_global_element,
             "Get an array of operators (for different spin quantum numbers) "
             "defined for the given (flattened) linear index (in global "
             "storage).")
        .def_property_readonly("n_local_nz", &StackOp_component<T>::get_size,
                               "Number of non-zero elements in local storage.")
        .def_property_readonly("n_global_nz", &StackOp_component<T>::size,
                               "Number of non-zero elements in global storage.")
        .def_property_readonly(
            "local_indices", &StackOp_component<T>::get_array,
            "A 2d array contains the site indices of non-zero elements in "
            "local storage. It gives a map from flattened single index to "
            "multiple site indices "
            "(which is represented as an array).")
        .def_property_readonly(
            "global_indices", &StackOp_component<T>::get_global_array,
            "A 1d array contains the site indices of non-zero elements (in "
            "local or global storage). It gives a map from flattened single "
            "index to multiple site indices. "
            "Then this array itself is flattened.");

    pybind_array_op<D>::template m<decltype(obj), StackOp_component<T>>(obj);
}

void pybind_operator(py::module &m) {

    py::enum_<opTypes>(m, "OpTypes", py::arithmetic(),
                       "Types of operators (enumerator).")
        .value("Hamiltonian", HAM)
        .value("Cre", CRE)
        .value("CreCre", CRE_CRE)
        .value("DesDesComp", DES_DESCOMP)
        .value("CreDes", CRE_DES)
        .value("CreDesComp", CRE_DESCOMP)
        .value("CreCreDesComp", CRE_CRE_DESCOMP)
        .value("Des", DES)
        .value("DesDes", DES_DES)
        .value("CreCreComp", CRE_CRECOMP)
        .value("DesCre", DES_CRE)
        .value("DesCreComp", DES_CRECOMP)
        .value("CreDesDesComp", CRE_DES_DESCOMP)
        .value("Overlap", OVERLAP);

    py::class_<StackMatrix>(
        m, "StackMatrix",
        "Very simple Matrix class that provides a Matrix type interface for a "
        "double array. It does not own its own data.\n\n"
        "Note that the C++ class used indices counting from 1. Here we count "
        "from 0. Row-major (C) storage.")
        .def(py::init<>())
        .def(py::init(
            [](Ref<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>> mat) {
                StackMatrix smat(mat.data(), mat.rows(), mat.cols());
                return smat;
            }))
        .def_property(
            "ref",
            [](StackMatrix *self) {
                return Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(
                    self->Store(), self->Nrows(), self->Ncols());
            },
            [](StackMatrix *self,
               Ref<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>> mat) {
                self->allocate(mat.data(), mat.rows(), mat.cols());
            },
            "A numpy.ndarray reference.")
        .def_property(
            "rows", (const int &(StackMatrix::*)() const)(&StackMatrix::Nrows),
            [](StackMatrix *self, int m) { self->Nrows() = m; })
        .def_property(
            "cols", (const int &(StackMatrix::*)() const)(&StackMatrix::Ncols),
            [](StackMatrix *self, int m) { self->Ncols() = m; })
        .def(py::pickle(
            [](StackMatrix *self) {
                return py::make_tuple((size_t)self->Store(), self->Nrows(), self->Ncols());
            },
            [](py::tuple t) {
                StackMatrix p((double*)t[0].cast<size_t>(), t[1].cast<int>(), t[2].cast<int>());
                return p;
            }
        ))
        .def("__repr__", [](StackMatrix *self) {
            stringstream ss;
            ss << Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(
                self->Store(), self->Nrows(), self->Ncols());
            return ss.str();
        });

    py::bind_vector<nz_blocks>(m, "VectorNonZeroStackMatrix");
    py::bind_map<nz_map>(m, "MapPairInt");

    py::class_<StackSparseMatrix, boost::shared_ptr<StackSparseMatrix>>(
        m, "StackSparseMatrix",
        "Block-sparse matrix. \n"
        "Non-zero blocks are identified by symmetry (quantum numbers) "
        "requirements and stored as :class:`StackMatrix` objects")
        .def(py::init<>())
        .def_property(
            "total_memory",
            [](StackSparseMatrix *self) { return self->set_totalMemory(); },
            [](StackSparseMatrix *self, long m) {
                self->set_totalMemory() = m;
            })
        .def_property(
            "initialized", &StackSparseMatrix::get_initialised,
            [](StackSparseMatrix *self, bool i) {
                self->set_initialised() = i;
            })
        .def("operator_element", (StackMatrix& (StackSparseMatrix::*)(int i, int j))
             &StackSparseMatrix::operator_element)
        .def_property(
            "non_zero_blocks",
            [](StackSparseMatrix *self) { return self->get_nonZeroBlocks(); },
            [](StackSparseMatrix *self, const nz_blocks &m) {
                self->get_nonZeroBlocks() = m;
            },
            "A list of non zero blocks. Each element in the list is a pair of "
            "a pair of bra and ket indices, and :class:`StackMatrix`.")
        .def_property(
            "ref",
            [](StackSparseMatrix *self) {
                return Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(
                    self->get_data(), 1, self->set_totalMemory());
            },
            [](StackSparseMatrix *self,
               Ref<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>> mat) {
                   assert(self->set_totalMemory() == mat.rows() * mat.cols());
                   memcpy(self->get_data(), mat.data(),
                       sizeof(double) * mat.rows() * mat.cols());
            },
            "A numpy.ndarray reference.")
        .def("allocate_memory", [](StackSparseMatrix *self, long m) {
            assert(self->set_totalMemory() == 0);
            self->set_totalMemory() = m;
            self->set_data(block2::current_page->allocate(m));
            self->allocateOperatorMatrix();
        })
        .def(py::pickle(
            [](StackSparseMatrix *self) {
                return pickle_stack_sparse_matrix(self);
            },
            [](py::tuple t) {
                return unpickle_stack_sparse_matrix(t);
            }
        ))
        .def("__repr__", [](StackSparseMatrix *self) {
            stringstream ss;
            for (auto &r: self->get_nonZeroBlocks()) {
                ss << "[SP] (" << r.first.first << ", " << r.first.second << ") = ["
                   << r.second.Nrows() << " x " << r.second.Ncols() << "]" << endl;
                ss << Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(
                    r.second.Store(), r.second.Nrows(), r.second.Ncols()) << endl;
            }
            return ss.str();
        })
        .def_property("map_to_non_zero_blocks",
                      [](StackSparseMatrix *self) {
                          return self->get_mapToNonZeroBlocks();
                      },
                      [](StackSparseMatrix *self, const nz_map &m) {
                          self->get_mapToNonZeroBlocks() = m;
                      },
                      "A map from pair of bra and ket indices, to the index in "
                      ":attr:`StackSparseMatrix.non_zero_blocks`.")
        .def_property("fermion", &StackSparseMatrix::get_fermion,
            [](StackSparseMatrix *self, bool f) {
                self->set_fermion() = f;
            })
        .def_property(
            "delta_quantum",
            [](StackSparseMatrix *self) {
                if (self->conjugacy() == 'n')
                    return self->set_deltaQuantum();
                else {
                    vector<SpinQuantum> vs = self->set_deltaQuantum();
                    for (size_t i = 0; i < vs.size(); i++)
                        vs[i] = -vs[i];
                    return vs;
                }
            },
            [](StackSparseMatrix *self, const vector<SpinQuantum> &m) {
                assert(self->conjugacy() == 'n');
                self->set_deltaQuantum() = m;
            },
            "Allowed change of quantum numbers between states.")
        .def_property_readonly(
            "rows", (int (StackSparseMatrix::*)() const)(&StackSparseMatrix::nrows))
        .def_property_readonly(
            "cols", (int (StackSparseMatrix::*)() const)(&StackSparseMatrix::ncols))
        .def_property(
            "conjugacy", &StackSparseMatrix::conjugacy, &StackSparseMatrix::set_conjugacy)
        .def_readwrite("symm_scale", &StackSparseMatrix::symm_scale)
        .def("allowed", [](StackSparseMatrix *self, int i, int j) -> bool {
            return (bool) self->allowed(i, j);
        })
        .def("get_scaling", &StackSparseMatrix::get_scaling, py::arg("leftq"), py::arg("rightq"))
        .def("transpose", [](StackSparseMatrix *self) {
            return SpinAdapted::Transpose(*self);
        })
        .def("clear", &StackSparseMatrix::Clear)
        .def("deep_copy", &StackSparseMatrix::deepCopy)
        .def("shallow_copy", &StackSparseMatrix::shallowCopy)
        .def("deep_clear_copy", &StackSparseMatrix::deepClearCopy)
        .def("allocate", [](StackSparseMatrix *self, const vector<boost::shared_ptr<StateInfo>> &sts) {
            if (sts.size() == 1)
                self->allocate(*sts[0]);
            else if (sts.size() == 2)
                self->allocate(*sts[0], *sts[1]);
        })
        .def("deallocate", &StackSparseMatrix::deallocate);
    
    py::class_<StackTransposeview, boost::shared_ptr<StackTransposeview>,
               StackSparseMatrix>(m, "StackTransposeView")
        .def(py::init<StackSparseMatrix&>());

    py::class_<StackWavefunction, boost::shared_ptr<StackWavefunction>,
               StackSparseMatrix>(m, "Wavefunction")
        .def(py::init<>())
        .def_property("onedot", &StackWavefunction::get_onedot,
                      (void (StackWavefunction::*)(bool)) &
                          StackWavefunction::set_onedot)
        .def("initialize", (void (StackWavefunction::*)(
                               const vector<SpinQuantum> &, const StateInfo &,
                               const StateInfo &, const bool &)) &
                               StackWavefunction::initialise)
        .def("initialize_from", (void (StackWavefunction::*)(const StackWavefunction&))
             &StackWavefunction::initialise)
        .def("copy_data", &StackWavefunction::copyData)
        .def("save_wavefunction_info",
             &StackWavefunction::SaveWavefunctionInfo);

    py::class_<StackDensityMatrix, boost::shared_ptr<StackDensityMatrix>,
               StackSparseMatrix>(m, "DensityMatrix")
        .def(py::init<>());

    py::bind_vector<vector<StackWavefunction>>(m, "VectorWavefunction");

    py::class_<StackOp_component_base,
               boost::shared_ptr<StackOp_component_base>>(m,
                                                          "OperatorArrayBase");

    py::bind_vector<vector<boost::shared_ptr<StackOp_component_base>>>(
        m, "VectorOperatorArrayBase");
    py::bind_vector<vector<boost::shared_ptr<StackSparseMatrix>>>(
        m, "VectorStackSparseMatrix");

    pybind_stack_op_component<1, StackCre>(m, "Cre");
    pybind_stack_op_component<1, StackDes>(m, "Des");

    pybind_stack_op_component<2, StackCreDes>(m, "CreDes");
    pybind_stack_op_component<2, StackDesCre>(m, "DesCre");
    pybind_stack_op_component<2, StackCreCre>(m, "CreCre");
    pybind_stack_op_component<2, StackDesDes>(m, "DesDes");

    pybind_stack_op_component<2, StackCreDesComp>(m, "CreDesComp");
    pybind_stack_op_component<2, StackDesCreComp>(m, "DesCreComp");
    pybind_stack_op_component<2, StackDesDesComp>(m, "DesDesComp");
    pybind_stack_op_component<2, StackCreCreComp>(m, "CreCreComp");

    pybind_stack_op_component<1, StackCreCreDesComp>(m, "CreCreDesComp");
    pybind_stack_op_component<1, StackCreDesDesComp>(m, "CreDesDesComp");

    pybind_stack_op_component<0, StackHam>(m, "Hamiltonian");
    pybind_stack_op_component<0, StackOverlap>(m, "Overlap");

    m.def("multiply_with_own_transpose",
          &operatorfunctions::MultiplyWithOwnTranspose);
}
