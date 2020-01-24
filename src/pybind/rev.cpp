
#include "rev/operator_functions.hpp"
#include "StackMatrix.h"
#include "StackOperators.h"
#include "enumerator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <string>

namespace py = pybind11;
using namespace std;
using namespace SpinAdapted;

PYBIND11_MAKE_OPAQUE(vector<boost::shared_ptr<StateInfo>>);
PYBIND11_MAKE_OPAQUE(vector<::Matrix>);

void pybind_rev(py::module &m) {
    
    m.def("tensor_trace", &block2::TensorTrace, py::arg("a"), py::arg("c"),
          py::arg("state_info"), py::arg("trace_right"), py::arg("scale") = 1.0);
    
    m.def("tensor_product", &block2::TensorProduct, py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("state_info"), py::arg("scale") = 1.0);
    
    m.def("tensor_rotate", &block2::TensorRotate, py::arg("a"), py::arg("c"),
          py::arg("state_info"), py::arg("rotate_matrix"));

}
