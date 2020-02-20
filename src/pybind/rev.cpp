
#include "rev/operator_functions.hpp"
#include "StackMatrix.h"
#include "StackOperators.h"
#include "enumerator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/iostream.h>
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
    
    m.def("product", &block2::Product, py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("state_info"), py::arg("scale") = 1.0);
    
    m.def("tensor_rotate", &block2::TensorRotate, py::arg("a"), py::arg("c"),
          py::arg("state_info"), py::arg("rotate_matrix"), py::arg("scale") = 1.0);
    
    m.def("tensor_trace_diagonal", &block2::TensorTraceDiagonal, py::arg("a"), py::arg("c"),
          py::arg("state_info"), py::arg("trace_right"), py::arg("scale") = 1.0);
    
    m.def("tensor_product_diagonal", &block2::TensorProductDiagonal, py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("state_info"), py::arg("scale") = 1.0);
    
    m.def("tensor_scale", &block2::TensorScale, py::arg("scale"), py::arg("a"));
    
    m.def("tensor_scale_add", (void (*)(double, const StackSparseMatrix &, StackSparseMatrix &,
        const StateInfo &)) &block2::TensorScaleAdd, py::arg("scale"),
          py::arg("a"), py::arg("c"), py::arg("state_info"));
    
    m.def("tensor_scale_add_no_trans", (void (*)(double, const StackSparseMatrix &, StackSparseMatrix &))
        &block2::TensorScaleAdd, py::arg("scale"), py::arg("a"), py::arg("c"));
        
    m.def("tensor_dot_product", &block2::TensorDotProduct, py::arg("a"), py::arg("b"));
    
    m.def("tensor_precondition", &block2::TensorPrecondition, py::arg("a"), py::arg("e"), py::arg("diag"));
    
    m.def("tensor_product_multiply", &block2::TensorProductMultiply, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("v"),
         py::arg("state_info"), py::arg("op_q"), py::arg("scale"));
    
    m.def("tensor_trace_multiply", &block2::TensorTraceMultiply, py::arg("a"), py::arg("c"), py::arg("v"),
         py::arg("state_info"), py::arg("trace_right"), py::arg("scale"));
//           py::call_guard<py::scoped_ostream_redirect,
//                      py::scoped_estream_redirect>());

}
