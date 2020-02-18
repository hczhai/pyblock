
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;

void pybind_io(py::module &m);
void pybind_block(py::module &m);
void pybind_symmetry(py::module &m);
void pybind_dmrg(py::module &m);
void pybind_operator(py::module &m);
void pybind_matrix(py::module &m);
void pybind_rev(py::module &m);
void pybind_data_page(py::module &m);

PYBIND11_MAKE_OPAQUE(vector<int>);
PYBIND11_MAKE_OPAQUE(vector<bool>);
PYBIND11_MAKE_OPAQUE(vector<double>);
PYBIND11_MAKE_OPAQUE(vector<vector<int>>);

PYBIND11_MODULE(block, m) {

    m.doc() = "Python3 wrapper for block 1.5.3 (spin adapted).";

    py::bind_vector<vector<int>>(m, "VectorInt");
    py::bind_vector<vector<bool>>(m, "VectorBool");
    py::bind_vector<vector<vector<int>>>(m, "VectorVectorInt");
    py::bind_vector<vector<double>>(m, "VectorDouble");

    pybind_matrix(m);

    py::module m_symm = m.def_submodule(
        "symmetry", "Classes for handling symmetries and quantum numbers.");
    pybind_symmetry(m_symm);

    py::module m_oper = m.def_submodule(
        "operator", "Classes for operator matrices and operations.");
    pybind_operator(m_oper);

    py::module m_io =
        m.def_submodule("io", "Contains Input/Output related interfaces.");
    pybind_io(m_io);

    py::module m_block =
        m.def_submodule("block", "Block definition and operator operations.");
    pybind_block(m_block);

    py::module m_dmrg = m.def_submodule("dmrg", "DMRG calculations.");
    pybind_dmrg(m_dmrg);
    
    py::module m_rev = m.def_submodule("rev", "Revised Block functions.");
    pybind_rev(m_rev);
    
    py::module m_data_page = m.def_submodule("data_page", "Revised data page functions.");
    pybind_data_page(m_data_page);
}
