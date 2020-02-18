
#include "rev/data_page.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace std;

void pybind_data_page(py::module &m) {
    
    m.def("init_data_pages", &block2::init_data_pages, py::arg("n_pages"));
    
    m.def("release_data_pages", &block2::release_data_pages);
    
    m.def("activate_data_page", &block2::activate_data_page, py::arg("ip"));
    
    m.def("get_data_page_pointer", &block2::get_data_page_pointer, py::arg("ip"));
    
    m.def("set_data_page_pointer", &block2::set_data_page_pointer, py::arg("ip"), py::arg("offset"));
    
    m.def("save_data_page", &block2::save_data_page, py::arg("ip"), py::arg("filename"));
    
    m.def("load_data_page", &block2::load_data_page, py::arg("ip"), py::arg("filename"));

}
