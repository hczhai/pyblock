
#include "StackBaseOperator.h"
#include "StackMatrix.h"
#include "StackOperators.h"
#include "enumerator.h"
#include "rotationmat.h"
#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <string>

namespace py = pybind11;
using namespace std;
using namespace Eigen;
using namespace SpinAdapted;

PYBIND11_MAKE_OPAQUE(vector<::Matrix>);

void pybind_matrix(py::module &m) {

    py::class_<::Matrix>(m, "Matrix", "NEWMAT10 matrix.")
        .def(py::init<>())
        .def(py::init(
            [](Ref<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>> mat) {
                ::Matrix smat(mat.rows(), mat.cols());
                memcpy(smat.data(), mat.data(),
                       sizeof(double) * mat.rows() * mat.cols());
                return smat;
            }))
        .def_property(
            "ref",
            [](::Matrix *self) {
                return Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(
                    self->data(), self->Nrows(), self->Ncols());
            },
            [](::Matrix *self,
               Ref<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>> mat) {
                   memcpy(self->data(), mat.data(),
                       sizeof(double) * mat.rows() * mat.cols());
            },
            "A numpy.ndarray reference.")
        .def_property_readonly(
            "rows", (int (::Matrix::*)() const)(&::Matrix::Nrows))
        .def_property_readonly(
            "cols", (int (::Matrix::*)() const)(&::Matrix::Ncols))
        .def("__repr__", [](::Matrix *self) {
            stringstream ss;
            ss << Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(
                self->data(), self->Nrows(), self->Ncols());
            return ss.str();
        });
    
    py::bind_vector<vector<::Matrix>>(m, "VectorMatrix");

    m.def("load_rotation_matrix", &LoadRotationMatrix, "Load rotation matrix.");

    m.def("save_rotation_matrix", &SaveRotationMatrix, "Save rotation matrix.");

    py::class_<::DiagonalMatrix>(m, "DiagonalMatrix",
                                 "NEWMAT10 diagonal matrix.")
        .def(py::init<>())
        .def(py::init(
            [](Ref<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>> mat) {
                ::DiagonalMatrix smat(mat.rows() * mat.cols());
                memcpy(smat.data(), mat.data(),
                       sizeof(double) * mat.rows() * mat.cols());
                return smat;
            }))
        .def_property(
            "ref",
            [](::DiagonalMatrix *self) {
                return Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(
                    self->data(), 1, self->Ncols());
            },
            [](::DiagonalMatrix *self,
               Ref<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>> mat) {
                   memcpy(self->data(), mat.data(),
                       sizeof(double) * mat.rows() * mat.cols());
            },
            "A numpy.ndarray reference.")
        .def_property_readonly(
            "rows", (int (::DiagonalMatrix::*)() const)(&::DiagonalMatrix::Nrows))
        .def_property_readonly(
            "cols", (int (::DiagonalMatrix::*)() const)(&::DiagonalMatrix::Ncols))
        .def("__add__", [](::DiagonalMatrix *self, ::DiagonalMatrix *other) -> ::DiagonalMatrix {
            return (*self) + (*other);
        })
        .def("resize", (void (::DiagonalMatrix::*)(int))&::DiagonalMatrix::ReSize, py::arg("nr"))
        .def("__repr__", [](::DiagonalMatrix *self) {
            stringstream ss;
            ss << Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(
                self->data(), 1, self->Ncols());
            return ss.str();
        });
}
