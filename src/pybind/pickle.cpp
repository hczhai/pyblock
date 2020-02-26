
#include "SpinQuantum.h"
#include "StackBaseOperator.h"
#include "StackMatrix.h"
#include "Stackwavefunction.h"
#include "enumerator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <string>

namespace py = pybind11;
using namespace std;
using namespace SpinAdapted;

template<typename T>
py::list t_pickle(const vector<T> &x) {
    py::list r(x.size());
    for (size_t i = 0; i < x.size(); i++)
        r[i] = x[i];
    return r;
}

template<typename T>
vector<T> v_unpickle(const py::list &x) {
    vector<T> r(x.size());
    for (size_t i = 0; i < x.size(); i++)
        r[i] = x[i].cast<T>();
    return r;
}

template<typename T>
py::list t_pickle(const vector<vector<T>> &x) {
    py::list r(x.size());
    for (size_t i = 0; i < x.size(); i++)
        r[i] = t_pickle(x[i]);
    return r;
}


template<typename T>
vector<vector<T>> vv_unpickle(const py::list &x) {
    vector<vector<T>> r(x.size());
    for (size_t i = 0; i < x.size(); i++)
        r[i] = v_unpickle<T>(x[i].cast<py::list>());
    return r;
}

template<typename K, typename T>
py::list t_pickle(const map<K, vector<T>> &x) {
    py::list r(x.size());
    int i = 0;
    for (auto &rr : x) {
        r[i] = py::make_tuple(rr.first, t_pickle(rr.second));
        i++;
    }
    return r;
}

template<typename K, typename T>
map<K, vector<T>> mv_unpickle(const py::list &x) {
    map<K, vector<T>> r;
    for (size_t i = 0; i < x.size(); i++)
        r[x[i].cast<py::tuple>()[0].cast<K>()] =
            v_unpickle<T>(x[i].cast<py::tuple>()[1].cast<py::list>());
    return r;
}

template<typename T>
py::list t_pickle(const map<pair<int, int>, T> &x) {
    py::list r(x.size());
    int i = 0;
    for (auto &rr : x) {
        r[i] = py::make_tuple(py::make_tuple(rr.first.first, rr.first.second), rr.second);
        i++;
    }
    return r;
}

template<typename T>
map<pair<int, int>, T> mp_unpickle(const py::list &x) {
    map<pair<int, int>, T> r;
    for (size_t i = 0; i < x.size(); i++) {
        auto f = x[i].cast<py::tuple>()[0].cast<py::tuple>();
        r[make_pair(f[0].cast<int>(), f[1].cast<int>())] =
            x[i].cast<py::tuple>()[1].cast<T>();
    }
    return r;
}

template<typename T>
py::list t_pickle(const vector<pair<pair<int, int>, T>> &x) {
    py::list r(x.size());
    for (size_t i = 0; i < x.size(); i++)
        r[i] = py::make_tuple(py::make_tuple(x[i].first.first, x[i].first.second), x[i].second);
    return r;
}


template<typename T>
vector<pair<pair<int, int>, T>> vpp_unpickle(const py::list &x) {
    vector<pair<pair<int, int>, T>> r(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        auto f = x[i].cast<py::tuple>()[0].cast<py::tuple>();
        r[i] = make_pair(make_pair(f[0].cast<int>(), f[1].cast<int>()), x[i].cast<py::tuple>()[1].cast<T>());
    }
    return r;
}

class PStackSparseMatrix : public StackSparseMatrix {
public:
    PStackSparseMatrix(py::tuple t) {
        this->totalMemory = t[0].cast<int>();
        this->data = (double *) t[1].cast<size_t>();
        this->conj = t[2].cast<char>();
        this->orbs = v_unpickle<int>(t[3].cast<py::list>());
        this->fermion = t[4].cast<bool>();
        this->allowedQuantaMatrix.nrs = t[5].cast<py::tuple>()[0].cast<int>();
        this->allowedQuantaMatrix.ncs = t[5].cast<py::tuple>()[1].cast<int>();
        this->allowedQuantaMatrix.rep = v_unpickle<char>(t[5].cast<py::tuple>()[2].cast<py::list>());
        this->initialised = t[6].cast<bool>();
        this->built = t[7].cast<bool>();
        this->built_on_disk = t[8].cast<bool>();
        this->deltaQuantum = v_unpickle<SpinQuantum>(t[9].cast<py::list>());
        this->Sign = t[10].cast<int>();
        this->build_pattern = t[11].cast<string>();
        this->quantum_ladder = mv_unpickle<string, SpinQuantum>(t[12].cast<py::list>());
        this->filename = t[13].cast<string>();
        this->rowCompressedForm = vv_unpickle<int>(t[14].cast<py::list>());
        this->colCompressedForm = vv_unpickle<int>(t[15].cast<py::list>());
        this->nonZeroBlocks = vpp_unpickle<StackMatrix>(t[16].cast<py::list>());
        this->mapToNonZeroBlocks = mp_unpickle<int>(t[17].cast<py::list>());
        this->symm_scale = t[18].cast<double>();
    }
    py::tuple pickle() const {
        return py::make_tuple(
            this->totalMemory,
            (size_t) this->data,
            this->conj,
            t_pickle(this->orbs),
            this->fermion,
            py::make_tuple(this->allowedQuantaMatrix.nrs,
                           this->allowedQuantaMatrix.ncs,
                           t_pickle(this->allowedQuantaMatrix.rep)),
            this->initialised,
            this->built,
            this->built_on_disk,
            t_pickle(this->deltaQuantum),
            this->Sign,
            this->build_pattern,
            t_pickle(this->quantum_ladder),
            this->filename,
            t_pickle<int>(this->rowCompressedForm),
            t_pickle<int>(this->colCompressedForm),
            t_pickle(this->nonZeroBlocks),
            t_pickle(this->mapToNonZeroBlocks),
            this->symm_scale
        );
    }
};

py::tuple pickle_stack_sparse_matrix(StackSparseMatrix *self) {
    return ((PStackSparseMatrix*)self)->pickle();
}

StackSparseMatrix unpickle_stack_sparse_matrix(py::tuple t) {
    return (StackSparseMatrix) PStackSparseMatrix(t);
}