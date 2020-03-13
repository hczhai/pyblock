
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#ifdef _HAS_INTEL_MKL
#include <mkl.h>
#endif
#include "global.h"
#include "alloc.h"
#include "data_page.hpp"

using namespace std;
using namespace SpinAdapted;

namespace block2 {

int main_page = 0;
StackAllocator<double> *current_page;
vector<StackAllocator<double>> DataPages;

void init_data_pages(int n_pages) {
    dmrginp.matmultFlops.resize(max(numthrds, dmrginp.quanta_thrds()), 0.);
    dmrginp.initCumulTimer();

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(dmrginp.mkl_thrds());
    mkl_set_dynamic(0);
#endif

    if (dmrginp.outputlevel() >= 0)
        cout << "allocating " << dmrginp.getMemory() << " doubles (in " <<
            n_pages << " data pages" << ")" << endl;
    
    size_t n_total = dmrginp.getMemory();
    size_t n_vari = n_total / 200;
    size_t n_base = n_total / (n_pages + 1) / 262144 * 262144;
    size_t n_acc = 0;
    double *ptr = new double[n_total];
    DataPages.resize(n_pages);
    
    for (int i = 1; i < n_pages; i++)
        DataPages[i].size = n_base, n_acc += n_base;
    
    DataPages[0].size = (n_total - n_acc) / 262144 * 262144;
    
    for (int i = 0; i < n_pages; i++,ptr += DataPages[i].size) {
        if (dmrginp.outputlevel() >= 0)
            cout << "page " << i << " allocated " << DataPages[i].size << " doubles" << endl;
        DataPages[i].memused = 0;
        DataPages[i].data = ptr;
    }
    
    activate_data_page(0);
}

void release_data_pages() {
    delete[] DataPages[0].data;
    DataPages.resize(0);
}

void activate_data_page(int ip) {
    main_page = ip;
    current_page = &DataPages[ip];
}

size_t get_data_page_pointer(int ip) {
    return DataPages[ip].memused;
}

void set_data_page_pointer(int ip, size_t offset) {
    DataPages[ip].memused = offset;
}

void save_data_page(int ip, const string& filename) {
    ofstream ofs(filename.c_str(), ios::binary);
    ofs.write((char*)&DataPages[ip].memused, sizeof(DataPages[ip].memused));
    ofs.write((char*)DataPages[ip].data, sizeof(double) * DataPages[ip].memused);
    ofs.close();
}

void load_data_page(int ip, const string& filename) {
    ifstream ifs(filename.c_str(), ios::binary);
    ifs.read((char*)&DataPages[ip].memused, sizeof(DataPages[ip].memused));
    ifs.read((char*)DataPages[ip].data, sizeof(double) * DataPages[ip].memused);
    ifs.close();
}
    
} // namespace block2