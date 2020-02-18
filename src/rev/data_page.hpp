
#ifndef DATA_PAGE_HPP_
#define DATA_PAGE_HPP_

#include <string>

using namespace std;

namespace block2 {

void init_data_pages(int n_pages);

void release_data_pages();

void activate_data_page(int ip);

size_t get_data_page_pointer(int ip);

void set_data_page_pointer(int ip, size_t offset);

void save_data_page(int ip, const string& filename);

void load_data_page(int ip, const string& filename);

} // namespace block2

#endif /* REV_OPERATOR_FUNCTIONS_H_ */