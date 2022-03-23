#include <vector>
#include <string>

void unpack_GPU(const char* data_, 
    unsigned int data_size,
    unsigned int restarts_, 
    int num_restart,
    int restart_interval, 
    int num_kv,
    std::vector<std::string>& key_vector,
    unsigned int* h_value_offset,
    unsigned int* h_value_length);