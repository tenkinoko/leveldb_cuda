#include <vector>
#include <string>

//void unpack_GPU(const char* data_, 
//    unsigned int data_size,
//    unsigned int restarts_, 
//    int num_restart,
//    int restart_interval, 
//    int num_kv,
//    std::vector<std::string>& key_vector,
//    unsigned int* h_value_offset,
//    unsigned int* h_value_length);

void unpack_LRC_GPU(
    const char* data_,
    uint32_t data_size,
    uint32_t KV_size_,
    int64_t slope,
    int64_t intercept,
    int64_t min_error,
    size_t* key);