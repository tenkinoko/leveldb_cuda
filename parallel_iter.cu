#include "parallel_iter.cuh"
#include "benchmark.h"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>


#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if(error != cudaSuccess)\
    {\
        printf("Error: %s:%d", __FILE__, __LINE__);\
        printf("Code:%d, reason: %s\n",error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct kv_pair {
    char key[129];
};

__device__ void strcpy_GPU(char* dest, const char* src, int str_len);
__device__ const char* GetVarint32PtrFallback_GPU(const char* p, const char* limit, unsigned int* value);
__device__ const char* GetVarint32Ptr_GPU(const char* p, const char* limit, unsigned int* value);
__device__ const char* DecodeEntry_GPU(const char* p, const char* limit,
    unsigned int* shared, unsigned int* non_shared,
    unsigned int* value_length);
__device__ unsigned int DecodeFixed32_GPU(const char* ptr);
__global__ void unpack_kernel(const char* data_, unsigned int restarts_,
    int num_restart, int restart_interval,
    kv_pair* output);




__device__ void strcpy_GPU(char* dest, const char* src, int str_len) {
    if (str_len == 0) return;
    while (str_len--) {
        *dest++ = *src++;
    }
}

__device__ const char* GetVarint32PtrFallback_GPU(const char* p, const char* limit, unsigned int* value) {
    unsigned int result = 0;
    for (unsigned int shift = 0; shift <= 28 && p < limit; shift += 7) {
        unsigned int byte = *(reinterpret_cast<const unsigned char*>(p));
        p++;
        if (byte & 128) {
            result |= ((byte & 127) << shift);
        }
        else {
            result |= (byte << shift);
            *value = result;
            return reinterpret_cast<const char*>(p);
        }
    }
    return nullptr;
}

__device__ const char* GetVarint32Ptr_GPU(const char* p, const char* limit, unsigned int* value) {
    if (p < limit) {
        unsigned int result = *(reinterpret_cast<const unsigned char*>(p));
        if ((result & 128) == 0) {
            *value = result;
            return p + 1;
        }
    }
    return GetVarint32PtrFallback_GPU(p, limit, value);
}

__device__ const char* DecodeEntry_GPU(const char* p, const char* limit,
    unsigned int* shared, unsigned int* non_shared,
    unsigned int* value_length) {
    if (limit - p < 3) {
        return nullptr;
    }
    *shared = reinterpret_cast<const unsigned char*>(p)[0];
    *non_shared = reinterpret_cast<const unsigned char*>(p)[1];
    *value_length = reinterpret_cast<const unsigned char*>(p)[2];
    if ((*shared | *non_shared | *value_length) < 128) {
        p += 3;
    }
    else {
        if ((p = GetVarint32Ptr_GPU(p, limit, shared)) == nullptr) return nullptr;
        if ((p = GetVarint32Ptr_GPU(p, limit, non_shared)) == nullptr) return nullptr;
        if ((p = GetVarint32Ptr_GPU(p, limit, value_length)) == nullptr) return nullptr;
    }
    if (static_cast<unsigned int>(limit - p) < (*non_shared + *value_length)) {
        return nullptr;
    }
    return p;
}

__device__ unsigned int DecodeFixed32_GPU(const char* ptr) {
    const unsigned char* const buffer = reinterpret_cast<const unsigned char*>(ptr);
    return (static_cast<unsigned int>(buffer[0])) |
        (static_cast<unsigned int>(buffer[1]) << 8) |
        (static_cast<unsigned int>(buffer[2]) << 16) |
        (static_cast<unsigned int>(buffer[3]) << 24);

}

__global__ void unpack_kernel(const char* data_, unsigned int restarts_,
    int num_restart, int restart_interval, kv_pair* output,
    unsigned int* d_value_offset, unsigned int* d_value_length) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_restart) return;
    //Get restart point 
    unsigned int restartpoint = 0;
    unsigned int limit = 0;
    restartpoint = DecodeFixed32_GPU(data_ + restarts_ + tid * sizeof(unsigned int));
    limit = DecodeFixed32_GPU(data_ + restarts_ + (tid + 1) * sizeof(unsigned int));
#ifdef DEBUG
    printf("Tid: %d, restartpoint is: %d \n", tid, restartpoint);
    printf("Tid: %d, Limit is: %d \n", tid, limit);
#endif
    const char* start_ptr = data_ + restartpoint;
    const char* limit_ptr = data_ + limit;
    //
    char shared_key[16];
    char recycle_key[16];
    const char* p = start_ptr;
    int i = 0;
    while (p < limit_ptr && i < restart_interval) {
        unsigned int shared, non_shared, value_length;
        p = DecodeEntry_GPU(/*param@ p */ p,
            /*param@ limit */ limit_ptr,
            /*param@ shared */ &shared,
            /*param@ non_shared */ &non_shared,
            /*param@ value_length */ &value_length);
#ifdef DEBUG
        printf("Tid: %d, shared is: %d \n", tid, shared);
        printf("Tid: %d, non_shared is: %d \n", tid, non_shared);
        printf("Tid: %d, value_length is: %d \n", tid, value_length);
#endif
        if (shared == 0) {
            strcpy_GPU(shared_key, p, non_shared);
            shared_key[non_shared] = '\0';
            strcpy_GPU(recycle_key, shared_key, non_shared + 1);
        }
        else {
            strcpy_GPU(recycle_key, shared_key, shared);
            strcpy_GPU(recycle_key + shared, p, non_shared);
            recycle_key[shared + non_shared] = '\0';
            strcpy_GPU(shared_key, recycle_key, shared + non_shared + 1);
        }
        strcpy_GPU(output[i + tid * restart_interval].key, recycle_key, shared + non_shared + 1);
        p += non_shared;
        d_value_offset[i + tid * restart_interval] = (p - data_);
        d_value_length[i + tid * restart_interval] = value_length;
        p += value_length;
        i++;
    }
}

void unpack_GPU(const char* data_, unsigned int data_size,
    unsigned int restarts_, int num_restart,
    int restart_interval, int num_kv,
    std::vector<std::string>& key_vector,
    unsigned int* h_value_offset,
    unsigned int* h_value_length) {
    char* d_data_;
    kv_pair* kv_pair_d;
    kv_pair* kv_pair_h;
    unsigned int* d_value_offset;
    unsigned int* d_value_length;

    kv_pair_h = (kv_pair*)malloc(sizeof(kv_pair) * num_kv);
    CHECK(cudaMalloc((char**)&d_data_, data_size));
    CHECK(cudaMalloc((kv_pair**)&kv_pair_d, sizeof(kv_pair) * num_kv));
    CHECK(cudaMalloc((unsigned int**)&d_value_offset, sizeof(unsigned int) * num_kv));
    CHECK(cudaMalloc((unsigned int**)&d_value_length, sizeof(unsigned int) * num_kv));

    //data transfer from host to device
    CHECK(cudaMemcpy(d_data_, data_, data_size, cudaMemcpyHostToDevice));
    dim3 block(256, 1);
    dim3 grid((num_restart + block.x - 1) / block.x, 1);

    //LARGE_INTEGER freq, head, tail;
    //prepare(freq);
    //timer(head);
    unpack_kernel << <grid, block >> > (/*const char* data_*/ d_data_,
        /* unsigned int restarts_ */ restarts_,
        /* int num_restart */ num_restart,
        /* int restart_interval */ restart_interval,
        /* kv_pair* */ kv_pair_d,
        /* unsigned int* */ d_value_offset,
        /* unsigned int* */ d_value_length);
    CHECK(cudaDeviceSynchronize());
    //timer(tail);
    //output("Kernel: ", total(head, tail));

    // data transfer from device to host
    CHECK(cudaMemcpy(kv_pair_h, kv_pair_d, sizeof(kv_pair) * num_kv, cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_kv; ++i) {
        key_vector.emplace_back(kv_pair_h[i].key);
    }
    CHECK(cudaMemcpy(h_value_offset, d_value_offset, sizeof(unsigned int) * num_kv, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_value_length, d_value_length, sizeof(unsigned int) * num_kv, cudaMemcpyDeviceToHost));

    free(kv_pair_h);
    cudaFree(d_data_);
    cudaFree(kv_pair_d);
    cudaFree(d_value_offset);
    cudaFree(d_value_length);
    return;
}


