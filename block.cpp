// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// Decodes the blocks generated by block_builder.cc.

#include "block.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <string>
#include "comparator.h"
#include "format.h"
#include "coding.h"
#include "logging.h"
#include "parallel_iter.cuh"
#include <iostream>


namespace leveldb {

    //inline uint32_t Block::NumRestarts() const {
    //    assert(size_ >= sizeof(uint32_t));
    //    return DecodeFixed32(data_ + size_ - sizeof(uint32_t));
    //}

    //inline uint32_t Block::Num_KV() const {
    //    return DecodeFixed32(data_ + size_ - 2 * sizeof(uint32_t));
    //}

    //inline uint32_t Block::Restart_interval() const {
    //    return DecodeFixed32(data_ + size_ - 3 * sizeof(uint32_t));
    //}

    inline size_t Block::Num_slope() const {
        return DecodeFixed64(data_);
    }

    inline size_t Block::Num_intercept() const {
        return DecodeFixed64(data_ + sizeof(size_t));
    }

    inline size_t Block::Num_min_error() const {
        return DecodeFixed64(data_ + 2 * sizeof(size_t));
    }

    inline uint32_t Block::Num_KV() const {
        return DecodeFixed32(data_ + size_ - sizeof(uint32_t));
    }



    Block::Block(const BlockContents& contents)
        : data_(contents.data.data()),
        size_(contents.data.size()),
        owned_(contents.heap_allocated) {
        slope_ = Num_slope();
        intercept_ = Num_intercept();
        min_error_ = Num_min_error();
        KV_size_ = Num_KV();

        //if (size_ < sizeof(uint32_t)) {
        //    size_ = 0;  // Error marker
        //}
        //else {
        //    size_t max_restarts_allowed = (size_ - sizeof(uint32_t)) / sizeof(uint32_t);
        //    if (NumRestarts() > max_restarts_allowed) {
        //        // The size is too small for NumRestarts()
        //        size_ = 0;
        //    }
        //    else {
        //        //last_bound restart_interval num_kv num_restart
        //        restart_offset_ = size_ - (4 + NumRestarts()) * sizeof(uint32_t); // why 1+numrestarts ? 重启点数组+数组个数
        //        //std::cout << "restart_offset from unpack" <<" "<< restart_offset_ << std::endl;
        //    }
        //}
    }

    Block::~Block() {
        if (owned_) {
            delete[] data_;
        }
    }

    // Helper routine: decode the next block entry starting at "p",
    // storing the number of shared key bytes, non_shared key bytes,
    // and the length of the value in "*shared", "*non_shared", and
    // "*value_length", respectively.  Will not dereference past "limit".
    //
    // If any errors are detected, returns nullptr.  Otherwise, returns a
    // pointer to the key delta (just past the three decoded values).
    static inline const char* DecodeEntry(const char* p, const char* limit,
        uint32_t* shared, uint32_t* non_shared,
        uint32_t* value_length) {
        if (limit - p < 3) return nullptr;
        *shared = reinterpret_cast<const uint8_t*>(p)[0];
        *non_shared = reinterpret_cast<const uint8_t*>(p)[1];
        *value_length = reinterpret_cast<const uint8_t*>(p)[2];
        if ((*shared | *non_shared | *value_length) < 128) {
            // Fast path: all three values are encoded in one byte each
            p += 3;
        }
        else {
            if ((p = GetVarint32Ptr(p, limit, shared)) == nullptr) return nullptr;
            if ((p = GetVarint32Ptr(p, limit, non_shared)) == nullptr) return nullptr;
            if ((p = GetVarint32Ptr(p, limit, value_length)) == nullptr) return nullptr;
        }

        if (static_cast<uint32_t>(limit - p) < (*non_shared + *value_length)) {
            return nullptr;
        }
        return p;
    }

    class Block::LRC_Iter : public Iterator {
    private:
        const Comparator* const comparator_;
        const char* data_;
        uint32_t KV_size_;
        uint32_t current_;
        //std::string key_;
        //size_t value_;

        size_t slope_;
        size_t intercept_;
        size_t min_error_;

        size_t Num_non_neg_error() const;
        size_t Num_value() const;

    public:
        LRC_Iter(const Comparator* comparator, const char* data, uint32_t KV_size,
            size_t slope, size_t intercept, size_t min_error)
            : comparator_(comparator),
            current_(0),
            data_(data),
            KV_size_(KV_size),
            slope_(slope),
            intercept_(intercept),
            min_error_(min_error) {}

        Slice key() const override {
            size_t key_int = slope_ * current_ + intercept_ + min_error_ + Num_non_neg_error();
            return Slice(std::to_string(key_int));
        }

        Slice value() const override{
            return Slice(std::to_string(Num_value()));
        }

        void Next() override {
            current_ += 1;
        }

        void Prev() override {
            current_ -= 1;
        }       
        

        void SeekToFirst() override {
            current_ = -1;
        }

        void SeekToLast() override {

        }

        bool Valid() const override {
            return current_ < KV_size_;
        }

        void Seek(const Slice& target) override {

        }

        Status status() const override { return Status::OK(); }
    };

    Iterator* Block::NewLRCIterator(const Comparator* comparator) {
        /*if (size_ < sizeof(uint32_t)) {
            return NewErrorIterator(Status::Corruption("bad block contents"));
        }
        const uint32_t num_restarts = NumRestarts();*/
        
        return new LRC_Iter(comparator, data_, KV_size_, slope_, intercept_, min_error_);
    }

    inline size_t Block::LRC_Iter::Num_non_neg_error() const {
        return DecodeFixed32(data_ + 3 * sizeof(size_t) + current_ * sizeof(uint32_t));
    }

    inline size_t Block::LRC_Iter::Num_value() const {
        return DecodeFixed32(data_ + 3 * sizeof(size_t) + KV_size_ * sizeof(uint32_t) + current_ * sizeof(uint32_t));
    }

    //class Block::Parallel_Iter : public Parallel_Iterator {
    //private:
    //    const Comparator* const comparator_;
    //    const char* const data_;
    //    uint32_t data_size_;
    //    uint32_t const restarts_;
    //    uint32_t const num_restarts_;
    //    uint32_t const restart_interval_;
    //    uint32_t const num_kv_;
    //    std::vector<std::string> key_vector;
    //    uint32_t* value_offset;
    //    uint32_t* value_length;
    //    std::string key_;
    //    Slice value_;
    //    uint32_t ptr;
    //    Status status_;
    //    inline int Compare(const Slice& a, const Slice& b) {
    //        return comparator_->Compare(a, b);
    //    }
    //public:
    //    Parallel_Iter(const Comparator* comparator, const char* data, uint32_t data_size, uint32_t restarts,
    //        uint32_t num_restarts, uint32_t restart_interval, uint32_t num_kv)
    //        : comparator_(comparator),
    //        data_(data),
    //        data_size_(data_size),
    //        restarts_(restarts),
    //        num_restarts_(num_restarts),
    //        restart_interval_(restart_interval),
    //        num_kv_(num_kv),
    //        ptr(0) {
    //        value_offset = new uint32_t[num_kv_];
    //        value_length = new uint32_t[num_kv_];
    //        unpack_parallel();
    //    }

    //    ~Parallel_Iter() {
    //        delete[] value_offset;
    //        delete[] value_length;
    //    }

    //    bool Valid() const override {
    //        return (ptr < num_kv_) && (ptr >= 0);
    //    }
    //    Status status() const override { return status_; }
    //    Slice key() const override {
    //        assert(Valid());
    //        return key_;
    //    }

    //    Slice value() const override {
    //        assert(Valid());
    //        return value_;
    //    }
    //    void Next() override {
    //        ptr++;
    //        if (Valid()) {
    //            key_ = key_vector[ptr];
    //            value_ = Slice(data_ + value_offset[ptr], value_length[ptr]);
    //        }
    //    }

    //    void Prev() override {
    //        ptr--;
    //        if (Valid()) {
    //            key_ = key_vector[ptr];
    //            value_ = Slice(data_ + value_offset[ptr], value_length[ptr]);
    //        }
    //    }
    //    // TODO: not to be finish
    //    void Seek(const Slice& target) override {
    //        uint32_t left = 0;
    //        uint32_t right = num_kv_ - 1;
    //        int current_key_compare = 0;
    //        if (Valid()) {
    //            current_key_compare = Compare(key_, target);
    //            if (current_key_compare < 0) {
    //                // key_ is smaller than target
    //                left = ptr;
    //            }
    //            else if (current_key_compare > 0) {
    //                right = ptr;
    //            }
    //            else {
    //                // current key_ is equal to target
    //                return;
    //            }
    //        }
    //        //binary seek
    //        while (left < right) {
    //            uint32_t mid = (left + right) / 2;
    //            Slice mid_key(key_vector[mid]);
    //            if (Compare(mid_key, target) < 0) {
    //                left = mid + 1;
    //            }
    //            else {
    //                right = mid;
    //            }
    //        }
    //        ptr = left;
    //    }

    //    void SeekToFirst() override {
    //        ptr = 0;
    //        key_ = key_vector[ptr];
    //        value_ = Slice(data_ + value_offset[ptr], value_length[ptr]);
    //    }

    //    void SeekToLast() override {
    //        ptr = num_kv_ - 1;
    //        key_ = key_vector[ptr];
    //        value_ = Slice(data_ + value_offset[ptr], value_length[ptr]);
    //    }

    //    void unpack_parallel() {
    //        unpack_GPU(/*const char* data_*/ data_,
    //            /*unsigned int*/ data_size_,
    //            /*unsigned int restarts_*/ restarts_,
    //            /*int num_restart*/ num_restarts_,
    //            /*int restart_interval*/ restart_interval_,
    //            /*int num_kv*/ num_kv_,
    //            /*std:vector<std::string>*/ key_vector,
    //            /*uint32_t* */ value_offset,
    //            /*uint32_t */ value_length);
    //    }
    //};

    // 数据块迭代器
    //class Block::Iter : public Iterator {
    //private:
    //    const Comparator* const comparator_;
    //    const char* const data_;       // underlying block contents  指向数据块的指针
    //    uint32_t const restarts_;      // Offset of restart array (list of fixed32) 重启点数组偏移
    //    uint32_t const num_restarts_;  // Number of uint32_t entries in restart array 重启点的数量

    //    // current_ is offset in data_ of current entry.  >= restarts_ if !Valid
    //    uint32_t current_;
    //    uint32_t restart_index_;  // Index of restart block in which current_ falls
    //    std::string key_;
    //    Slice value_;
    //    Status status_;

    //    inline int Compare(const Slice& a, const Slice& b) const {
    //        return comparator_->Compare(a, b);
    //    }

    //    // Return the offset in data_ just past the end of the current entry.
    //    inline uint32_t NextEntryOffset() const {
    //        return (value_.data() + value_.size()) - data_;
    //    }

    //    // 定位到第 index 个重启点，即得到第 index 组重启点键值对的偏移
    //    uint32_t GetRestartPoint(uint32_t index) {
    //        assert(index < num_restarts_);
    //        return DecodeFixed32(data_ + restarts_ + index * sizeof(uint32_t));
    //    }

    //    void SeekToRestartPoint(uint32_t index) {
    //        key_.clear();
    //        restart_index_ = index;
    //        // current_ will be fixed by ParseNextKey();

    //        // ParseNextKey() starts at the end of value_, so set value_ accordingly
    //        uint32_t offset = GetRestartPoint(index);
    //        value_ = Slice(data_ + offset, 0);
    //    }

    //public:
    //    Iter(const Comparator* comparator, const char* data, uint32_t restarts,
    //        uint32_t num_restarts)
    //        : comparator_(comparator),
    //        data_(data),
    //        restarts_(restarts),
    //        num_restarts_(num_restarts),
    //        current_(restarts_),
    //        restart_index_(num_restarts_) {
    //        assert(num_restarts_ > 0);
    //    }

    //    bool Valid() const override { return current_ < restarts_; } // 指针移动的位置(current_) 一定要小于 restarts_
    //    Status status() const override { return status_; }
    //    Slice key() const override {
    //        assert(Valid());
    //        return key_;
    //    }
    //    Slice value() const override {
    //        assert(Valid());
    //        return value_;
    //    }

    //    void Next() override {
    //        assert(Valid());
    //        ParseNextKey();
    //    }

    //    void Prev() override {
    //        assert(Valid());

    //        // Scan backwards to a restart point before current_
    //        const uint32_t original = current_;
    //        while (GetRestartPoint(restart_index_) >= original) {
    //            if (restart_index_ == 0) {
    //                // No more entries
    //                current_ = restarts_;
    //                restart_index_ = num_restarts_;
    //                return;
    //            }
    //            restart_index_--;
    //        }

    //        SeekToRestartPoint(restart_index_);
    //        do {
    //            // Loop until end of current entry hits the start of original entry
    //        } while (ParseNextKey() && NextEntryOffset() < original);
    //    }

    //    void Seek(const Slice& target) override {
    //        // Binary search in restart array to find the last restart point
    //        // with a key < target
    //        uint32_t left = 0;
    //        uint32_t right = num_restarts_ - 1;
    //        int current_key_compare = 0;

    //        if (Valid()) {
    //            // If we're already scanning, use the current position as a starting
    //            // point. This is beneficial if the key we're seeking to is ahead of the
    //            // current position.
    //            current_key_compare = Compare(key_, target);
    //            if (current_key_compare < 0) {
    //                // key_ is smaller than target
    //                left = restart_index_;
    //            }
    //            else if (current_key_compare > 0) {
    //                right = restart_index_;
    //            }
    //            else {
    //                // We're seeking to the key we're already at.
    //                return;
    //            }
    //        }

    //        while (left < right) {
    //            uint32_t mid = (left + right + 1) / 2;
    //            uint32_t region_offset = GetRestartPoint(mid);
    //            uint32_t shared, non_shared, value_length;
    //            const char* key_ptr =
    //                DecodeEntry(data_ + region_offset, data_ + restarts_, &shared,
    //                    &non_shared, &value_length);
    //            if (key_ptr == nullptr || (shared != 0)) {
    //                CorruptionError();
    //                return;
    //            }
    //            Slice mid_key(key_ptr, non_shared);
    //            if (Compare(mid_key, target) < 0) {
    //                // Key at "mid" is smaller than "target".  Therefore all
    //                // blocks before "mid" are uninteresting.
    //                left = mid;
    //            }
    //            else {
    //                // Key at "mid" is >= "target".  Therefore all blocks at or
    //                // after "mid" are uninteresting.
    //                right = mid - 1;
    //            }
    //        }

    //        // We might be able to use our current position within the restart block.
    //        // This is true if we determined the key we desire is in the current block
    //        // and is after than the current key.
    //        assert(current_key_compare == 0 || Valid());
    //        bool skip_seek = left == restart_index_ && current_key_compare < 0;
    //        if (!skip_seek) {
    //            SeekToRestartPoint(left);
    //        }
    //        // Linear search (within restart block) for first key >= target
    //        while (true) {
    //            if (!ParseNextKey()) {
    //                return;
    //            }
    //            if (Compare(key_, target) >= 0) {
    //                return;
    //            }
    //        }
    //    }

    //    void SeekToFirst() override {
    //        SeekToRestartPoint(0);
    //        ParseNextKey();
    //    }

    //    void SeekToLast() override {
    //        SeekToRestartPoint(num_restarts_ - 1);
    //        while (ParseNextKey() && NextEntryOffset() < restarts_) {
    //            // Keep skipping
    //        }
    //    }

    //private:
    //    void CorruptionError() {
    //        current_ = restarts_;
    //        restart_index_ = num_restarts_;
    //        status_ = Status::Corruption("bad entry in block");
    //        key_.clear();
    //        value_.clear();
    //    }

    //    bool ParseNextKey() {
    //        current_ = NextEntryOffset();
    //        const char* p = data_ + current_;
    //        const char* limit = data_ + restarts_;  // Restarts come right after data
    //        if (p >= limit) {
    //            // No more entries to return.  Mark as invalid.
    //            current_ = restarts_;
    //            restart_index_ = num_restarts_;
    //            return false;
    //        }

    //        // Decode next entry
    //        uint32_t shared, non_shared, value_length;
    //        p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
    //        if (p == nullptr || key_.size() < shared) {
    //            CorruptionError();
    //            return false;
    //        }
    //        else {
    //            key_.resize(shared);
    //            key_.append(p, non_shared);
    //            value_ = Slice(p + non_shared, value_length);
    //            while (restart_index_ + 1 < num_restarts_ &&
    //                GetRestartPoint(restart_index_ + 1) < current_) {
    //                ++restart_index_;
    //            }
    //            return true;
    //        }
    //    }
    //};

    //Parallel_Iterator* Block::NewParallel_Iterator(const Comparator* comparator) {
    //    const uint32_t num_restarts = NumRestarts();
    //    const uint32_t restart_interval_ = Restart_interval();
    //    const uint32_t num_kv_ = Num_KV();
    //    return new Parallel_Iter(comparator, data_, size_, restart_offset_, num_restarts, restart_interval_, num_kv_);
    //}

    //Iterator* Block::NewIterator(const Comparator* comparator) {
    //    if (size_ < sizeof(uint32_t)) {
    //        return NewErrorIterator(Status::Corruption("bad block contents"));
    //    }
    //    /*const uint32_t num_restarts = NumRestarts();*/
    //    if (size_ == 0) {
    //        return NewEmptyIterator();
    //    }
    //    else {
    //        return new Iter(comparator, data_, restart_offset_, num_restarts);
    //    }
    //}

}  // namespace leveldb
