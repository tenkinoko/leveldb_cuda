// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef STORAGE_LEVELDB_TABLE_FORMAT_H_
#define STORAGE_LEVELDB_TABLE_FORMAT_H_

#include <cstdint>
#include <string>

#include "slice.h"
#include "status.h"

namespace leveldb {

    struct BlockContents {
        BlockContents() = default;
        BlockContents(Slice& data_, bool is_cachable, bool is_heap) :
            data(data_),
            cachable(is_cachable),
            heap_allocated(is_heap) {}
        Slice data;           // Actual contents of data
        bool cachable;        // True iff data can be cached
        bool heap_allocated;  // True iff caller should delete[] data.data()
    };



}  // namespace leveldb

#endif  // STORAGE_LEVELDB_TABLE_FORMAT_H_
