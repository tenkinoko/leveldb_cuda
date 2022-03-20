// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef STORAGE_LEVELDB_TABLE_BLOCK_H_
#define STORAGE_LEVELDB_TABLE_BLOCK_H_

#include <cstddef>
#include <cstdint>

#include "iterator.h"
#include "parallel_iterator.h"
namespace leveldb {

	struct BlockContents;
	class Comparator;

	class Block {
	public:
		// Initialize the block with the specified contents.
		explicit Block(const BlockContents& contents);

		Block(const Block&) = delete;
		Block& operator=(const Block&) = delete;

		~Block();

		size_t size() const { return size_; }
		Iterator* NewIterator(const Comparator* comparator);
		Parallel_Iterator* NewParallel_Iterator(const Comparator* comparator);
	private:
		class Iter; //Êý¾Ý¿éµü´úÆ÷
		class Parallel_Iter;
		uint32_t NumRestarts() const;
		uint32_t Num_KV() const;
		uint32_t Restart_interval() const;
		const char* data_;
		size_t size_;
		uint32_t restart_offset_;  // Offset in data_ of restart array
		// uint32_t num_kv_;
		// uint32_t restart_interval_;
		bool owned_;               // Block owns data_[]
	};

}  // namespace leveldb

#endif  // STORAGE_LEVELDB_TABLE_BLOCK_H_
