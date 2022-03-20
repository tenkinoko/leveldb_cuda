#include <iostream>
#include "block_builder.h"
#include "slice.h"
#include "comparator.h"
#include "options.h"
#include "format.h"
#include "block.h"
#include <string>

int main() {
    //parrllel unpack using GPU
    {
        leveldb::Options* options = new leveldb::Options();
        options->block_restart_interval = 20;
        options->comparator = leveldb::BytewiseComparator();
        leveldb::BlockBuilder blockbuilder(options);

        for (int i = 1000000; i < 1999999; i++) {
            std::string key_base = "abcdefgh";
            key_base += std::to_string(i);
            leveldb::Slice key(key_base);
            leveldb::Slice value("yuisq");
            blockbuilder.Add(key, value);
        }
        leveldb::Slice datablock = blockbuilder.Finish();
        leveldb::BlockContents contents(datablock,
            /*cachable*/ false,
            /*heap_allocate*/ false);
        leveldb::Block block(contents);

        leveldb::Parallel_Iterator* P_Iter = block.NewParallel_Iterator(leveldb::BytewiseComparator());

        //start
        for (; P_Iter->Valid(); P_Iter->Next()) {
            leveldb::Slice key(P_Iter->key());
            leveldb::Slice value(P_Iter->value());
            //std::string value_(value.data(), value.size());
            //std::cout << key.data() << " " << value_ << std::endl;
        }
        //end
        delete options;
    }

    // unpack using cpu
    {
        leveldb::Options* options = new leveldb::Options();
        options->block_restart_interval = 20;
        options->comparator = leveldb::BytewiseComparator();
        leveldb::BlockBuilder blockbuilder(options);

        for (int i = 1000000; i < 1999999; i++) {
            std::string key_base = "abcdefgh";
            key_base += std::to_string(i);
            leveldb::Slice key(key_base);
            leveldb::Slice value("yuisq");
            blockbuilder.Add(key, value);
        }
        leveldb::Slice datablock = blockbuilder.Finish();
        leveldb::BlockContents contents(datablock,
            /*cachable*/ false,
            /*heap_allocate*/ false);
        leveldb::Block block(contents);

        leveldb::Iterator* Iter = block.NewIterator(leveldb::BytewiseComparator());
        Iter->SeekToFirst();
        Iter->Next();

        //start
        for (; Iter->Valid(); Iter->Next()) {
            leveldb::Slice key(Iter->key());
            leveldb::Slice value(Iter->value());
        }
        //end
        delete options;
    }

    return 0;
}

