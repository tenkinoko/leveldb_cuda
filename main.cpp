#include <iostream>
#include "block_builder.h"
#include "slice.h"
#include "comparator.h"
#include "options.h"
#include "format.h"
#include "block.h"
#include "benchmark.h"
#include <string>

#define VALUESIZE str_128

int main() {
    //parallel unpack using GPU
    double t1 = 0.0;
    double t2 = 0.0;
    char bytes_1024[1024];
    for (int i = 0; i < 1024; i++) {
        bytes_1024[i] = '0';
    }
    std::string str_128(bytes_1024, 127);
    std::string str_256(bytes_1024, 255);
    std::string str_512(bytes_1024, 511);
    std::string str_1024(bytes_1024, 1023);
    //std::cout << TOTALNUM << std::endl;
    //for(int j = 0; j < COUNT; j++)
    //{
    //    leveldb::Options* options = new leveldb::Options();
    //    options->block_restart_interval = 20;
    //    options->comparator = leveldb::BytewiseComparator();
    //    leveldb::BlockBuilder blockbuilder(options);

    //    for (int i = BASE; i < BASE + TOTALNUM; i++) {
    //        std::string key_base = "abcdefgh";
    //        key_base += std::to_string(i);
    //        leveldb::Slice key(key_base); 
    //        size_t value = ULLONG_MAX;
    //        blockbuilder.Add(key, value);
    //    }
    //    leveldb::Slice datablock = blockbuilder.Finish();
    //    leveldb::BlockContents contents(datablock,
    //        /*cachable*/ false,
    //        /*heap_allocate*/ false);
    //    leveldb::Block block(contents);

    //    leveldb::Parallel_Iterator* P_Iter = block.NewParallel_Iterator(leveldb::BytewiseComparator());

    //    //start
    //    LARGE_INTEGER freq, head, tail;
    //    prepare(freq);
    //    timer(head);
    //    for (; P_Iter->Valid(); P_Iter->Next()) {
    //        leveldb::Slice key(P_Iter->key());
    //        leveldb::Slice value(P_Iter->value());
    //        std::string value_(value.data(), value.size());
    //        std::cout << key.data() << " " << value_ << std::endl;
    //    }
    //    timer(tail);
    //    t1 += total(head, tail);
    //    
    //    //end
    //    delete options;
    //}
    //std::cout << "GPU time: " << t1 / COUNT << "us" << " Throughput: " << (TOTALNUM / t1) * COUNT << "req/us" << std::endl;
    // unpack using cpu
    for (int j = 0; j < COUNT; j++)
    {
        leveldb::Options* options = new leveldb::Options();
        options->block_restart_interval = 20;
        options->comparator = leveldb::BytewiseComparator();
        leveldb::BlockBuilder blockbuilder(options);

        for (int i = BASE; i < BASE + TOTALNUM; i++) {
            std::string key_base = "abcdefgh";
            key_base += std::to_string(i);
            leveldb::Slice key(key_base);
            size_t value = ULLONG_MAX;
            //TODO:
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
        LARGE_INTEGER freq, head, tail;
        prepare(freq);
        timer(head);
        for (; Iter->Valid(); Iter->Next()) {
            leveldb::Slice key(Iter->key());
            leveldb::Slice value(Iter->value());
            std::string value_(value.data(), value.size());
            std::cout << key.data() << " " << value_ << std::endl;
        }
        timer(tail);
        t2 += total(head, tail);
        
        
        //end
        delete options;
    }
    std::cout << "CPU time: " << t2 / COUNT << "us" << " Throughput: " << (TOTALNUM / t2) * COUNT << "req/us" << std::endl;
    std::cout << "Speedup: " << t2 / t1 << std::endl;
    return 0;
}

