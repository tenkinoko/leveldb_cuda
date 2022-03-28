#pragma once
#ifndef LINEAR_REGRESSION_CODING_H_
#define LINEAR_REGRESSION_CODING_H_

#include<string>
#include<vector>
#include <limits>

#include "slice.h"

namespace leveldb {
    class LinearModelBuilder {
    public:

        LinearModelBuilder() = default;
        LinearModelBuilder(const LinearModelBuilder&) = delete;
        LinearModelBuilder& operator=(const LinearModelBuilder&) = delete;

        std::vector<unsigned int> non_negative_error() const {
            return non_neg_error_;
        }

        size_t slope() const {
            return slope_;
        }

        size_t intercept() const {
            return intercept_;
        }

        size_t min_error() const {
            return min_error_;
        }

        size_t count() const {
            return count_;
        }

        void Add(Slice& key) {
            auto y_temp = key.ull();
            y_.push_back(y_temp);
            x_sum_ += count_;
            y_sum_ += y_temp;
            xx_sum_ += count_ * count_;
            xy_sum_ += count_ * y_temp;
            count_++;
        }

        //void Add(size_t& key) {
        //    auto y_temp = key;
        //    y_.push_back(y_temp);
        //    x_sum_ += count_;
        //    y_sum_ += y_temp;
        //    xx_sum_ += count_ * count_;
        //    xy_sum_ += count_ * y_temp;
        //    count_++;
        //}

        void build() {
            auto x_bar = x_sum_ / count_;
            auto y_bar = y_sum_ / count_;
            auto l_x_y = xy_sum_ - x_sum_ * y_sum_ / count_;
            auto l_x_x = xx_sum_ - x_sum_ * x_sum_ / count_;
            auto slope = l_x_y / l_x_x;
            auto intercept = y_bar - static_cast<size_t>(slope * x_bar);

            slope_ = slope;
            intercept_ = intercept;
        }

        void finish() {
            auto min_error = std::numeric_limits<int>::max();
            for (int i = 0; i < count_; i++) {
                auto error = y_[i] - slope_ * i + intercept_;
                if (error < min_error) min_error = error;
                error_.emplace_back(error);
            }
            // set min_error_
            min_error_ = min_error;
            for (auto e : error_) {
                e -= min_error;
                non_neg_error_.emplace_back(static_cast<unsigned int> (e));
            }
            return;
        }

    private:
        unsigned int count_ = 0;
        size_t x_sum_ = 0;
        size_t y_sum_ = 0;
        size_t xy_sum_ = 0;
        size_t xx_sum_ = 0;
        size_t slope_ = 0;
        size_t intercept_ = 0;
        size_t min_error_ = 0;
        std::vector<size_t> y_;
        std::vector<size_t> error_;
        std::vector<unsigned int> non_neg_error_;
    };

} // namespace leveldb


#endif