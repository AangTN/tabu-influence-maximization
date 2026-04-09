#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace im {

inline std::size_t popcount_u64(std::uint64_t value) {
    std::size_t count = 0;
    while (value != 0ULL) {
        value &= (value - 1ULL);
        ++count;
    }
    return count;
}

class BitMask {
public:
    BitMask() = default;
    explicit BitMask(std::size_t word_count) : words_(word_count, 0ULL) {}

    void resize_words(std::size_t word_count) {
        words_.resize(word_count, 0ULL);
    }

    std::size_t word_count() const {
        return words_.size();
    }

    void set_bit(std::size_t bit_index) {
        const std::size_t word_index = bit_index / 64;
        if (word_index >= words_.size()) {
            words_.resize(word_index + 1, 0ULL);
        }
        words_[word_index] |= (1ULL << (bit_index % 64));
    }

    void or_with(const BitMask& other) {
        if (other.words_.size() > words_.size()) {
            words_.resize(other.words_.size(), 0ULL);
        }
        for (std::size_t i = 0; i < other.words_.size(); ++i) {
            words_[i] |= other.words_[i];
        }
    }

    std::size_t count_bits() const {
        std::size_t total = 0;
        for (std::uint64_t word : words_) {
            total += popcount_u64(word);
        }
        return total;
    }

    std::size_t or_count(const BitMask& other) const {
        const std::size_t max_words = words_.size() > other.words_.size() ? words_.size() : other.words_.size();
        std::size_t total = 0;
        for (std::size_t i = 0; i < max_words; ++i) {
            const std::uint64_t a = i < words_.size() ? words_[i] : 0ULL;
            const std::uint64_t b = i < other.words_.size() ? other.words_[i] : 0ULL;
            total += popcount_u64(a | b);
        }
        return total;
    }

    std::size_t and_not_count(const BitMask& mask_to_remove) const {
        const std::size_t max_words = words_.size() > mask_to_remove.words_.size()
            ? words_.size()
            : mask_to_remove.words_.size();
        std::size_t total = 0;
        for (std::size_t i = 0; i < max_words; ++i) {
            const std::uint64_t a = i < words_.size() ? words_[i] : 0ULL;
            const std::uint64_t b = i < mask_to_remove.words_.size() ? mask_to_remove.words_[i] : 0ULL;
            total += popcount_u64(a & (~b));
        }
        return total;
    }

private:
    std::vector<std::uint64_t> words_;
};

}  // namespace im
