#pragma once

#include "Definitions.h"
#include <algorithm>
#include <numeric>
#include <vector>

namespace macussp {

// ordering[priority] = objective index (0 .. dim-1)
using ObjectiveOrdering = std::vector<int>;

enum class LexCompareResult { Less, Equal, Greater };

inline ObjectiveOrdering identity_ordering(int dim) {
    ObjectiveOrdering o(static_cast<size_t>(dim));
    std::iota(o.begin(), o.end(), 0);
    return o;
}

inline LexCompareResult lex_compare(const CostVector& a,
                                    const CostVector& b,
                                    const ObjectiveOrdering& ordering) {
    const int d = static_cast<int>(ordering.size());
    for (int p = 0; p < d; ++p) {
        const int idx = ordering[static_cast<size_t>(p)];
        if (idx < 0 || idx >= static_cast<int>(a.size()) || idx >= static_cast<int>(b.size())) {
            continue;
        }
        const size_t u = static_cast<size_t>(idx);
        if (a[u] < b[u]) return LexCompareResult::Less;
        if (a[u] > b[u]) return LexCompareResult::Greater;
    }
    if (a.size() < b.size()) return LexCompareResult::Less;
    if (a.size() > b.size()) return LexCompareResult::Greater;
    return LexCompareResult::Equal;
}

inline bool lex_less(const CostVector& a, const CostVector& b, const ObjectiveOrdering& ordering) {
    return lex_compare(a, b, ordering) == LexCompareResult::Less;
}

inline bool lex_less(const CostVector& a, const CostVector& b) {
    return lex_less(a, b, identity_ordering(static_cast<int>(std::min(a.size(), b.size()))));
}

}  // namespace macussp
