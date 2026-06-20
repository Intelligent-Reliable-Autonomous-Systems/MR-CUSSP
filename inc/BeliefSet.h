#pragma once

#include <cmath>
#include <set>
#include <vector>

namespace macussp {

// Belief update per landmark observation: intersect viable contexts with observation subset.
inline std::set<int> belief_intersect(const std::set<int>& belief,
                                      const std::set<int>& observation_subset) {
    std::set<int> out;
    std::set_intersection(belief.begin(), belief.end(),
                          observation_subset.begin(), observation_subset.end(),
                          std::inserter(out, out.begin()));
    return out;
}

// Entropy metric used in experiments: |C| (cardinality of viable context set).
inline int belief_entropy_cardinality(const std::set<int>& belief) {
    return static_cast<int>(belief.size());
}

// Shannon entropy H = -sum p(c) log p(c) with uniform distribution over viable contexts.
inline double belief_shannon_entropy(const std::set<int>& belief, int num_contexts) {
    if (belief.empty() || num_contexts <= 0) return 0.0;
    const double p = 1.0 / static_cast<double>(belief.size());
    return -static_cast<double>(belief.size()) * p * std::log(p);
}

inline double cumulative_entropy_cardinality(const std::vector<std::set<int>>& trace) {
    double sum = 0.0;
    for (const auto& s : trace) sum += static_cast<double>(belief_entropy_cardinality(s));
    return sum;
}

}  // namespace macussp
