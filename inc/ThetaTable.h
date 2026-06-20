#pragma once

#include "LexCompare.h"
#include <set>
#include <string>
#include <vector>

namespace macussp {

// Paper contexts c1..c3 map to ids 1..3 (id k = paper c_k). TrueContext=2 => paper c2.
// Distractor ids {0,4} use identity ordering (never paper contexts).
ObjectiveOrdering theta_for_context(const std::string& domain, int context_id);

std::set<int> viable_true_context_ids(const std::string& domain);

bool is_viable_true_context(const std::string& domain, int context_id);

}  // namespace macussp
