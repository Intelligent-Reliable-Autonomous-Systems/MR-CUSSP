#pragma once

#include "Definitions.h"
#include "LexCompare.h"

#include <limits>
#include <unordered_map>
#include <vector>

extern std::unordered_map<size_t, std::vector<int>> id2coord;

namespace macussp {

inline int select_frontier_index(const std::vector<CostVector>& costs,
                                 const ObjectiveOrdering& ordering) {
    if (costs.empty()) return -1;
    int best = 0;
    for (int i = 1; i < static_cast<int>(costs.size()); ++i) {
        if (lex_less(costs[static_cast<size_t>(i)], costs[static_cast<size_t>(best)], ordering)) {
            best = i;
        }
    }
    return best;
}

inline AgentPaths build_agent_paths_from_ids(
    const std::vector<std::vector<size_t>>& per_agent_path_ids)
{
    AgentPaths out;
    for (int ag = 0; ag < static_cast<int>(per_agent_path_ids.size()); ++ag) {
        std::vector<std::tuple<int, int>> path_xy;
        for (size_t node_id : per_agent_path_ids[static_cast<size_t>(ag)]) {
            const auto it = id2coord.find(node_id);
            if (it == id2coord.end()) continue;
            const auto& coord = it->second;
            path_xy.emplace_back(coord[1], coord[0]);
        }
        out[ag] = std::move(path_xy);
    }
    return out;
}

}  // namespace macussp
