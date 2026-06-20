#pragma once

#include "Definitions.h"
#include "MAP.h"
#include "agent.h"
#include "environment.h"
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace macussp {

struct MapfProblem {
    Map map;
    std::vector<Edge> edges;
    std::vector<std::pair<size_t, size_t>> start_goal;
};

// Build MAPF problem in memory from Environment + agent states (no .scen file).
MapfProblem build_mapf_problem(const Environment& env,
                               const std::vector<Agent>& agents,
                               const std::unordered_map<int, LocalState>& agent_goals,
                               int cost_dim);

// Map solver vertex id -> Environment LocalState (coarse grid).
LocalState local_state_from_solver_id(const Environment& env, int row, int col);

}  // namespace macussp
