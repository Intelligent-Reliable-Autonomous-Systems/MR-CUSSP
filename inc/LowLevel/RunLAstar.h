#pragma once

#include "LowLevel/ShortestPathHeuristic.h"
#include "LowLevel/Utils/Logger.h"
#include "Definitions.h"
#include "LexCompare.h"

#include <queue>
#include <unordered_map>
#include <vector>

class RunLAstar {
public:
    RunLAstar(size_t graph_size,
              const AdjacencyMatrix& graph,
              Heuristic& heuristic,
              size_t source,
              size_t target,
              LoggerPtr logger,
              unsigned int time_limit,
              VertexConstraint& vertex_constraints,
              EdgeConstraint& edge_constraints,
              int turn_dim,
              int turn_cost);

    std::string get_solver_name() { return "LAstar"; }

    bool search(PathSet& waypoints,
                CostSet& apex_costs,
                CostSet& real_costs,
                std::unordered_map<int, int>& conflict_num_map);

    void set_ordering(const macussp::ObjectiveOrdering& ordering) { ordering_ = ordering; }

private:
    size_t graph_size_;
    const AdjacencyMatrix& graph_;
    Heuristic& heuristic_;
    size_t source_, target_;
    LoggerPtr logger_;
    unsigned int time_limit_;
    VertexConstraint& v_constraints_;
    EdgeConstraint& e_constraints_;
    int turn_dim_;
    int turn_cost_;
    macussp::ObjectiveOrdering ordering_;

    struct Node {
        size_t id;
        CostVector g;
        CostVector h;
        std::vector<size_t> path;

        macussp::ObjectiveOrdering ord;

        bool operator>(const Node& other) const {
            return macussp::lex_less(other.g, g, ord);
        }
    };

    bool is_constrained(size_t node_id, size_t timestep);
    bool is_edge_constrained(size_t from, size_t to, size_t timestep);
};
