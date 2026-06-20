#include "LowLevel/RunLAstar.h"

#include <limits>
#include <queue>

RunLAstar::RunLAstar(size_t graph_size,
                     const AdjacencyMatrix& graph,
                     Heuristic& heuristic,
                     size_t source,
                     size_t target,
                     LoggerPtr logger,
                     unsigned int time_limit,
                     VertexConstraint& vertex_constraints,
                     EdgeConstraint& edge_constraints,
                     int turn_dim,
                     int turn_cost)
    : graph_size_(graph_size),
      graph_(graph),
      heuristic_(heuristic),
      source_(source),
      target_(target),
      logger_(logger),
      time_limit_(time_limit),
      v_constraints_(vertex_constraints),
      e_constraints_(edge_constraints),
      turn_dim_(turn_dim),
      turn_cost_(turn_cost),
      ordering_(macussp::identity_ordering(static_cast<int>(graph.get_num_of_objectives())))
{}

bool RunLAstar::is_constrained(size_t node_id, size_t timestep) {
    auto it = v_constraints_.find(timestep);
    return it != v_constraints_.end() &&
           std::find(it->second.begin(), it->second.end(), node_id) != it->second.end();
}

bool RunLAstar::is_edge_constrained(size_t from, size_t to, size_t timestep) {
    auto it = e_constraints_.find(timestep);
    return it != e_constraints_.end() &&
           it->second.count(from) &&
           std::find(it->second.at(from).begin(), it->second.at(from).end(), to) != it->second.at(from).end();
}

bool RunLAstar::search(PathSet& waypoints,
                       CostSet& apex_costs,
                       CostSet& real_costs,
                       std::unordered_map<int, int>& conflict_num_map)
{
    (void)graph_size_;
    (void)logger_;
    (void)time_limit_;
    (void)turn_dim_;
    (void)turn_cost_;
    (void)conflict_num_map;

    using PQ = std::priority_queue<Node, std::vector<Node>, std::greater<Node>>;
    PQ open;
    std::unordered_map<size_t, CostVector> best_cost;

    const bool if_turn = turn_dim_ != -1;
    CostVector init_g(graph_.get_num_of_objectives(), 0);
    CostVector h = heuristic_(source_, static_cast<size_t>(-1), if_turn);
    open.push({source_, init_g, h, {source_}, ordering_});
    best_cost[source_] = init_g;

    const int solution_id = 0;
    while (!open.empty()) {
        Node curr = open.top();
        open.pop();
        const size_t curr_node = curr.id;
        const CostVector g = curr.g;
        const std::vector<size_t> path = curr.path;

        if (is_constrained(curr_node, path.size() - 1)) continue;

        if (curr_node == target_) {
            waypoints[solution_id] = path;
            apex_costs[solution_id] = g;
            real_costs[solution_id] = g;
            return true;
        }

        for (const auto& neighbor : graph_[curr_node]) {
            const size_t next_id = neighbor.target;
            if (is_edge_constrained(curr_node, next_id, path.size() - 1)) continue;

            CostVector new_g = g;
            for (size_t i = 0; i < new_g.size(); ++i) {
                new_g[i] += neighbor.cost[i];
            }

            if (best_cost.count(next_id) &&
                !macussp::lex_less(new_g, best_cost[next_id], ordering_) &&
                macussp::lex_compare(new_g, best_cost[next_id], ordering_) != macussp::LexCompareResult::Greater) {
                continue;
            }

            const CostVector new_h = heuristic_(next_id, curr_node, if_turn);
            std::vector<size_t> new_path = path;
            new_path.push_back(next_id);
            open.push({next_id, new_g, new_h, new_path, ordering_});
            best_cost[next_id] = new_g;
        }
    }

    return false;
}
