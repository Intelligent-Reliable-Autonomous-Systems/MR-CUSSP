#include "LowLevel/RunLAstar.h"
#include "Definitions.h"

#include <functional>
#include <iostream>
#include <unordered_map>

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond "\n"; return 1; } } while (0)

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    // 5-node line: 0-1-2-3-4, unit costs on 3 objectives
    std::vector<Edge> edges;
    auto add_edge = [&](size_t u, size_t v, size_t c) {
        edges.emplace_back(u, v, CostVector{c, c, c});
        edges.emplace_back(v, u, CostVector{c, c, c});
    };
    for (size_t i = 0; i < 5; ++i) {
        edges.emplace_back(i, i, CostVector{0, 0, 0});
    }
    add_edge(0, 1, 1);
    add_edge(1, 2, 1);
    add_edge(2, 3, 1);
    add_edge(3, 4, 1);

    AdjacencyMatrix graph(5, edges);
    Heuristic h = [](size_t, size_t, bool) { return CostVector{0, 0, 0}; };

    VertexConstraint vc;
    EdgeConstraint ec;
  LoggerPtr logger = nullptr;

    RunLAstar solver(5, graph, h, 0, 4, logger, 10, vc, ec, 0, 0);
    PathSet waypoints;
    CostSet apex_costs;
    CostSet real_costs;
    std::unordered_map<int, int> conflict_num;
    const bool ok = solver.search(waypoints, apex_costs, real_costs, conflict_num);
    CHECK(ok);
    CHECK(real_costs[0] == CostVector({4, 4, 4}));
    CHECK(waypoints[0].size() == 5);

    std::cout << "test_lastar: OK\n";
    return 0;
}
