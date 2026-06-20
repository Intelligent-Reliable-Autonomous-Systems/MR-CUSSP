#include "LowLevel/RunLAstar.h"
#include "LowLevel/RunApex.h"
#include "LowLevel/ShortestPathHeuristic.h"
#include "LexCompare.h"
#include "Definitions.h"

#include <functional>
#include <iostream>
#include <random>
#include <unordered_map>

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond "\n"; return 1; } } while (0)

static Heuristic make_heuristic(size_t n, const std::vector<Edge>& edges, size_t target) {
    std::vector<Edge> mutable_edges = edges;
    AdjacencyMatrix inv_graph(n, mutable_edges, true);
    auto sp = std::make_shared<ShortestPathHeuristic>(target, n, inv_graph, -1, 0);
    return [sp](size_t node_id, size_t parent_id, bool if_turn) {
        return (*sp)(node_id, parent_id, if_turn);
    };
}

static bool run_apex(const std::vector<Edge>& edges, size_t n, size_t source, size_t target,
                     const macussp::ObjectiveOrdering& theta, CostVector& out_cost) {
    std::vector<Edge> mutable_edges = edges;
    AdjacencyMatrix graph(n, mutable_edges);
    Heuristic h = make_heuristic(n, edges, target);
    VertexConstraint vc;
    EdgeConstraint ec;
    VertexCAT vertex_cat;
    EdgeCAT edge_cat;
    std::unordered_map<int, int> conflict_num;
    PathSet waypoints;
    CostSet apex_costs;
    CostSet real_costs;
    single_run_map(n, graph, h, source, target, LSolver::APEX, 0.0, MergingStrategy::MORE_SLACK,
                   nullptr, 5, waypoints, apex_costs, real_costs, vc, ec, vertex_cat, edge_cat,
                   conflict_num, -1, 0);
    if (real_costs.empty()) return false;
    bool have = false;
    for (const auto& kv : real_costs) {
        if (!have || macussp::lex_less(kv.second, out_cost, theta)) {
            out_cost = kv.second;
            have = true;
        }
    }
    return have;
}

static bool run_lastar(const std::vector<Edge>& edges, size_t n, size_t source, size_t target,
                       const macussp::ObjectiveOrdering& ord, CostVector& out_cost) {
    std::vector<Edge> mutable_edges = edges;
    AdjacencyMatrix graph(n, mutable_edges);
    Heuristic h = make_heuristic(n, edges, target);
    VertexConstraint vc;
    EdgeConstraint ec;
    RunLAstar solver(n, graph, h, source, target, nullptr, 5, vc, ec, -1, 0);
    solver.set_ordering(ord);
    PathSet waypoints;
    CostSet apex_costs;
    CostSet real_costs;
    std::unordered_map<int, int> conflict_num;
    if (!solver.search(waypoints, apex_costs, real_costs, conflict_num)) return false;
    out_cost = real_costs.begin()->second;
    return true;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    const std::vector<macussp::ObjectiveOrdering> thetas = {
        {0, 1, 2}, {1, 0, 2}, {2, 0, 1}, {1, 2, 0}
    };

    // Hand-built + seeded random small graphs with heterogeneous 3D costs.
    const std::vector<std::vector<Edge>> graphs = [] {
        std::vector<std::vector<Edge>> out;
        auto line = [](std::initializer_list<std::tuple<int,int,int,int,int>> es) {
            std::vector<Edge> edges;
            for (auto [u,v,a,b,c] : es) {
                CostVector cost = {static_cast<size_t>(a), static_cast<size_t>(b), static_cast<size_t>(c)};
                edges.emplace_back(static_cast<size_t>(u), static_cast<size_t>(v), cost);
                edges.emplace_back(static_cast<size_t>(v), static_cast<size_t>(u), cost);
            }
            return edges;
        };
        out.push_back(line({{0,1,1,1,1},{1,2,1,1,1},{2,3,1,1,1},{3,4,1,1,1}}));
        out.push_back(line({{0,1,1,2,3},{1,2,1,1,1},{2,3,2,1,1},{3,4,1,3,2}}));
        out.push_back(line({{0,1,2,1,1},{1,2,1,2,1},{2,3,1,1,2},{3,4,3,1,1},{0,2,4,1,1},{1,3,2,2,2}}));

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> cost_dist(1, 4);
        for (int g = 0; g < 8; ++g) {
            const size_t n = 6;
            std::vector<Edge> edges;
            for (size_t i = 0; i + 1 < n; ++i) {
                CostVector c = {static_cast<size_t>(cost_dist(rng)),
                                static_cast<size_t>(cost_dist(rng)),
                                static_cast<size_t>(cost_dist(rng))};
                edges.emplace_back(i, i + 1, c);
                edges.emplace_back(i + 1, i, c);
            }
            if (g % 2 == 0 && n >= 4) {
                CostVector c = {static_cast<size_t>(cost_dist(rng)),
                                static_cast<size_t>(cost_dist(rng)),
                                static_cast<size_t>(cost_dist(rng))};
                edges.emplace_back(0, 3, c);
                edges.emplace_back(3, 0, c);
            }
            out.push_back(std::move(edges));
        }
        return out;
    }();

    int graph_idx = 0;
    for (const auto& edges : graphs) {
        size_t n = 0;
        for (const auto& e : edges) {
            n = std::max(n, std::max(e.source, e.target) + 1);
        }
        const size_t source = 0;
        const size_t target = n - 1;
        for (const auto& theta : thetas) {
            CostVector apex;
            CostVector lastar;
            CHECK(run_apex(edges, n, source, target, theta, apex));
            CHECK(run_lastar(edges, n, source, target, theta, lastar));
            CHECK(macussp::lex_compare(apex, lastar, theta) == macussp::LexCompareResult::Equal);
        }
        ++graph_idx;
        (void)graph_idx;
    }

    std::cout << "test_apex_lastar_equiv: OK (A*pex@eps=0 matches Algorithm 3 LA* under Theta)\n";
    return 0;
}
