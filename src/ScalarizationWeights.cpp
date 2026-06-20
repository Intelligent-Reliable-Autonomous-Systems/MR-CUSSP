#include "ScalarizationWeights.h"

#include "GraphBridge.h"
#include "LexCompare.h"
#include "IOUtils.h"
#include "continuous.h"
#include "params.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>

namespace macussp {

std::vector<double> geometric_scaling_weights(const std::vector<double>& M) {
    const int d = static_cast<int>(M.size());
    if (d == 0) return {};
    double product = 1.0;
    for (double v : M) product *= std::max(v, 1.0);
    const double geo = std::pow(product, 1.0 / static_cast<double>(d));
    std::vector<double> w(static_cast<size_t>(d));
    for (int i = 0; i < d; ++i) {
        w[static_cast<size_t>(i)] = geo / std::max(M[static_cast<size_t>(i)], 1.0);
    }
    return w;
}

static std::vector<std::vector<int>> build_adjacency(const Environment& env) {
    const int n = static_cast<int>(env.vertices().size());
    std::vector<std::vector<int>> adj(static_cast<size_t>(n));
    for (int u = 0; u < n; ++u) {
        for (int v : env.neighbors(u)) {
            adj[static_cast<size_t>(u)].push_back(v);
        }
    }
    return adj;
}

static CostVector dijkstra_cost(const std::vector<Edge>& edges, int source, int target, int dim) {
    std::unordered_map<size_t, std::vector<std::pair<size_t, CostVector>>> adj;
    for (const auto& e : edges) {
        adj[e.source].emplace_back(e.target, e.cost);
    }
    const size_t S = static_cast<size_t>(source);
    const size_t T = static_cast<size_t>(target);
    std::unordered_map<size_t, CostVector> dist;
    using QItem = std::pair<CostVector, size_t>;
    auto cmp = [](const QItem& a, const QItem& b) {
        for (size_t i = 0; i < a.first.size(); ++i) {
            if (a.first[i] != b.first[i]) return a.first[i] > b.first[i];
        }
        return false;
    };
    std::priority_queue<QItem, std::vector<QItem>, decltype(cmp)> pq(cmp);
    dist[S] = CostVector(static_cast<size_t>(dim), 0);
    pq.push({dist[S], S});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d != dist[u]) continue;
        if (u == T) return d;
        for (const auto& [v, c] : adj[u]) {
            CostVector nd = d;
            for (int i = 0; i < dim; ++i) nd[static_cast<size_t>(i)] += c[static_cast<size_t>(i)];
            auto it = dist.find(v);
            if (it == dist.end() || macussp::lex_less(nd, it->second)) {
                dist[v] = std::move(nd);
                pq.push({dist[v], v});
            }
        }
    }
    return CostVector(static_cast<size_t>(dim), 0);
}

std::vector<double> estimate_domain_M(const Environment& env,
                                      const DomainObjectiveGrids& grids,
                                      int samples) {
    (void)grids;
    PreProcessor p;
    Map map;
    std::vector<Edge> edges;
    std::unordered_map<size_t, std::vector<int>> local_id2coord;
    const std::string map_path = "../maps/" + env.getMapName() + "/" + env.getMapName() + ".map";
    p.read_map(map_path, map, local_id2coord);
    p.cost_init(map, edges, env.numOfObjectives);
    const auto paths = domain_objective_cost_paths(env.getMapName());
    for (int i = 0; i < env.numOfObjectives; ++i) {
        p.read_cost(paths[static_cast<size_t>(i)], map, edges, i);
    }

    std::vector<int> free_ids;
    free_ids.reserve(map.graph_size);
    for (size_t id = 0; id < map.graph_size; ++id) free_ids.push_back(static_cast<int>(id));

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(free_ids.size()) - 1);

    std::vector<double> M(static_cast<size_t>(env.numOfObjectives), 1.0);
    for (int s = 0; s < samples; ++s) {
        const int a = free_ids[static_cast<size_t>(pick(rng))];
        int b = a;
        while (b == a) b = free_ids[static_cast<size_t>(pick(rng))];
        const CostVector c = dijkstra_cost(edges, a, b, env.numOfObjectives);
        for (int i = 0; i < env.numOfObjectives; ++i) {
            M[static_cast<size_t>(i)] = std::max(M[static_cast<size_t>(i)],
                                                 static_cast<double>(c[static_cast<size_t>(i)]));
        }
    }
    for (double& v : M) v = std::max(v, 1.0);
    return M;
}

const std::vector<double>& cached_domain_M(const std::string& domain, const Environment& env) {
    static std::unordered_map<std::string, std::vector<double>> cache;
    auto it = cache.find(domain);
    if (it == cache.end()) {
        const auto& grids = ensure_domain_costs(const_cast<Environment&>(env));
        cache[domain] = estimate_domain_M(env, grids, kScalarizationSamples);
        it = cache.find(domain);
    }
    return it->second;
}

}  // namespace macussp
