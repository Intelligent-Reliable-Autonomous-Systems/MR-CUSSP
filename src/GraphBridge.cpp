#include "GraphBridge.h"

#include "DomainCosts.h"
#include "IOUtils.h"
#include "continuous.h"
#include "params.h"

#include <iostream>
#include <stdexcept>

extern std::unordered_map<size_t, std::vector<int>> id2coord;
extern std::string map_name;

namespace macussp {

static void assign_edge_costs_from_vertex_grids(
    Map& map, std::vector<Edge>& edges, const DomainObjectiveGrids& grids, int cost_dim)
{
    std::vector<std::vector<CostVector>> cell_cost(static_cast<size_t>(map.height),
        std::vector<CostVector>(static_cast<size_t>(map.width),
                                CostVector(static_cast<size_t>(cost_dim), 0)));
    for (int y = 0; y < map.height; ++y) {
        for (int x = 0; x < map.width; ++x) {
            if (map.getVal(y, x) == -1) continue;
            for (int d = 0; d < cost_dim; ++d) {
                cell_cost[static_cast<size_t>(y)][static_cast<size_t>(x)][static_cast<size_t>(d)] =
                    static_cast<size_t>(grids.grids[static_cast<size_t>(d)][static_cast<size_t>(y)][static_cast<size_t>(x)]);
            }
        }
    }
    for (auto& e : edges) {
        if (e.source == e.target) {
            e.cost.assign(static_cast<size_t>(cost_dim), 0);
            continue;
        }
        int ty = -1, tx = -1;
        for (int r = 0; r < map.height; ++r) {
            for (int c = 0; c < map.width; ++c) {
                if (map.getID(r, c) == e.target) {
                    ty = r;
                    tx = c;
                    break;
                }
            }
        }
        if (ty >= 0) {
            e.cost = cell_cost[static_cast<size_t>(ty)][static_cast<size_t>(tx)];
        }
    }
}

MapfProblem build_mapf_problem(const Environment& env,
                               const std::vector<Agent>& agents,
                               const std::unordered_map<int, LocalState>& agent_goals,
                               int cost_dim)
{
    PreProcessor p;
    MapfProblem problem;
    const std::string domain = env.getMapName();
    const std::string map_path = "../maps/" + domain + "/" + domain + ".map";
    map_name = map_path;

    const DomainObjectiveGrids& grids = env.ensureDomainObjectiveGrids();
    const int dim = std::min(cost_dim, env.numOfObjectives);

    const bool use_continuous =
        ((domain == "salp") || (domain.find("salp") != std::string::npos)) && (SALP_SUBRES > 1);

    if (use_continuous) {
        std::vector<std::string> rows;
        p.read_map_rows(map_path, rows);
        std::vector<std::vector<double>> cost_grids;
        cost_grids.reserve(static_cast<size_t>(dim));
        for (int i = 0; i < dim; ++i) {
            std::vector<double> flat;
            flat.reserve(static_cast<size_t>(grids.height * grids.width));
            for (int r = 0; r < grids.height; ++r) {
                for (int c = 0; c < grids.width; ++c) {
                    flat.push_back(grids.grids[static_cast<size_t>(i)][static_cast<size_t>(r)][static_cast<size_t>(c)]);
                }
            }
            cost_grids.push_back(std::move(flat));
        }
        ContinuousDomain dom = build_continuous_from_raster(rows, cost_grids);
        discretize_lattice_from_continuous(dom, SALP_SUBRES, SALP_ALLOW_DIAGONAL,
                                           problem.map, problem.edges, dim);
        id2coord.clear();
        id2coord.reserve(problem.map.graph_size);
        for (int r = 0; r < problem.map.height; ++r) {
            for (int c = 0; c < problem.map.width; ++c) {
                size_t id = problem.map.getID(r, c);
                if (id != static_cast<size_t>(-1)) {
                    id2coord[id] = {r, c};
                }
            }
        }
    } else {
        p.read_map(map_path, problem.map, id2coord);
        p.cost_init(problem.map, problem.edges, dim);
        assign_edge_costs_from_vertex_grids(problem.map, problem.edges, grids, dim);
    }

    std::vector<std::pair<int,int>> starts;
    std::vector<std::pair<int,int>> goals;
    starts.reserve(agents.size());
    goals.reserve(agents.size());
    for (const auto& agent : agents) {
        const auto s = agent.getState();
        starts.emplace_back(s.x, s.y);
        const auto& g = agent_goals.at(agent.getId());
        goals.emplace_back(g.x, g.y);
    }

    const int refine = use_continuous ? SALP_SUBRES : 1;
    p.build_start_goal_from_coords(problem.map, starts, goals, problem.start_goal, refine);
    return problem;
}

LocalState local_state_from_solver_id(const Environment& env, int row, int col) {
    const bool using_refined =
        ((env.getMapName() == "salp") || (env.getMapName().find("salp") != std::string::npos))
        && (SALP_SUBRES > 1);
    int cx = using_refined ? col / SALP_SUBRES : col;
    int cy = using_refined ? row / SALP_SUBRES : row;
    return env.getLocalState(cx, cy);
}

}  // namespace macussp
