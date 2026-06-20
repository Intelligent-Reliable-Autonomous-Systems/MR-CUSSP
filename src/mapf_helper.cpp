#include "params.h"
#include "mapf_helper.h"
#include "mapf_solver.h"
#include "GraphBridge.h"
#include "IOUtils.h"
#include "environment.h"
#include "agent.h"
#include "state.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

void createScenarioFile(const std::string &scenarioFileName,
                        const std::vector<Agent> &agents,
                        const std::unordered_map<int, LocalState> &agent2TargetStatesMap,
                        const Environment &env) {
    (void)scenarioFileName;
    (void)agents;
    (void)agent2TargetStatesMap;
    (void)env;
    // Legacy file-based scenario generation removed from composition path.
}

std::unordered_map<int, std::vector<LocalState>> runMapfSolver(
    const std::string &algorithm,
    Environment &env,
    const std::vector<Agent> &agents,
    const std::unordered_map<int, LocalState> &agent2TargetStatesMap,
    int timeLimitSec,
    std::string /*scenarioFileName*/)
{
    std::unordered_map<int, std::vector<LocalState>> empty;

    constexpr int kFormationDim = 6;
    MapfSolver mapfSolver(algorithm, timeLimitSec, kFormationDim);

    SolverConfig cfg;
    cfg.algorithm = algorithm;
    cfg.time_limit_sec = timeLimitSec;
    cfg.dim = kFormationDim;
    cfg.agent_num = static_cast<int>(agents.size());
    const std::string domain = env.getMapName();
    cfg.map_file = "../maps/" + domain + "/" + domain + ".map";
    cfg.output_file = MapfSolver::makeOutputPath(domain, timeLimitSec);
    cfg.cost_maps = {
        "../maps/" + domain + "/cost-1.cost",
        "../maps/" + domain + "/cost-2.cost",
        "../maps/" + domain + "/cost-3.cost",
        "../maps/" + domain + "/cost-4.cost",
        "../maps/" + domain + "/cost-5.cost",
        "../maps/" + domain + "/cost-6.cost",
        "../maps/" + domain + "/cost-7.cost",
        "../maps/" + domain + "/cost-8.cost",
        "../maps/" + domain + "/cost-9.cost",
        "../maps/" + domain + "/cost-10.cost"
    };
    cfg.conflict_based = true;
    cfg.eager = false;
    cfg.eps_i = {0,0,0,0,0,0,0,0,0,0};
    cfg.turn_dim = 0;
    cfg.turn_cost = 0;
    cfg.solution_num = 1;
    cfg.use_continuous = false;

    PreProcessor p;
    macussp::MapfProblem problem;
    extern std::unordered_map<size_t, std::vector<int>> id2coord;
    p.read_map(cfg.map_file, problem.map, id2coord);
    p.cost_init(problem.map, problem.edges, kFormationDim);
    for (int i = 0; i < kFormationDim; ++i) {
        p.read_cost(cfg.cost_maps[static_cast<size_t>(i)], problem.map, problem.edges, i);
    }

    std::vector<std::pair<int,int>> starts;
    std::vector<std::pair<int,int>> goals;
    starts.reserve(agents.size());
    goals.reserve(agents.size());
    for (const auto& agent : agents) {
        const auto s = agent.getState();
        starts.emplace_back(s.x, s.y);
        const auto& g = agent2TargetStatesMap.at(agent.getId());
        goals.emplace_back(g.x, g.y);
    }
    p.build_start_goal_from_coords(problem.map, starts, goals, problem.start_goal, 1);

    MapfSolveResult solved = mapfSolver.solve(cfg, problem);

    for (int i = 0; i < env.numOfObjectives && i < static_cast<int>(solved.cost.size()); ++i) {
        env.TotalCostVector[static_cast<size_t>(i)] += static_cast<double>(solved.cost[static_cast<size_t>(i)]);
    }

    if (!solved.success) {
        std::cerr << "MAPF solver failed to find a solution." << std::endl;
        return empty;
    }

    const bool using_refined =
        ((env.getMapName() == "salp") || (env.getMapName().find("salp") != std::string::npos))
        && (SALP_SUBRES > 1);

    std::unordered_map<int, std::vector<LocalState>> agentPaths;
    for (int agentId = 0; agentId < static_cast<int>(agents.size()); ++agentId) {
        const auto &path = solved.paths.at(agentId);
        std::vector<LocalState> localStatesPath;
        localStatesPath.reserve(path.size());
        for (const auto &step : path) {
            int rr = std::get<0>(step);
            int cc = std::get<1>(step);
            int cx = using_refined ? rr / SALP_SUBRES : rr;
            int cy = using_refined ? cc / SALP_SUBRES : cc;
            localStatesPath.push_back(env.getLocalState(cx, cy));
        }
        agentPaths[agentId] = std::move(localStatesPath);
    }
    return agentPaths;
}
