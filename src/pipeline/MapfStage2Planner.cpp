#include "pipeline/MapfStage2Planner.h"

#include "GraphBridge.h"
#include "ScalarizationWeights.h"
#include "mapf_solver.h"
#include "helper_utils.h"
#include "high_level_planner.h"
#include "params.h"

#include <iostream>

namespace macussp {

static SolverConfig make_stage2_config(const Environment& env,
                                       const std::string& algorithm,
                                       int agent_count,
                                       int time_limit_sec,
                                       const ObjectiveOrdering& ordering)
{
    SolverConfig cfg;
    cfg.algorithm = algorithm;
    cfg.time_limit_sec = time_limit_sec;
    cfg.dim = env.numOfObjectives;
    cfg.agent_num = agent_count;
    cfg.map_file = "../maps/" + env.getMapName() + "/" + env.getMapName() + ".map";
    cfg.output_file = MapfSolver::makeOutputPath(env.getMapName(), time_limit_sec);
    cfg.conflict_based = true;
    cfg.eager = false;
    cfg.eps_i = {0,0,0,0,0,0,0,0,0,0};
    cfg.turn_dim = 0;
    cfg.turn_cost = 0;
    cfg.objective_ordering = ordering;
    cfg.use_continuous = ((env.getMapName() == "salp") || (env.getMapName().find("salp") != std::string::npos))
                         && (SALP_SUBRES > 1);

    if (algorithm == "LCBS" || algorithm == "SCALARIZATION") {
        cfg.solution_num = 1;
    } else if (algorithm == "BBMOCBS-k" || algorithm == "BBMOCBS-pex") {
        cfg.solution_num = 8;
    }

    if (algorithm == "SCALARIZATION") {
        const auto& M = cached_domain_M(env.getMapName(), env);
        cfg.scalarization_weights = geometric_scaling_weights(M);
    }
    return cfg;
}

static std::unordered_map<int, std::vector<LocalState>> paths_from_solve(
    const Environment& env,
    const std::vector<Agent>& agents,
    const AgentPaths& solver_paths)
{
    const bool using_refined =
        ((env.getMapName() == "salp") || (env.getMapName().find("salp") != std::string::npos))
        && (SALP_SUBRES > 1);

    std::unordered_map<int, std::vector<LocalState>> agentPaths;
    for (int agentId = 0; agentId < static_cast<int>(agents.size()); ++agentId) {
        const auto& path = solver_paths.at(agentId);
        std::vector<LocalState> localStatesPath;
        localStatesPath.reserve(path.size());
        for (const auto& step : path) {
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

class MapfStage2Planner : public Stage2Planner {
public:
    MapfStage2Planner(std::string algorithm, std::string stage2_name)
        : algorithm_(std::move(algorithm)), stage2_name_(std::move(stage2_name)) {}

    std::string name() const override { return stage2_name_; }

    Stage2Result run(const Stage2Input& input) override {
        Stage2Result result;
        TimingHarness timer;
        timer.start();

        Environment& env = input.env;
        std::vector<Agent>& agents = input.agents;
        PipelineDeadline& deadline = input.deadline;

        MapfSolver solver(algorithm_, deadline.mapf_time_limit_sec(), env.numOfObjectives);

        bool all_tasks_completed = false;
        CostVector accumulated_cost(static_cast<size_t>(env.numOfObjectives), 0);

        while (!all_tasks_completed) {
            if (deadline.expired()) {
                result.timed_out = true;
                result.success = false;
                result.wall_time_sec = timer.elapsed_sec();
                return result;
            }

            const int per_call_limit = deadline.mapf_time_limit_sec();
            SolverConfig cfg = make_stage2_config(env, algorithm_, static_cast<int>(agents.size()),
                                                  per_call_limit, input.ordering);

            auto taskGoalsResult = high_level_planner::getTaskGoals4AllAgents(env, agents);
            all_tasks_completed = taskGoalsResult.first;
            const auto& agent2Goal = taskGoalsResult.second;
            env.clearFireCache();

            MapfProblem problem = build_mapf_problem(env, agents, agent2Goal, cfg.dim);
            MapfSolveResult solved = solver.solve(cfg, problem);
            result.solver_timings.hl_merging_sec += solved.timings.hl_merging_sec;
            result.solver_timings.low_level_sec += solved.timings.low_level_sec;
            result.solver_timings.total_sec += solved.timings.total_sec;

            if (deadline.expired()) {
                result.timed_out = true;
                result.success = false;
                result.wall_time_sec = timer.elapsed_sec();
                return result;
            }

            if (!solved.success) {
                result.success = false;
                result.wall_time_sec = timer.elapsed_sec();
                return result;
            }

            for (size_t i = 0; i < accumulated_cost.size() && i < solved.cost.size(); ++i) {
                accumulated_cost[i] += solved.cost[i];
            }

            auto finalPaths = paths_from_solve(env, agents, solved.paths);
            appendMapfTrajectories(agents, finalPaths, env, all_tasks_completed);
        }

        result.success = !deadline.expired();
        result.timed_out = deadline.expired();
        result.joint_cost = accumulated_cost;
        result.wall_time_sec = timer.elapsed_sec();
        return result;
    }

private:
    std::string algorithm_;
    std::string stage2_name_;
};

std::unique_ptr<Stage2Planner> make_mapf_stage2_planner(const std::string& algorithm) {
    if (algorithm == "lcbs" || algorithm == "LCBS") {
        return std::make_unique<MapfStage2Planner>("LCBS", "lcbs");
    }
    if (algorithm == "scalarization" || algorithm == "SCALARIZATION" || algorithm == "scalarized") {
        return std::make_unique<MapfStage2Planner>("SCALARIZATION", "scalarization");
    }
    if (algorithm == "bbmocbs-k" || algorithm == "BBMOCBS-k" || algorithm == "mocbs_k") {
        return std::make_unique<MapfStage2Planner>("BBMOCBS-k", "bbmocbs-k");
    }
    if (algorithm == "bbmocbs-pex" || algorithm == "BBMOCBS-pex" || algorithm == "mocbs") {
        return std::make_unique<MapfStage2Planner>("BBMOCBS-pex", "bbmocbs-pex");
    }
    return nullptr;
}

}  // namespace macussp
