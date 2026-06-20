#include "mapf_solver.h"

#include "GraphBridge.h"
#include "continuous.h"
#include "IOUtils.h"
#include "params.h"
#include <iostream>
#include <memory>
#include <optional>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <ctime>

#include "LowLevel/Utils/Logger.h"
#include "HighLevel/HighLevelSolver.h"
#include "HighLevel/epsSolver.h"
#include "HighLevel/pexSolver.h"
#include "HighLevel/kSolver.h"
#include "HighLevel/lcbsSolver.h"
#include "HighLevel/scalarizationSolver.h"
#include "ScalarizationWeights.h"
#include "Definitions.h"

std::unordered_map<size_t, std::vector<int>> id2coord;
std::string map_name;
std::ofstream output;

static void apply_solver_config(HighLevelSolver& h, const SolverConfig& cfg) {
    if (!cfg.objective_ordering.empty()) {
        h.set_objective_ordering(cfg.objective_ordering);
    }
    if (!cfg.scalarization_weights.empty()) {
        if (auto* s = dynamic_cast<scalarizationSolver*>(&h)) {
            s->set_scalarization_weights(cfg.scalarization_weights);
        }
    }
}

static std::unique_ptr<HighLevelSolver> make_high_level_solver(const SolverConfig& cfg,
                                                               size_t graph_size,
                                                               MergingStrategy ms)
{
    std::unique_ptr<HighLevelSolver> h;
    if (cfg.algorithm == "BBMOCBS-eps") {
        h = std::make_unique<epsSolver>(graph_size, cfg.agent_num, Algorithm::BBMOCBS_EPS,
                                        cfg.eager, cfg.dim, cfg.turn_dim, cfg.turn_cost, cfg.time_limit_sec);
        static_cast<epsSolver*>(h.get())->set_eps(cfg.eps);
    } else if (cfg.algorithm == "BBMOCBS-pex") {
        h = std::make_unique<pexSolver>(graph_size, cfg.agent_num, Algorithm::BBMOCBS_PEX,
                                        cfg.eager, cfg.dim, cfg.turn_dim, cfg.turn_cost, cfg.time_limit_sec);
        static_cast<pexSolver*>(h.get())->set_eps(cfg.eps);
        static_cast<pexSolver*>(h.get())->set_merging_strategy(ms);
    } else if (cfg.algorithm == "BBMOCBS-k") {
        h = std::make_unique<kSolver>(graph_size, cfg.agent_num, Algorithm::BBMOCBS_K,
                                      cfg.eager, cfg.dim, cfg.turn_dim, cfg.turn_cost, cfg.time_limit_sec);
        static_cast<kSolver*>(h.get())->set_merging_strategy(ms);
        static_cast<kSolver*>(h.get())->set_solution_num(cfg.solution_num);
    } else if (cfg.algorithm == "LCBS") {
        h = std::make_unique<lcbsSolver>(graph_size, cfg.agent_num, Algorithm::LCBS,
                                         cfg.eager, cfg.dim, cfg.turn_dim, cfg.turn_cost, cfg.time_limit_sec);
        static_cast<lcbsSolver*>(h.get())->set_eps(cfg.eps_i[0], cfg.eps_i[1], cfg.eps_i[2], cfg.eps_i[3], cfg.eps_i[4],
                                                   cfg.eps_i[5], cfg.eps_i[6], cfg.eps_i[7], cfg.eps_i[8], cfg.eps_i[9]);
        static_cast<lcbsSolver*>(h.get())->set_merging_strategy(ms);
        static_cast<lcbsSolver*>(h.get())->set_solution_num(std::min(1, cfg.solution_num));
    } else if (cfg.algorithm == "SCALARIZATION") {
        h = std::make_unique<scalarizationSolver>(graph_size, cfg.agent_num, Algorithm::SCALARIZATION,
                                                  cfg.eager, cfg.dim, cfg.turn_dim, cfg.turn_cost, cfg.time_limit_sec);
        static_cast<scalarizationSolver*>(h.get())->set_eps(cfg.eps_i[0], cfg.eps_i[1], cfg.eps_i[2], cfg.eps_i[3], cfg.eps_i[4],
                                                            cfg.eps_i[5], cfg.eps_i[6], cfg.eps_i[7], cfg.eps_i[8], cfg.eps_i[9]);
        static_cast<scalarizationSolver*>(h.get())->set_merging_strategy(ms);
        static_cast<scalarizationSolver*>(h.get())->set_solution_num(std::max(1, cfg.solution_num));
    }
    if (h) apply_solver_config(*h, cfg);
    return h;
}

MapfSolveResult MapfSolver::solve(const SolverConfig& cfg) {
    return solve_internal(cfg, nullptr);
}

MapfSolveResult MapfSolver::solve(const SolverConfig& cfg, const macussp::MapfProblem& problem) {
    return solve_internal(cfg, &problem);
}

MapfSolveResult MapfSolver::solve_internal(const SolverConfig& cfg,
                                           const macussp::MapfProblem* in_memory_problem)
{
    MapfSolveResult out;
    if (cfg.dim <= 0) {
        std::cerr << "[mapf_solver] dim must be >= 1.\n";
        return out;
    }
    if (static_cast<int>(cfg.cost_maps.size()) < cfg.dim && in_memory_problem == nullptr) {
        std::cerr << "[mapf_solver] cost_maps must have at least 'dim' entries.\n";
        return out;
    }

    LoggerPtr logger = nullptr;
    if (!cfg.logging_file.empty()) {
        logger = new Logger(cfg.logging_file);
    }

    if (!cfg.output_file.empty()) {
        output.open(cfg.output_file, std::ios::app);
        output.seekp(0, std::ios::end);
    }

    PreProcessor p;
    std::optional<Map> owned_map;
    const Map* active_map = nullptr;
    std::vector<Edge> edges;
    std::vector<std::pair<size_t, size_t>> start_goal;

    if (in_memory_problem != nullptr) {
        active_map = &in_memory_problem->map;
        edges = in_memory_problem->edges;
        start_goal = in_memory_problem->start_goal;
    } else {
        owned_map.emplace();
        Map& map = *owned_map;
        if (cfg.map_file.empty() || cfg.scenario_file.empty()) {
            std::cerr << "[mapf_solver] Required paths missing (map/scenario).\n";
            if (logger) delete logger;
            return out;
        }
        map_name = cfg.map_file;
        if (cfg.use_continuous) {
            std::vector<std::string> rows;
            p.read_map_rows(cfg.map_file, rows);
            int H = static_cast<int>(rows.size());
            int W = static_cast<int>(rows.front().size());
            auto grids = p.load_cost_grids(cfg.cost_maps, H, W);
            ContinuousDomain dom = build_continuous_from_raster(rows, grids);
            discretize_lattice_from_continuous(dom, SALP_SUBRES, SALP_ALLOW_DIAGONAL, map, edges, cfg.dim);
            id2coord.clear();
            id2coord.reserve(map.graph_size);
            for (int r = 0; r < map.height; ++r) {
                for (int c = 0; c < map.width; ++c) {
                    size_t id = map.getID(r, c);
                    if (id != static_cast<size_t>(-1)) {
                        id2coord[id] = {r, c};
                    }
                }
            }
            p.read_scenario(cfg.scenario_file, map, cfg.agent_num, start_goal, SALP_SUBRES);
        } else {
            p.read_map(map_name, map, id2coord);
            p.cost_init(map, edges, cfg.dim);
            for (int i = 0; i < cfg.dim; ++i) {
                p.read_cost(cfg.cost_maps[i], map, edges, i);
            }
            p.read_scenario(cfg.scenario_file, map, cfg.agent_num, start_goal);
        }
        active_map = &map;
    }

    MergingStrategy ms = cfg.conflict_based ? MergingStrategy::CONFLICT_BASED : MergingStrategy::MORE_SLACK;
    auto h_solver = make_high_level_solver(cfg, active_map->graph_size, ms);
    if (!h_solver) {
        if (output.is_open()) {
            output << "\n\n" << cfg.algorithm << " is not an allowed algorithm";
            output.close();
        }
        if (logger) delete logger;
        return out;
    }

    HSolutionID hsolution_ids;
    std::vector<CostVector> hsolution_costs;
    try {
        auto result = h_solver->run(edges, start_goal, hsolution_ids, hsolution_costs, logger);

        out.timings.hl_merging_sec = std::get<0>(result);
        out.timings.low_level_sec = std::get<1>(result);
        out.timings.total_sec = std::get<2>(result);
        out.success = std::get<5>(result);
        out.paths = std::get<6>(result);
        out.cost = std::get<7>(result);
    } catch (const std::exception& e) {
        std::cerr << "[mapf_solver] " << cfg.algorithm << " aborted: " << e.what() << "\n";
        out.success = false;
    } catch (...) {
        std::cerr << "[mapf_solver] " << cfg.algorithm << " aborted: unknown error\n";
        out.success = false;
    }

    if (output.is_open()) {
        time_t timestamp;
        time(&timestamp);
        output << "Algorithm = " << cfg.algorithm << "\n";
        output << "Agents = " << cfg.agent_num << "\n\n";
        output << std::ctime(&timestamp);
        output << "--------------------------\n";
        output.close();
    }
    if (logger) delete logger;
    return out;
}

std::tuple<bool,AgentPaths,CostVector> MapfSolver::mapf_solver(const SolverConfig& cfg)
{
    const MapfSolveResult r = solve(cfg);
    return std::make_tuple(r.success, r.paths, r.cost);
}

std::tuple<bool,AgentPaths,CostVector> MapfSolver::runSolver(const std::string& map_name_in,
                                                             int agent_count,
                                                             std::string scenarioFileName)
{
    SolverConfig cfg;
    cfg.algorithm = algo;
    cfg.time_limit_sec = time_lim_sec;
    cfg.dim = 3;
    cfg.agent_num = agent_count;
    cfg.map_file = "../maps/" + map_name_in + "/" + map_name_in + ".map";
    cfg.scenario_file = "../maps/" + map_name_in + "/scenario_files/" + scenarioFileName + ".scen";
    cfg.output_file = MapfSolver::makeOutputPath(map_name_in, time_lim_sec);
    cfg.cost_maps = {};
    cfg.conflict_based = true;
    cfg.eager = false;
    cfg.eps_i = {0,0,0,0,0,0,0,0,0,0};
    cfg.turn_dim = 0;
    cfg.turn_cost = 0;
    if (cfg.algorithm == "LCBS" || cfg.algorithm == "SCALARIZATION") {
        cfg.solution_num = 1;
    }
    cfg.use_continuous = ((map_name_in == "salp") || (map_name_in.find("salp") != std::string::npos))
                         && (SALP_SUBRES > 1);

    const MapfSolveResult r = solve(cfg);
    return std::make_tuple(r.success, r.paths, r.cost);
}

int MapfSolver::countSuccesses(const std::string& output_file) {
    std::ifstream infile(output_file);
    std::string line;
    int count = 0;
    while (std::getline(infile, line)) {
        if (line.find("SUCCESS") != std::string::npos) {
            ++count;
        }
    }
    return count;
}
