#ifndef MAPF_SOLVER_H
#define MAPF_SOLVER_H

#include <string>
#include <vector>
#include <iostream>
#include <array>
#include "Definitions.h"
#include "TimingHarness.h"
#include "LexCompare.h"

struct MapfSolveResult {
    bool success{false};
    AgentPaths paths;
    CostVector cost;
    macussp::SolverTimings timings;
};

// Returns true on success, false otherwise.
// This bundles what used to be command-line flags.
struct SolverConfig {
    // Required for file-based path
    std::string map_file;
    std::string scenario_file;
    int agent_num = 5;
    int dim = 2;
    std::vector<std::string> cost_maps;

    std::string algorithm = "LCBS";

    double eps = 0.0;
    std::array<double,10> eps_i{{0,0,0,0,0,0,0,0,0,0}};

    bool conflict_based = true;
    bool eager = true;

    int turn_dim = 0;
    int turn_cost = 0;

    int solution_num = 1;
    int time_limit_sec = 120;

    std::string output_file;
    std::string logging_file;

    bool use_continuous = false;

    macussp::ObjectiveOrdering objective_ordering;
    std::vector<double> scalarization_weights;
};

namespace macussp { struct MapfProblem; }

class MapfSolver {
public:
    explicit MapfSolver(const std::string& algorithm, int time_limit_sec, int dim = 3)
        : algo(algorithm), time_lim_sec(time_limit_sec), dim(dim)
    {
        if (algo != "LCBS" && algo != "BBMOCBS-k" && algo != "BBMOCBS-pex" && algo != "BBMOCBS-eps" && algo != "SCALARIZATION") {
            std::cerr << "Invalid algorithm specified: " << algo << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::cout << "\n----\n";
        std::cout << "Initialized MAPF solver: \033[1;33m" << algo << "\033[0m for \033[1;33m" << dim
                  << "\033[0m objectives for \033[1;33m" << time_limit_sec << " sec.\033[0m" << std::endl;
    }

    std::tuple<bool,AgentPaths,CostVector> runSolver(const std::string& map_name, int agent_count, std::string scenarioFileName);

    MapfSolveResult solve(const SolverConfig& cfg);
    MapfSolveResult solve(const SolverConfig& cfg, const macussp::MapfProblem& problem);

    std::tuple<bool,AgentPaths,CostVector> mapf_solver(const SolverConfig& cfg);

    static int countSuccesses(const std::string& output_file);

    static std::string makeOutputPath(const std::string& map_name, int time_lim_sec) {
        return "../results/MAPF_LOG/log_" + map_name + "_" + std::to_string(time_lim_sec) + "sec.txt";
    };

    std::string algo;
    int time_lim_sec;
    int dim;

private:
    MapfSolveResult solve_internal(const SolverConfig& cfg,
                                   const macussp::MapfProblem* in_memory_problem);
};

#endif // MAPF_SOLVER_H
