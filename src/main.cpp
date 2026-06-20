#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <sstream>   // for path building
#include <cstdio>    // std::remove (optional)

// project
#include "environment.h"
#include "agent.h"
#include "state.h"
#include "animation.h"
#include "helper_utils.h"
#include "formation.h"
#include "mapf_solver.h"
#include "mapf_helper.h"
#include "high_level_planner.h"
#include "params.h"  // sim time limit for solver, true context, etc.
#include "InfoGathering/ARVI.h"

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
std::vector<std::string> SOLVERS = {"OURS" , "ARVI"};
std::vector<std::string> DOMAINS = {"salp", "forestfire", "warehouse"};
std::vector<int> NUM_AGENTS = {5,10,15,20,25,30,35}; // extend as needed
int domain_id = 0; // 0: salp, 1: forestfire, 2: warehouse

// Row for the per-solver aggregated output file
struct StepsRow {
    int    agents;
    int    steps_to_collapse;
    double phase1_time_sec;   // printed with 2 decimals
};

// --- tiny helper to ensure a directory exists (avoids <filesystem>) ----------
static void ensureDir(const std::string& dir) {
#ifdef _WIN32
    std::string cmd = "mkdir \"" + dir + "\" >NUL 2>&1";
#else
    std::string cmd = "mkdir -p \"" + dir + "\" >/dev/null 2>&1";
#endif
    std::system(cmd.c_str());
}

// -----------------------------------------------------------------------------
// Aggregated file (kept as before): ../results/<domain>_steps_vs_agents_<SOLVER>.txt
// -----------------------------------------------------------------------------
static void writeStepsVsAgentsFile(const std::string& solver,
                                   const std::vector<StepsRow>& rows)
{
    const std::string domain = DOMAINS[domain_id];
    const std::string base   = "../results/inference_steps_vs_agents";
    const std::string ddir   = base + "/" + domain;
    ensureDir("../results");
    ensureDir(base);
    ensureDir(ddir);

    const std::string out_path = ddir + "/" + domain + "_steps_vs_agents_" + solver + ".txt";

    // Optional: keep rows grouped by agent count for easy averaging later
    std::vector<StepsRow> sorted = rows;
    std::stable_sort(sorted.begin(), sorted.end(),
        [](const StepsRow& a, const StepsRow& b){
            if (a.agents != b.agents) return a.agents < b.agents;
            return a.phase1_time_sec < b.phase1_time_sec; // tie-breaker
        });

    std::ofstream out(out_path);
    if (!out.is_open()) {
        std::cerr << "[main] ERROR: cannot open " << out_path << " for writing.\n";
        return;
    }
    out << "#agents\tsteps_to_collapse\tphase1_time_sec\n";
    out << std::fixed << std::setprecision(2);
    for (const auto& r : sorted) {
        out << r.agents << "\t" << r.steps_to_collapse << "\t" << r.phase1_time_sec << "\n";
    }
    out.close();
    std::cout << "[main] Wrote: " << out_path << "\n";
}


// -----------------------------------------------------------------------------
// Per-scenario steps row APPEND:
//
// ../results/inference_steps_vs_num_agents/<domain>/<solver>/
//     <domain>_steps_vs_agents_<solver>_scen<k>.txt
//
// Appends one row per (#agents, scenario k).
// -----------------------------------------------------------------------------
static void appendStepsRowPerScenario(const std::string& domain,
                                      const std::string& solver,
                                      int scen_id,             // 0-based in code; file uses 1-based
                                      const StepsRow& row)
{
    std::ostringstream ddir;
    ddir << "../results/inference_steps_vs_num_agents/" << domain << "/" << solver;
    ensureDir("../results");
    ensureDir("../results/inference_steps_vs_num_agents");
    ensureDir("../results/inference_steps_vs_num_agents/" + domain);
    ensureDir(ddir.str());

    std::ostringstream fpath;
    fpath << ddir.str() << "/" << domain
          << "_steps_vs_agents_" << solver
          << "_scen" << (scen_id + 1) << ".txt";

    const std::string path = fpath.str();

    // If file is new, write header first.
    bool need_header = false;
    {
        std::ifstream test(path);
        if (!test.good()) need_header = true;
    }
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "[main] ERROR: cannot open " << path << " for appending.\n";
        return;
    }
    if (need_header) {
        out << "#agents\tsteps_to_collapse\tphase1_time_sec\n";
        out << std::fixed << std::setprecision(2);
    }
    out << row.agents << "\t" << row.steps_to_collapse << "\t"
        << std::fixed << std::setprecision(2) << row.phase1_time_sec << "\n";
    out.close();
    std::cout << "[main] Appended per-scenario row: " << path << "\n";
}

// -----------------------------------------------------------------------------
// Per-run belief entropy file:
//
// ../results/belief_entropy/<domain>/<solver>/
//     <domain>_belief_entropy_<solver>_A<agents>_scen<k>.txt
//
// One integer per line: |belief context set| at step t.
// -----------------------------------------------------------------------------
static void writeBeliefEntropyRunFile(const std::string& domain,
                                      const std::string& solver,
                                      int agents,
                                      int scen_id, // 0-based; filename uses 1-based
                                      const std::vector<std::set<int>>& contextTracking)
{
    std::ostringstream ddir;
    ddir << "../results/belief_entropy/" << domain << "/" << solver;
    ensureDir("../results");
    ensureDir("../results/belief_entropy");
    ensureDir("../results/belief_entropy/" + domain);
    ensureDir(ddir.str());

    std::ostringstream fpath;
    fpath << ddir.str() << "/"
          << domain << "_belief_entropy_" << solver
          << "_A" << agents
          << "_scen" << (scen_id + 1) << ".txt";

    const std::string out_path = fpath.str();
    std::ofstream out(out_path);
    if (!out.is_open()) {
        std::cerr << "[main] ERROR: cannot open " << out_path << " for writing.\n";
        return;
    }
    // Optional header for readability
    out << "# |belief_context_set| per step (domain=" << domain
        << ", solver=" << solver << ", agents=" << agents
        << ", scen=" << (scen_id + 1) << ")\n";
    for (const auto& S : contextTracking) {
        out << static_cast<int>(S.size()) << "\n";
    }
    out.close();
    std::cout << "[main] Wrote belief-entropy series: " << out_path << "\n";
}

int main(){
    std::unordered_map<int, double>       PlanningTimePerSolver; // key: solver_idx, value: sum over runs
    std::unordered_map<int, ARVICostVec>  PerformanceInObjectivesPerSolver;
    std::unordered_map<int, int>          Step2InferTrueContextPerSolver;
    std::unordered_map<int, std::vector<std::set<int>>> ContextSetTrackingPerSolver;

    // Aggregated rows per solver (kept as before)
    std::unordered_map<int, std::vector<StepsRow>> StepsVsAgentsPerSolver;

    for (int solver_idx = 0; solver_idx < (int)SOLVERS.size(); ++solver_idx) {
        const std::string SOLVER = SOLVERS[solver_idx];
        const std::string fixed_domain = DOMAINS[domain_id];
        auto& rows_for_solver = StepsVsAgentsPerSolver[solver_idx];
        rows_for_solver.clear();

        for (int numAgents : NUM_AGENTS) {
            for (int scen_id = 0; scen_id < 5; ++scen_id) { // 5 scenarios per numAgents
                int stepsToInferTrueContext = 0;

                std::string domain = fixed_domain;
                int numOfGroups = numAgents / ((domain == "salp")?5:(domain == "forestfire")?4:2);

                Environment env(domain, numOfGroups);
                env.getLandmarkStates2ContextSubsetMap_manually();

                std::cout << "\033[1;36m(Hidden) True context = " << TrueContext << "\033[0m" << std::endl;

                // init_agents now uses scen_id to pick salp1.scen ... salp5.scen, etc.
                std::vector<Agent> agents = init_agents(env, numAgents, scen_id);

                std::unordered_map<int, int> agentID2groupID =
                    high_level_planner::assignGroups_kmeans(env, agents, numOfGroups);

                std::unordered_map<int, std::vector<Agent>> groupsIdToAgentsVectorMap =
                    getGroupToAgentsVectorMap(agents, agentID2groupID);

                for (int g = 0; g < numOfGroups; ++g) {
                    std::cout << "[m.cpp l62] group " << g
                              << " has " << groupsIdToAgentsVectorMap[g].size() << " agents\n";
                }

                std::unordered_map<LocalState, std::string> landmarkState2FormationName =
                    env.getLandmarkStateToFormationMap();

                std::chrono::duration<double> timePhase1 = std::chrono::duration<double>::zero();

                std::vector<LocalState> landmark_sequence =
                    high_level_planner::getLandmarkInformativeSequence(env, SOLVER);

                std::cout << "\033[1;33m\n\n[Pre-check]Landmark visit sequence:\033[0m" << std::endl;
                std::unordered_map<LocalState, std::set<int>> landmarkToContextSubsetMap =
                    env.getLandmarkToContextSubsetMap();
                for (int i = 0; i < (int)landmark_sequence.size(); ++i) {
                    printLocalState(landmark_sequence[i]);
                    std::cout << " --> observed context subset: {";
                    for (const auto &context : landmarkToContextSubsetMap[landmark_sequence[i]]) {
                        std::cout << context << " ";
                    }
                    std::cout << "}\n";
                }

                // Build formations
                std::unordered_map<int, LocalState> groupIDToLandmarkState;
                bool beliefCollapsed = false;
                std::vector<Formation> formations;
                formations.reserve(groupsIdToAgentsVectorMap.size());
                for (int i = 0; i < (int)groupsIdToAgentsVectorMap.size(); i++) {
                    Formation formation(env, groupsIdToAgentsVectorMap[i], i, groupIDToLandmarkState[i]);
                    formations.push_back(formation);
                }

                // ---------------------- Phase 1 ----------------------
                auto startPhase1 = std::chrono::steady_clock::now();
                auto endPhase1   = std::chrono::steady_clock::now();
                std::cout << "\033[1;35m********[Phase 1] Running Belief Collapsing Phase...\033[0m" << std::endl;

                while (!beliefCollapsed){
                    groupIDToLandmarkState =
                        high_level_planner::getMapGroupIDToLandmarkState2(
                            groupIDToLandmarkState,
                            groupsIdToAgentsVectorMap,
                            landmark_sequence);

                    printGroupAssignmentInfo(groupsIdToAgentsVectorMap, groupIDToLandmarkState,
                                             landmarkState2FormationName, env);

                    for (int i = 0; i < (int)groupsIdToAgentsVectorMap.size(); ++i) {
                        formations[i].assignLandmark2Formation(groupIDToLandmarkState[i]);
                        printFormationDetails(formations[i]);
                    }

                    env.clearFireCache();
                    startPhase1 = std::chrono::steady_clock::now();
                    performBeliefCollapse(formations, env, SOLVER);
                    endPhase1 = std::chrono::steady_clock::now();
                    timePhase1 += endPhase1 - startPhase1;

                    beliefCollapsed = high_level_planner::hasBeliefCollapsed(formations);
                    if (beliefCollapsed) {
                        env.collectiveContextSetTracking =
                            high_level_planner::synchronizeFormationsTillCollapse(formations, env);
                        stepsToInferTrueContext = (int)env.collectiveContextSetTracking.size();
                    }
                }

                // Per-run belief entropy (distinct file per solver × agents × scenario)
                writeBeliefEntropyRunFile(fixed_domain, SOLVER, numAgents, scen_id,
                                          env.collectiveContextSetTracking);

                std::cout << "\n\n\033[1;34m[(" << SOLVER <<") " << numAgents
                          << " Agents]Avg. Time taken for Phase 1 (scen " << scen_id << "): "
                          << std::setprecision(2) << timePhase1.count() << " sec.\033[0m" << std::endl;

                // Update agents with trajectories from formations
                for (int i = 0; i < (int)formations.size(); ++i) {
                    for (Agent &agent : formations[i].agentsInFormation) {
                        agents[agent.getId()] = agent;
                    }
                }
                if (SOLVER == "ARVI") recomputeARVI_CostsFromFiles(formations, env);

                // Equal length sanity
                for (int i = 0; i < (int)agents.size(); ++i) {
                    if (agents[i].trajectory.size() != agents[0].trajectory.size()) {
                        std::cerr << "Error: Agent trajectories are not of equal length.\n";
                        return 0;
                    }
                }

                saveTrajectoryToFile(agents, env, "../results/MAPF_OUTPUT/trajectory.json");

                std::cout << "\n\n\033[1;34m[" << numAgents
                          << " Agents]Avg. Time taken for Phase 1: "
                          << std::fixed << std::setprecision(2)  << timePhase1.count()
                          << " seconds.\033[0m" << std::endl;

                auto planningTime = timePhase1; // phase 1 only
                std::cout << "DONE!\033[0m" << std::endl;

                // Accumulate solver totals (across all scenarios / agent counts)
                PlanningTimePerSolver[solver_idx] += (double)planningTime.count();
                PerformanceInObjectivesPerSolver[solver_idx][0] += env.TotalCostVector[0];
                PerformanceInObjectivesPerSolver[solver_idx][1] += env.TotalCostVector[1];
                PerformanceInObjectivesPerSolver[solver_idx][2] += env.TotalCostVector[2];
                Step2InferTrueContextPerSolver[solver_idx]       += stepsToInferTrueContext;

                // Keep last run's context set tracking (for your on-screen plotter)
                ContextSetTrackingPerSolver[solver_idx] = env.collectiveContextSetTracking;

                // Row for aggregated file
                StepsRow row{numAgents, stepsToInferTrueContext, planningTime.count()};
                rows_for_solver.push_back(row);

                // Append this row to the per-scenario file immediately
                appendStepsRowPerScenario(fixed_domain, SOLVER, scen_id, row);
            } // scen_id
        } // numAgents

        // Aggregated file per solver (kept as before)
        writeStepsVsAgentsFile(SOLVER, rows_for_solver);
    } // solver_idx

    // -------------------------- Summary printout ------------------------------
    std::cout << "\n\n\033[1;35m****************************** RESULTS ***************************************\033[0m" << std::endl;
    for (int solver_idx = 0; solver_idx < (int)SOLVERS.size(); ++solver_idx) {
        auto planningTime = PlanningTimePerSolver[solver_idx];
        auto performanceInObjectives = PerformanceInObjectivesPerSolver[solver_idx];
        auto step2InferTrueContext = Step2InferTrueContextPerSolver[solver_idx];
        std::cout << "\033[1;93;101m[" << SOLVERS[solver_idx] << "]\033[0m\033[1;34m\tPlanning time = \033[1;102m"
                  << planningTime << " sec\033[0m" << std::endl;
        std::cout << "\033[1;93;101m[" << SOLVERS[solver_idx] << "]\033[0m\033[1;34m\tTotal cost in objectives = \033[1;102m["
                  << performanceInObjectives[0] << " , " << performanceInObjectives[1] << " , "
                  << performanceInObjectives[2] << "]\033[0m" << std::endl;
        std::cout << "\033[1;93;101m[" << SOLVERS[solver_idx] << "]\033[0m\033[1;34m\tTotal Steps to infer context (summed over runs) = \033[1;102m"
                  << step2InferTrueContext << " steps\033[0m" << std::endl;
        std::cout << "\n\033[1;35m------------------------------------------------------------------------------\033[0m" << std::endl;
    }

    std::cout << "Plotting the context set tracking per solver...\n";
    plotContextSetTrackingFigure(ContextSetTrackingPerSolver, SOLVERS);
    std::cout << "\n\033[1;35m******************************************************************************\033[0m" << std::endl;

    return 0;
}
