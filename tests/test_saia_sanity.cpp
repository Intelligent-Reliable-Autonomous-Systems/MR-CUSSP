#include "pipeline/Stage1Loop.h"
#include "ThetaTable.h"
#include "environment.h"
#include "helper_utils.h"
#include "params.h"

#include <iostream>
#include <string>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond "\n"; return 1; } } while (0)

static int run_fast_checks() {
    Environment env("salp", 1);
    env.loadManualLandmarkToContextSubsetMap();
    const auto cimop_seq = high_level_planner::getLandmarkInformativeSequence(env, "OURS");
    const auto saia_seq = high_level_planner::getLandmarkVisitSequence_SAIA(env, 0);
    CHECK(!cimop_seq.empty());
    CHECK(!saia_seq.empty());
    std::cout << "test_saia_sanity: fast OK (landmark seq CIMOP=" << cimop_seq.size()
              << " SAIA=" << saia_seq.size() << ")\n";
    return 0;
}

static macussp::Stage1Result run_stage1(const std::string& tag, unsigned seed) {
    const int robots = 5;
    const int scen_id = 0;
    const int budget_sec = 120;

    Environment env("salp", robots / 5);
    env.loadManualLandmarkToContextSubsetMap();
    std::vector<Agent> agents = init_agents(env, robots, scen_id);

    macussp::PipelineDeadline deadline(static_cast<double>(budget_sec));
    return macussp::run_belief_collapse_stage1(env, agents, tag, deadline, seed);
}

static int run_slow_checks() {
    const int true_context = TrueContext;
    const auto cimop = run_stage1("OURS", 0);
    CHECK(!cimop.timed_out);
    CHECK(cimop.inferred_context == true_context);
    CHECK(cimop.ordering == (macussp::ObjectiveOrdering{1, 0, 2}));

    const auto arvi = run_stage1("ARVI", 0);
    CHECK(!arvi.timed_out);
    CHECK(arvi.inferred_context == true_context);

    const auto saia = run_stage1("SAIA", 0);
    CHECK(!saia.timed_out);
    CHECK(saia.inferred_context == true_context);

    CHECK(cimop.stage1_steps < arvi.stage1_steps);
    CHECK(cimop.stage1_steps < saia.stage1_steps);

    std::cout << "test_saia_sanity: slow OK (CIMOP steps=" << cimop.stage1_steps
              << ", ARVI steps=" << arvi.stage1_steps
              << ", SAIA steps=" << saia.stage1_steps
              << ", inferred c" << cimop.inferred_context << ")\n";
    return 0;
}

int main(int argc, char** argv) {
    bool run_slow = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--slow") run_slow = true;
    }

    if (!run_slow) {
        return run_fast_checks();
    }
    return run_slow_checks();
}
