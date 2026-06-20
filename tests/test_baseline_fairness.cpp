#include "pipeline/Stage1Loop.h"
#include "pipeline/PipelineTypes.h"
#include "environment.h"
#include "helper_utils.h"
#include "params.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond "\n"; return 1; } } while (0)

struct Stats {
    double mean_steps{0};
    double mean_time{0};
    int c2_hits{0};
    int n{0};
};

static Stats run_solver(const std::string& tag, int seed, int budget_sec) {
    Environment env("salp", 1);
    env.loadManualLandmarkToContextSubsetMap();
    std::vector<Agent> agents = init_agents(env, 5, 0);

    macussp::PipelineDeadline deadline(static_cast<double>(budget_sec));
    macussp::Stage1Result r = macussp::run_belief_collapse_stage1(
        env, agents, tag, deadline, static_cast<unsigned>(seed));

    Stats s;
    s.n = 1;
    s.mean_steps = static_cast<double>(r.stage1_steps);
    s.mean_time = r.wall_time_sec;
    if (!r.timed_out && r.inferred_context == TrueContext) s.c2_hits = 1;
    std::cout << "  [" << tag << " seed=" << seed << "] steps=" << r.stage1_steps
              << " time=" << r.wall_time_sec << " c2=" << (s.c2_hits ? 1 : 0)
              << " timeout=" << (r.timed_out ? 1 : 0) << "\n";
    return s;
}

static Stats run_pipeline_stage1_arvi(int seed, int budget_sec) {
    macussp::PipelineConfig cfg;
    cfg.stage1 = "arvi";
    cfg.stage2 = "lcbs";
    cfg.domain = "salp";
    cfg.robots = 5;
    cfg.scen_id = 0;
    cfg.seed = static_cast<unsigned>(seed);
    cfg.time_budget_sec = budget_sec;

    macussp::PipelineDeadline deadline(static_cast<double>(budget_sec));
    const int numOfGroups = cfg.robots / 5;
    Environment env(cfg.domain, numOfGroups);
    env.loadManualLandmarkToContextSubsetMap();
    std::vector<Agent> agents = init_agents(env, cfg.robots, cfg.scen_id);
    auto s1 = macussp::make_stage1_planner(cfg.stage1);
    macussp::Stage1Input input{env, agents, cfg.scen_id, cfg.seed, deadline};
    macussp::Stage1Result r = s1->run(input);

    Stats s;
    s.n = 1;
    s.mean_steps = static_cast<double>(r.stage1_steps);
    s.mean_time = r.wall_time_sec;
    if (!r.timed_out && r.inferred_context == TrueContext) s.c2_hits = 1;
    std::cout << "  [pipeline ARVI seed=" << seed << "] steps=" << r.stage1_steps
              << " time=" << r.wall_time_sec << " c2=" << (s.c2_hits ? 1 : 0)
              << " timeout=" << (r.timed_out ? 1 : 0) << "\n";
    return s;
}

static Stats merge(const Stats& a, const Stats& b) {
    Stats out;
    out.n = a.n + b.n;
    out.mean_steps = (a.mean_steps * a.n + b.mean_steps * b.n) / out.n;
    out.mean_time = (a.mean_time * a.n + b.mean_time * b.n) / out.n;
    out.c2_hits = a.c2_hits + b.c2_hits;
    return out;
}

int main(int argc, char** argv) {
  (void)argc; (void)argv;
  const int budget = 120;
  const int seeds[] = {0, 1, 2, 3, 4};

  Stats cimop{};
  Stats arvi{};
  Stats saia{};
  for (int seed : seeds) {
      cimop = merge(cimop, run_solver("OURS", seed, budget));
      arvi = merge(arvi, run_solver("ARVI", seed, budget));
      saia = merge(saia, run_solver("SAIA", seed, budget));
  }

  (void)run_pipeline_stage1_arvi(0, budget);

  std::cout << "baseline_fairness gate (salp 5 robots scen1 seeds 0-4, budget=" << budget << "s)\n";
  std::cout << "  CIMOP mean_steps=" << cimop.mean_steps << " mean_time=" << cimop.mean_time
            << " c2=" << cimop.c2_hits << "/" << cimop.n << "\n";
  std::cout << "  ARVI  mean_steps=" << arvi.mean_steps << " mean_time=" << arvi.mean_time
            << " c2=" << arvi.c2_hits << "/" << arvi.n << "\n";
  std::cout << "  SAIA  mean_steps=" << saia.mean_steps << " mean_time=" << saia.mean_time
            << " c2=" << saia.c2_hits << "/" << saia.n << "\n";

  CHECK(cimop.c2_hits == cimop.n);
  CHECK(arvi.c2_hits == arvi.n);
  CHECK(saia.c2_hits == saia.n);
  CHECK(cimop.mean_steps < arvi.mean_steps);
  CHECK(cimop.mean_steps < saia.mean_steps);

  std::cout << "test_baseline_fairness: PASS (CIMOP ahead by "
            << (arvi.mean_steps - cimop.mean_steps) << " vs ARVI, "
            << (saia.mean_steps - cimop.mean_steps) << " vs SAIA steps)\n";
  return 0;
}
