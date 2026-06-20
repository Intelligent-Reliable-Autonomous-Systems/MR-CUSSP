#pragma once

#include "LexCompare.h"
#include "TimingHarness.h"
#include "Definitions.h"
#include "agent.h"
#include "state.h"
#include <set>
#include <string>
#include <vector>

class Environment;

namespace macussp {

struct Stage1Input {
    Environment& env;
    std::vector<Agent>& agents;
    int scen_id{0};
    unsigned seed{0};
    PipelineDeadline& deadline;
};

struct Stage1Result {
    int inferred_context{-1};
    ObjectiveOrdering ordering;
    std::vector<std::set<int>> entropy_trace;
    double cumulative_entropy{0.0};
    double wall_time_sec{0.0};
    int robots_used{0};
    int stage1_steps{0};
    bool timed_out{false};
};

class Stage1Planner {
public:
    virtual ~Stage1Planner() = default;
    virtual std::string name() const = 0;
    virtual Stage1Result run(const Stage1Input& input) = 0;
};

struct Stage2Input {
    Environment& env;
    std::vector<Agent>& agents;
    ObjectiveOrdering ordering;
    int inferred_context{-1};
    PipelineDeadline& deadline;
};

struct Stage2Result {
    bool success{false};
    CostVector joint_cost;
    double wall_time_sec{0.0};
    SolverTimings solver_timings;
    bool timed_out{false};
};

class Stage2Planner {
public:
    virtual ~Stage2Planner() = default;
    virtual std::string name() const = 0;
    virtual Stage2Result run(const Stage2Input& input) = 0;
};

struct PipelineConfig {
    std::string label{"Ours"};
    std::string stage1{"cimop"};
    std::string stage2{"lcbs"};
    std::string domain{"salp"};
    int robots{5};
    int scen_id{0};
    unsigned seed{0};
    int time_budget_sec{120};
    double redundant_landmark_pct{0.0};
    std::string output_csv;
    std::string entropy_trace_path;
};

struct PipelineResultRow {
    std::string label;
    std::string stage1;
    std::string stage2;
    std::string domain;
    int robots{0};
    int scen_id{0};
    unsigned seed{0};
    int time_budget_sec{0};
    double redundant_pct{0.0};
    int inferred_context{-1};
    CostVector joint_cost;
    double cumulative_entropy{0.0};
    int stage1_steps{0};
    double stage1_time_sec{0.0};
    double stage2_time_sec{0.0};
    double total_time_sec{0.0};
    bool success{false};
    bool timed_out{false};
};

class PipelineDriver {
public:
    PipelineDriver(std::unique_ptr<Stage1Planner> s1, std::unique_ptr<Stage2Planner> s2);

    PipelineResultRow run(const PipelineConfig& cfg);

private:
    std::unique_ptr<Stage1Planner> stage1_;
    std::unique_ptr<Stage2Planner> stage2_;
};

std::unique_ptr<Stage1Planner> make_stage1_planner(const std::string& name);
std::unique_ptr<Stage2Planner> make_stage2_planner(const std::string& name);

void append_pipeline_csv_row(const std::string& path, const PipelineResultRow& row, bool write_header);

}  // namespace macussp
