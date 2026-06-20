#include "pipeline/PipelineTypes.h"
#include "pipeline/CimopStage1Planner.h"
#include "pipeline/ArviStage1Planner.h"
#include "pipeline/SaiaStage1Planner.h"
#include "pipeline/MapfStage2Planner.h"

#include "environment.h"
#include "helper_utils.h"

#include <fstream>
#include <iostream>
#include <memory>

namespace macussp {

namespace {

void write_entropy_trace_file(const std::string& path,
                              const std::vector<std::set<int>>& trace)
{
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) {
        std::cerr << "[pipeline] cannot open entropy trace: " << path << "\n";
        return;
    }
    for (const auto& belief : trace) {
        out << belief.size() << '\n';
    }
}

}  // namespace

std::unique_ptr<Stage1Planner> make_stage1_planner(const std::string& name) {
    if (name == "cimop" || name == "ours" || name == "CIMOP") {
        return make_cimop_stage1_planner();
    }
    if (name == "arvi" || name == "ARVI") {
        return make_arvi_stage1_planner();
    }
    if (name == "saia" || name == "SAIA") {
        return make_saia_stage1_planner();
    }
    std::cerr << "[pipeline] Unknown stage1: " << name << "\n";
    return nullptr;
}

std::unique_ptr<Stage2Planner> make_stage2_planner(const std::string& name) {
    auto p = make_mapf_stage2_planner(name);
    if (p) return p;
    std::cerr << "[pipeline] Unknown stage2: " << name << "\n";
    return nullptr;
}

PipelineDriver::PipelineDriver(std::unique_ptr<Stage1Planner> s1, std::unique_ptr<Stage2Planner> s2)
    : stage1_(std::move(s1)), stage2_(std::move(s2)) {}

PipelineResultRow PipelineDriver::run(const PipelineConfig& cfg) {
    PipelineResultRow row;
    row.label = cfg.label;
    row.stage1 = cfg.stage1;
    row.stage2 = cfg.stage2;
    row.domain = cfg.domain;
    row.robots = cfg.robots;
    row.scen_id = cfg.scen_id;
    row.seed = cfg.seed;
    row.time_budget_sec = cfg.time_budget_sec;
    row.redundant_pct = cfg.redundant_landmark_pct;

    PipelineDeadline deadline(static_cast<double>(cfg.time_budget_sec));

    const int numOfGroups = cfg.robots / ((cfg.domain == "salp") ? 5 : (cfg.domain == "forestfire") ? 4 : 2);
    Environment env(cfg.domain, numOfGroups);
    env.loadManualLandmarkToContextSubsetMap();
    env.applyRedundantLandmarkPct(cfg.redundant_landmark_pct, cfg.seed);

    std::vector<Agent> agents = init_agents(env, cfg.robots, cfg.scen_id);

    Stage1Input s1in{env, agents, cfg.scen_id, cfg.seed, deadline};
    Stage1Result s1out = stage1_->run(s1in);

    row.inferred_context = s1out.inferred_context;
    row.cumulative_entropy = s1out.cumulative_entropy;
    row.stage1_steps = s1out.stage1_steps;
    row.stage1_time_sec = s1out.wall_time_sec;
    row.timed_out = s1out.timed_out;

    write_entropy_trace_file(cfg.entropy_trace_path, s1out.entropy_trace);

    if (s1out.timed_out || deadline.expired()) {
        row.timed_out = true;
        row.success = false;
        row.total_time_sec = deadline.elapsed_sec();
        if (!cfg.output_csv.empty()) {
            append_pipeline_csv_row(cfg.output_csv, row, true);
        }
        return row;
    }

    Stage2Input s2in{env, agents, s1out.ordering, s1out.inferred_context, deadline};
    Stage2Result s2out = stage2_->run(s2in);

    row.joint_cost = s2out.joint_cost;
    row.stage2_time_sec = s2out.wall_time_sec;
    row.total_time_sec = deadline.elapsed_sec();
    row.timed_out = s2out.timed_out || deadline.expired();
    row.success = s2out.success && !row.timed_out;

    if (!cfg.output_csv.empty()) {
        append_pipeline_csv_row(cfg.output_csv, row, true);
    }

    return row;
}

void append_pipeline_csv_row(const std::string& path, const PipelineResultRow& row, bool write_header) {
    bool need_header = write_header;
    {
        std::ifstream test(path);
        if (test.good()) need_header = false;
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::cerr << "[pipeline] cannot open csv: " << path << "\n";
        return;
    }
    if (need_header) {
        out << "label,stage1,stage2,domain,robots,scenario,seed,time_budget,redundant_pct,inferred_context,"
               "joint_cost_0,joint_cost_1,joint_cost_2,cumulative_entropy,stage1_steps,"
               "stage1_time,stage2_time,total_time,success,timed_out\n";
    }
    out << row.label << ',' << row.stage1 << ',' << row.stage2 << ',' << row.domain << ','
        << row.robots << ',' << (row.scen_id + 1) << ',' << row.seed << ',' << row.time_budget_sec << ','
        << row.redundant_pct << ',' << row.inferred_context << ',';
    for (size_t i = 0; i < 3; ++i) {
        if (i < row.joint_cost.size()) out << row.joint_cost[i];
        else out << 0;
        if (i < 2) out << ',';
    }
    out << ',' << row.cumulative_entropy << ',' << row.stage1_steps << ','
        << row.stage1_time_sec << ',' << row.stage2_time_sec << ',' << row.total_time_sec << ','
        << (row.success ? 1 : 0) << ',' << (row.timed_out ? 1 : 0) << '\n';
}

}  // namespace macussp
