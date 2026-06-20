#include "pipeline/PipelineTypes.h"
#include "pipeline/CimopStage1Planner.h"
#include "ThetaTable.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " [--stage1 cimop|arvi|saia] [--stage2 lcbs|scalarization|bbmocbs-k|bbmocbs-pex]"
              << " [--domain salp|warehouse|forestfire]"
              << " [--robots N] [--scenario K] [--seed S] [--time_budget SEC]"
              << " [--redundant_pct PCT] [--output_csv PATH] [--entropy_trace PATH] [--label NAME]\n"
              << "Figure 6 pipelines (examples):\n"
              << "  Ours:   cimop + lcbs | B1: cimop + scalarization | B2: cimop + bbmocbs-k | B3: cimop + bbmocbs-pex\n"
              << "  B4-B7:  arvi  + stage2 | B8-B11: saia + stage2\n";
}

int main(int argc, char** argv) {
    macussp::PipelineConfig cfg;
    cfg.label = "Ours";
    cfg.stage1 = "cimop";
    cfg.stage2 = "lcbs";
    cfg.domain = "salp";
    cfg.robots = 5;
    cfg.scen_id = 0;
    cfg.seed = 0;
    cfg.time_budget_sec = 120;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--stage1") cfg.stage1 = need("--stage1");
        else if (arg == "--stage2") cfg.stage2 = need("--stage2");
        else if (arg == "--domain") cfg.domain = need("--domain");
        else if (arg == "--robots") cfg.robots = std::atoi(need("--robots"));
        else if (arg == "--scenario") cfg.scen_id = std::atoi(need("--scenario")) - 1;
        else if (arg == "--seed") cfg.seed = static_cast<unsigned>(std::strtoul(need("--seed"), nullptr, 10));
        else if (arg == "--time_budget" || arg == "--time_budget_seconds")
            cfg.time_budget_sec = std::atoi(need(arg.c_str()));
        else if (arg == "--redundant_pct" || arg == "--redundant_landmark_pct")
            cfg.redundant_landmark_pct = std::atof(need(arg.c_str()));
        else if (arg == "--output_csv") cfg.output_csv = need("--output_csv");
        else if (arg == "--entropy_trace") cfg.entropy_trace_path = need("--entropy_trace");
        else if (arg == "--label") cfg.label = need("--label");
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    auto s1 = macussp::make_stage1_planner(cfg.stage1);
    auto s2 = macussp::make_stage2_planner(cfg.stage2);
    if (!s1 || !s2) return 1;

    macussp::PipelineDriver driver(std::move(s1), std::move(s2));
    macussp::PipelineResultRow row = driver.run(cfg);

    const auto theta = macussp::theta_for_context(cfg.domain, row.inferred_context);

    std::cout << "[pipeline] label=" << row.label
              << " success=" << (row.success ? "true" : "false")
              << " timed_out=" << (row.timed_out ? "true" : "false")
              << " inferred_context=" << row.inferred_context
              << " theta=[" << theta[0] << "," << theta[1] << "," << theta[2] << "]"
              << " stage1_time=" << row.stage1_time_sec
              << " stage2_time=" << row.stage2_time_sec
              << " total_time=" << row.total_time_sec
              << " budget=" << row.time_budget_sec << "\n";
    std::cout << "[pipeline] joint_cost=[";
    for (size_t i = 0; i < row.joint_cost.size(); ++i) {
        if (i > 0) std::cout << ',';
        std::cout << row.joint_cost[i];
    }
    for (size_t i = row.joint_cost.size(); i < 3; ++i) {
        if (i > 0 || !row.joint_cost.empty()) std::cout << ',';
        std::cout << '0';
    }
    std::cout << "]\n";

    return row.success ? 0 : 1;
}
