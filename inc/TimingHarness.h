#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>

namespace macussp {

class TimingHarness {
public:
    using Clock = std::chrono::steady_clock;

    void start() { start_ = Clock::now(); }

    double elapsed_sec() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

    static double elapsed_sec(const Clock::time_point& t0) {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }

private:
    Clock::time_point start_{Clock::now()};
};

// Shared wall-clock budget for one full pipeline run (Stage 1 + Stage 2).
class PipelineDeadline {
public:
    explicit PipelineDeadline(double budget_sec)
        : budget_sec_(std::max(0.0, budget_sec)), start_(TimingHarness::Clock::now()) {}

    double budget_sec() const { return budget_sec_; }

    double elapsed_sec() const {
        return TimingHarness::elapsed_sec(start_);
    }

    double remaining_sec() const {
        return std::max(0.0, budget_sec_ - elapsed_sec());
    }

    bool expired() const { return elapsed_sec() >= budget_sec_; }

    int remaining_sec_ceiled() const {
        return static_cast<int>(std::ceil(remaining_sec()));
    }

    // Per-MAPF-call limit: at least 1 second while budget remains.
    int mapf_time_limit_sec() const {
        if (expired()) return 1;
        return std::max(1, remaining_sec_ceiled());
    }

private:
    double budget_sec_;
    TimingHarness::Clock::time_point start_;
};

struct SolverTimings {
    double hl_merging_sec{0.0};
    double low_level_sec{0.0};
    double total_sec{0.0};
};

}  // namespace macussp
