#pragma once

#include "ObservationPhysics.h"
#include "pipeline/PipelineTypes.h"
#include "ThetaTable.h"
#include "TimingHarness.h"
#include "InfoGathering/ARVI.h"
#include "BeliefSet.h"
#include "environment.h"
#include "helper_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace macussp {

namespace detail {

inline std::vector<LocalState> blind_stencil_for_team(
    const LocalState& landmark,
    int team_size,
    const Environment& env,
    std::mt19937& rng)
{
    std::vector<LocalState> targets;
    if (team_size <= 0) return targets;

    if (team_size == 5 && env.getMapName() == "salp") {
        std::uniform_int_distribution<int> layout(0, 2);
        const int choice = layout(rng);
        if (choice == 0) {
            for (int i = 0; i < 5; ++i) {
                targets.push_back(env.getLocalState(landmark.x - 2 + i, landmark.y));
            }
        } else if (choice == 1) {
            for (int i = 0; i < 5; ++i) {
                targets.push_back(env.getLocalState(landmark.x, landmark.y - 2 + i));
            }
        } else {
            const std::array<std::pair<int, int>, 5> offs = {{{-1, -1}, {+1, -1}, {0, 0}, {-1, +1}, {+1, +1}}};
            for (const auto& o : offs) {
                targets.push_back(env.getLocalState(landmark.x + o.first, landmark.y + o.second));
            }
        }
        return targets;
    }

    std::vector<LocalState> candidates;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int x = landmark.x + dx;
            const int y = landmark.y + dy;
            if (!env.isWithinBounds(x, y)) continue;
            const LocalState s = env.getLocalState(x, y);
            if (env.isValidState(s)) candidates.push_back(s);
        }
    }
    std::shuffle(candidates.begin(), candidates.end(), rng);
    while (static_cast<int>(candidates.size()) < team_size) {
        candidates.push_back(landmark);
    }
    candidates.resize(static_cast<size_t>(team_size));
    return candidates;
}

struct BlindPolicy {
    LocalState landmark;
    std::vector<int> team_ids;
};

struct TeamMovePlan {
    double vi_cost{std::numeric_limits<double>::infinity()};
    std::unordered_map<int, std::vector<LocalState>> paths;
};

inline TeamMovePlan plan_team_with_vi(
    const std::vector<Agent>& agents_snapshot,
    const BlindPolicy& policy,
    Environment& env,
    std::mt19937& rng,
    const ARVIParams& params)
{
    TeamMovePlan plan;
    if (policy.team_ids.empty()) return plan;

    const auto stencil = blind_stencil_for_team(
        policy.landmark, static_cast<int>(policy.team_ids.size()), env, rng);

    double total_cost = 0.0;

    for (size_t idx = 0; idx < policy.team_ids.size(); ++idx) {
        const int agent_id = policy.team_ids[idx];
        const Agent& agent = agents_snapshot[static_cast<size_t>(agent_id)];
        const LocalState slot = stencil[std::min(idx, stencil.size() - 1)];
        const std::vector<LocalState> slots = {slot};

        ReservationTable solo_reservations;
        std::unordered_set<long long> solo_claimed;
        std::vector<LocalState> path = runARVIToAnySlot_VI(
            agent, slots, solo_claimed, env, solo_reservations, params);
        if (path.empty()) {
            path.push_back(agent.getState());
        }

        total_cost += static_cast<double>(path.size() > 0 ? path.size() - 1 : 0) * params.step_cost;
        plan.paths[agent_id] = std::move(path);
    }

    plan.vi_cost = total_cost;
    return plan;
}

inline void apply_team_plan(std::vector<Agent>& agents, const TeamMovePlan& plan) {
    for (const auto& kv : plan.paths) {
        agents[static_cast<size_t>(kv.first)].trajectory = kv.second;
        agents[static_cast<size_t>(kv.first)].setState(kv.second.back());
    }
}

inline BlindPolicy sample_random_policy(
    const Environment& env,
    const std::vector<Agent>& agents,
    std::mt19937& rng)
{
    const auto landmarks = env.getAllLandmarkLocalStates();
    BlindPolicy policy;
    if (landmarks.empty() || agents.empty()) return policy;

    std::uniform_int_distribution<int> pick_lm(0, static_cast<int>(landmarks.size()) - 1);
    policy.landmark = landmarks[static_cast<size_t>(pick_lm(rng))];

    const int n = static_cast<int>(agents.size());
    std::uniform_int_distribution<int> pick_size(1, n);
    const int team_size = pick_size(rng);

    std::vector<int> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    policy.team_ids.assign(ids.begin(), ids.begin() + team_size);
    return policy;
}

struct PolicySelection {
    BlindPolicy policy;
    TeamMovePlan plan;
};

inline PolicySelection select_policy_arvi(
    const Environment& env,
    const std::vector<Agent>& agents,
    std::mt19937& rng)
{
    PolicySelection selection;
    selection.policy = sample_random_policy(env, agents, rng);
    return selection;
}

inline PolicySelection select_policy_saia(
    const Environment& env,
    const std::vector<Agent>& agents,
    std::mt19937& rng)
{
    constexpr int kSamples = 32;
    std::vector<BlindPolicy> pool;
    pool.reserve(kSamples);
    for (int s = 0; s < kSamples; ++s) {
        pool.push_back(sample_random_policy(env, agents, rng));
    }
    std::uniform_int_distribution<int> pick(0, kSamples - 1);
    PolicySelection selection;
    selection.policy = pool[static_cast<size_t>(pick(rng))];
    return selection;
}

}  // namespace detail

inline Stage1Result run_baseline_belief_collapse_stage1(Environment& env,
                                                      std::vector<Agent>& agents,
                                                      const std::string& solver_tag,
                                                      PipelineDeadline& deadline,
                                                      unsigned seed = 0)
{
    Stage1Result result;
    result.robots_used = static_cast<int>(agents.size());

    TimingHarness timer;
    timer.start();

    std::set<int> belief = env.getPossibleContextSet();
    env.collectiveContextSetTracking.clear();
    env.collectiveContextSetTracking.push_back(belief);

    std::mt19937 rng(seed == 0 ? 1u : seed);
    int outer_round = 0;
    constexpr int kMaxOuterRounds = 300;

    ARVIParams exec_params;
    exec_params.max_time_cap = 60;
    exec_params.step_cost = 1.0;
    exec_params.gamma = 1.0;

    while (belief.size() > 1 && outer_round < kMaxOuterRounds) {
        if (deadline.expired()) {
            result.timed_out = true;
            break;
        }

        detail::PolicySelection selection;
        if (solver_tag == "SAIA") {
            selection = detail::select_policy_saia(env, agents, rng);
            if (!selection.policy.team_ids.empty()) {
                selection.plan = detail::plan_team_with_vi(
                    agents, selection.policy, env, rng, exec_params);
            }
        } else {
            selection = detail::select_policy_arvi(env, agents, rng);
            if (!selection.policy.team_ids.empty()) {
                selection.plan = detail::plan_team_with_vi(
                    agents, selection.policy, env, rng, exec_params);
            }
        }
        const detail::BlindPolicy& policy = selection.policy;
        if (policy.team_ids.empty()) break;

        detail::apply_team_plan(agents, selection.plan);

        std::vector<LocalState> team_positions;
        team_positions.reserve(policy.team_ids.size());
        for (int id : policy.team_ids) {
            team_positions.push_back(agents[static_cast<size_t>(id)].getState());
        }

        belief = observation_physics::apply_baseline_landmark_observation(
            belief, team_positions, policy.landmark, env);
        env.collectiveContextSetTracking.push_back(belief);
        ++outer_round;

        if (belief.size() == 1) break;
    }

    result.entropy_trace = env.collectiveContextSetTracking;
    result.cumulative_entropy = cumulative_entropy_cardinality(result.entropy_trace);
    result.stage1_steps = static_cast<int>(result.entropy_trace.size());

    if (!result.timed_out && !result.entropy_trace.empty() && result.entropy_trace.back().size() == 1) {
        result.inferred_context = *result.entropy_trace.back().begin();
    }
    result.ordering = theta_for_context(env.getMapName(), result.inferred_context);
    result.wall_time_sec = timer.elapsed_sec();
    return result;
}

}  // namespace macussp
