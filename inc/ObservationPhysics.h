#pragma once

#include "BeliefSet.h"
#include "environment.h"
#include "state.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <string>
#include <vector>

namespace macussp {
namespace observation_physics {

inline int required_team_size(const Environment& env) {
    const std::string& d = env.getMapName();
    if (d == "salp") return 5;
    if (d == "forestfire") return 4;
    if (d == "warehouse") return 2;
    return static_cast<int>(env.getAllLandmarkLocalStates().size());
}

inline std::string required_formation_name(const Environment& env, const LocalState& landmark) {
    const auto& m = env.getLandmarkStateToFormationMap();
    const auto it = m.find(landmark);
    if (it == m.end()) return {};
    return it->second;
}

inline std::vector<LocalState> required_stencil_slots(
    const LocalState& landmark,
    const std::string& formation_name,
    int team_size,
    const Environment& env)
{
    std::vector<LocalState> slots;
    const int n = team_size;
    const int x0 = landmark.x;
    const int y0 = landmark.y;

    if (env.getMapName() == "salp") {
        if (formation_name == "chain") {
            const int off = (n - 1) / 2;
            for (int i = 0; i < n; ++i) {
                slots.push_back(env.getLocalState(x0 - off + i, y0));
            }
        } else if (formation_name == "cross" && n == 5) {
            const std::array<std::pair<int, int>, 5> offs = {{{-1, -1}, {+1, -1}, {0, 0}, {-1, +1}, {+1, +1}}};
            for (const auto& o : offs) {
                slots.push_back(env.getLocalState(x0 + o.first, y0 + o.second));
            }
        }
    } else if (env.getMapName() == "warehouse") {
        for (int k = 1; k <= n; ++k) {
            slots.push_back(env.getLocalState(x0, y0 + k));
        }
    } else if (env.getMapName() == "forestfire") {
        slots.push_back(env.getLocalState(x0, y0 - 1));
        slots.push_back(env.getLocalState(x0, y0 + 1));
        slots.push_back(env.getLocalState(x0 - 1, y0));
        slots.push_back(env.getLocalState(x0 + 1, y0));
    }
    return slots;
}

inline bool positions_match_stencil(
    const std::vector<LocalState>& team_positions,
    const std::vector<LocalState>& required_slots)
{
    if (team_positions.size() != required_slots.size() || required_slots.empty()) {
        return false;
    }
    auto norm = [](LocalState s) {
        return std::make_pair(s.x, s.y);
    };
    std::vector<std::pair<int, int>> a;
    std::vector<std::pair<int, int>> b;
    a.reserve(team_positions.size());
    b.reserve(required_slots.size());
    for (const auto& s : team_positions) a.push_back(norm(s));
    for (const auto& s : required_slots) b.push_back(norm(s));
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

inline bool team_satisfies_landmark_requirements(
    const std::vector<LocalState>& team_positions,
    const LocalState& landmark,
    const Environment& env)
{
    const int req_size = required_team_size(env);
    if (static_cast<int>(team_positions.size()) != req_size) {
        return false;
    }
    const std::string req_form = required_formation_name(env, landmark);
    if (req_form.empty()) return false;
    const auto slots = required_stencil_slots(landmark, req_form, req_size, env);
    return positions_match_stencil(team_positions, slots);
}

// Baseline observe: no information unless team size and formation match ground truth.
inline std::set<int> apply_baseline_landmark_observation(
    const std::set<int>& prior_belief,
    const std::vector<LocalState>& team_positions,
    const LocalState& landmark,
    const Environment& env)
{
    if (!team_satisfies_landmark_requirements(team_positions, landmark, env)) {
        return prior_belief;
    }
    const auto& lm_map = env.getLandmarkToContextSubsetMap();
    const auto it = lm_map.find(landmark);
    if (it == lm_map.end()) return prior_belief;
    return belief_intersect(prior_belief, it->second);
}

}  // namespace observation_physics
}  // namespace macussp
