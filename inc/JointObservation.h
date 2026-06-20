#pragma once

#include "BeliefSet.h"
#include "environment.h"
#include "action.h"
#include "state.h"

#include <set>
#include <string>
#include <vector>

namespace macussp {

// Shared modified joint observation after synchronized "observe".
inline std::set<int> apply_joint_landmark_observation(
    const std::set<int>& prior_belief,
    const JointState& joint_after_action,
    const LocalState& formation_landmark,
    const Environment& env)
{
    const auto& lm_map = env.getLandmarkToContextSubsetMap();
    if (env.getMapName() == "forestfire") {
        std::set<int> belief = prior_belief;
        for (const auto& agent_state : joint_after_action.states) {
            const auto it = lm_map.find(agent_state);
            if (it == lm_map.end()) {
                return std::set<int>{};
            }
            belief = belief_intersect(belief, it->second);
        }
        return belief;
    }

    const auto it = lm_map.find(formation_landmark);
    if (it == lm_map.end()) return prior_belief;
    return belief_intersect(prior_belief, it->second);
}

inline AugmentedState apply_synchronized_observe_step(
    const AugmentedState& prior,
    const std::vector<std::string>& joint_action,
    const LocalState& formation_landmark,
    const Environment& env)
{
    JointState next_joint = do_joint_action(prior.joint, joint_action, env);
    std::set<int> next_belief = prior.contextSet;
    const bool all_observe = !joint_action.empty() &&
        std::all_of(joint_action.begin(), joint_action.end(),
                    [](const std::string& a) { return a == "observe"; });
    if (all_observe) {
        next_belief = apply_joint_landmark_observation(
            prior.contextSet, next_joint, formation_landmark, env);
    }
    return AugmentedState(next_joint, next_belief);
}

}  // namespace macussp
