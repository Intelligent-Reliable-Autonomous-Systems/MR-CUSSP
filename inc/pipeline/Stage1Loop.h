#pragma once

#include "pipeline/PipelineTypes.h"
#include "pipeline/BaselineStage1.h"
#include "ThetaTable.h"
#include "TimingHarness.h"
#include "params.h"

#include "environment.h"
#include "formation.h"
#include "helper_utils.h"
#include "high_level_planner.h"
#include "BeliefSet.h"

#include <string>
#include <vector>

namespace macussp {

class MapfTimeLimitScope {
public:
    explicit MapfTimeLimitScope(int limit_sec) : prev_(g_active_mapf_time_limit_sec) {
        g_active_mapf_time_limit_sec = limit_sec;
    }
    ~MapfTimeLimitScope() { g_active_mapf_time_limit_sec = prev_; }

private:
    int prev_;
};

inline Stage1Result run_belief_collapse_stage1(Environment& env,
                                               std::vector<Agent>& agents,
                                               const std::string& solver_tag,
                                               PipelineDeadline& deadline,
                                               unsigned seed = 0)
{
    if (solver_tag == "ARVI" || solver_tag == "SAIA") {
        return run_baseline_belief_collapse_stage1(env, agents, solver_tag, deadline, seed);
    }

    Stage1Result result;
    result.robots_used = static_cast<int>(agents.size());

    TimingHarness timer;
    timer.start();

    const int numAgents = static_cast<int>(agents.size());
    const std::string domain = env.getMapName();
    const int numOfGroups = numAgents / ((domain == "salp") ? 5 : (domain == "forestfire") ? 4 : 2);

    auto agentID2groupID = high_level_planner::assignGroups_kmeans(env, agents, numOfGroups);
    auto groupsIdToAgentsVectorMap = getGroupToAgentsVectorMap(agents, agentID2groupID);

    std::vector<LocalState> landmark_sequence;
    if (solver_tag == "SAIA") {
        landmark_sequence = high_level_planner::getLandmarkVisitSequence_SAIA(env, seed);
    } else {
        landmark_sequence = high_level_planner::getLandmarkInformativeSequence(env, solver_tag);
    }

    std::unordered_map<int, LocalState> groupIDToLandmarkState;
    std::vector<Formation> formations;
    formations.reserve(groupsIdToAgentsVectorMap.size());
    for (int i = 0; i < static_cast<int>(groupsIdToAgentsVectorMap.size()); ++i) {
        formations.emplace_back(env, groupsIdToAgentsVectorMap[i], i, groupIDToLandmarkState[i]);
    }

    bool beliefCollapsed = false;
    while (!beliefCollapsed) {
        if (deadline.expired()) {
            result.timed_out = true;
            break;
        }

        groupIDToLandmarkState = high_level_planner::getMapGroupIDToLandmarkState2(
            groupIDToLandmarkState, groupsIdToAgentsVectorMap, landmark_sequence);

        for (size_t i = 0; i < groupsIdToAgentsVectorMap.size(); ++i) {
            formations[i].assignLandmark2Formation(groupIDToLandmarkState[static_cast<int>(i)]);
        }

        env.clearFireCache();
        {
            MapfTimeLimitScope mapf_limit(deadline.mapf_time_limit_sec());
            performBeliefCollapse(formations, env, solver_tag);
        }
        if (deadline.expired()) {
            result.timed_out = true;
            break;
        }

        beliefCollapsed = high_level_planner::hasBeliefCollapsed(formations);
        if (beliefCollapsed) {
            env.collectiveContextSetTracking =
                high_level_planner::synchronizeFormationsTillCollapse(formations, env);
        }
    }

    for (auto& formation : formations) {
        for (Agent& agent : formation.agentsInFormation) {
            agents[agent.getId()] = agent;
        }
    }

    result.entropy_trace = env.collectiveContextSetTracking;
    result.cumulative_entropy = cumulative_entropy_cardinality(result.entropy_trace);
    result.stage1_steps = static_cast<int>(result.entropy_trace.size());

    if (!result.timed_out && !result.entropy_trace.empty() && result.entropy_trace.back().size() == 1) {
        result.inferred_context = *result.entropy_trace.back().begin();
    }
    result.ordering = theta_for_context(domain, result.inferred_context);
    result.wall_time_sec = timer.elapsed_sec();
    return result;
}

}  // namespace macussp
