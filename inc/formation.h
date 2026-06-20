// Formation.h
#ifndef FORMATION_H
#define FORMATION_H


#include <vector>
#include <cmath>
#include <iostream>
#include "mapf_solver.h"
#include "mapf_helper.h"
#include "agent.h"
#include "state.h"
#include "environment.h"
#include "action.h"
#include <algorithm> 
#include <limits> 
#include <numeric>
#include "params.h"
#include "InfoGathering/ARVI.h"
#include "JointObservation.h"

struct FormationState {
    std::vector<LocalState> states; // The positions of the agents in the formation.
    std::string formationName; // The name of the formation (e.g., "chain", "ring", "cross").
    std::set<int> contextSet; // The context of the formation (e.g., -1 for uncertain, or a specific context).
    
    FormationState(const std::vector<LocalState> &s, const std::string &f, std::set<int> c = {})
        : states(s), formationName(f), contextSet(c) {}
};

class Formation {
public:
    Environment &env; // The environment in which the formation is being formed.
    std::vector<Agent> agentsInFormation; // The agents involved in the formation.
    int groupID; // The ID of the group forming the formation.
    LocalState landmarkGoalState; // The goal state for the formation (e.g., the landmark position).
    FormationState currentFormationState; // The current state of the formation.
    Agent navigationAgent; // The agent responsible for navigation within the formation.    
    std::vector<std::set<int>> contextSetTracking; // trajectory of the context of the agent
    std::string formationName; // The name of the formation (e.g., "chain", "ring", "cross").
    std::vector<AugmentedState> augmentedStatesTrajForThisFormation; // trajectory of the augmented states of the agents in the formation   
    AugmentedState currentAugState; // The current augmented state of the formation.
    Formation(Environment &env_, const std::vector<Agent> &agents_, const int groupID_, const LocalState &landmarkGoalState_)
        : env(env_), agentsInFormation(agents_), groupID(groupID_), landmarkGoalState(landmarkGoalState_), currentFormationState({}, "", env.getBeliefContextSet()), navigationAgent(agents_[0]),
          contextSetTracking({}), formationName(""), augmentedStatesTrajForThisFormation({}), currentAugState(JointState(std::vector<LocalState>(getLocalStatesForAgents(agents_))), env.getBeliefContextSet()) {
        // Initialize the formation state with the current positions of the agents.
        currentFormationState = getFormationState();
        navigationAgent = getNavigationAgent();
        formationName = env.getLandmarkStateToFormationMap()[landmarkGoalState];
        // // print currentAugState
        // std::cout << "\033[1;33m[Formation] Current Augmented State: \033[0m";
        // printAugmentedState(currentAugState);
        // std::cout << std::endl;
        
    }

    inline FormationState getFormationState() const {
        
        // Initialize the formation state with the current positions of the agents.
        std::vector<LocalState> currentAgentLocalStates;
        for (int i = 0; i < agentsInFormation.size(); ++i) {
            currentAgentLocalStates.push_back(agentsInFormation[i].getState());
        }
        FormationState fstate(currentAgentLocalStates, env.getLandmarkStateToFormationMap()[landmarkGoalState], env.getBeliefContextSet());
        return fstate;
    }

    inline void assignLandmark2Formation(const LocalState &landmarkState) {
        landmarkGoalState = landmarkState;
        // debug print landmark state
        // std::cout << "\033[1;33m[formation.h] Assigned Landmark State:";
        // printLocalState(landmarkGoalState);
        // std::cout << "\033[0m" << std::endl;
        formationName = env.getLandmarkStateToFormationMap()[landmarkGoalState];
        // std::cout << "YO!" << std::endl;
    }

    inline std::vector<LocalState> getLocalStatesForAgents(std::vector<Agent> agentsInThisFormation) const {
        
        // Initialize the formation state with the current positions of the agents.
        std::vector<LocalState> currentAgentLocalStates;
        for (int i = 0; i < agentsInThisFormation.size(); ++i) {
            currentAgentLocalStates.push_back(agentsInThisFormation[i].getState());
        }
        return currentAgentLocalStates;
    }

    // Get navigation agent: it returns the agent in the formation that is closest to the goal (landmark) state
    inline Agent getNavigationAgent() const {
        double minDistance = std::numeric_limits<double>::max();
        Agent closestAgent = agentsInFormation[0];
        for (const auto &agent : agentsInFormation) {
            double distance = std::sqrt(std::pow(agent.getState().x - landmarkGoalState.x, 2) +
                                         std::pow(agent.getState().y - landmarkGoalState.y, 2));
            if (distance < minDistance) {
                minDistance = distance;
                closestAgent = agent;
            }
        }
        return closestAgent;  // Return the closest agent to the goal state. 
        // Now we can just compute the formation policy over local state space 
        // and make all agents do the same action assuming that when information
        // transitions are deterministic and no agents fall out of formation.
    }


    inline FormationState do_formation_action(const FormationState &fstate, const std::vector<std::string> &formationAction) const {
        // Check if the action is valid for the formation and all action in it are same
        if (formationAction.size() != agentsInFormation.size()) {
            std::cerr << "Error: Formation action size does not match number of agents." << std::endl;
            return fstate;
        }
        for (size_t i = 1; i < formationAction.size(); ++i) {
            if (formationAction[i] != formationAction[0]) {
                std::cerr << "Error: All agents must perform the same action in the formation." << std::endl;
                return fstate;
            }
        }
        // Apply the formation action to the formation state.
        FormationState newFormationState = fstate;
        for (size_t i = 0; i < agentsInFormation.size(); ++i) {
            LocalState &agentState = newFormationState.states[i];
            const std::string &action = formationAction[i];
            if (action == "up") {
                agentState.y -= 1;
            } else if (action == "down") {
                agentState.y += 1;
            } else if (action == "left") {
                agentState.x -= 1;
            } else if (action == "right") {
                agentState.x += 1;
            } else if (action == "observe") {
                // if an of the agents are on the goal landmark state, then set the context to the true context
                if (agentState == landmarkGoalState) {
                    newFormationState.contextSet = env.getLandmarkToContextSubsetMap().at(landmarkGoalState);
                }
                else { // nothing changes
                }
            }
            agentState.type = env.getLocalState(agentState.x,agentState.y).type; // Update the type of the cell
        }
        // the new formation state is valid if all agents are within the bounds of the environment (use env.isWithinBounds)
        for (const auto &agentState : newFormationState.states) {
            if (!env.isWithinBounds(agentState.x, agentState.y)) {
                return fstate; // Return the original state if out of bounds.
            }
        }
        // update the type of 
        return newFormationState; // Return the new formation state.
    }


    AugmentedState do_augmented_state_action(const AugmentedState &augState, const std::vector<std::string> &jointAction) {
        JointState nextJoint = do_joint_action(augState.joint, jointAction, env);
        std::set<int> newContextSet = augState.contextSet;
        std::unordered_map<LocalState, std::set<int>> landmarkToContextSubsetMap = env.getLandmarkToContextSubsetMap();
        bool allObserve = true;
        for (const auto &act : jointAction) {
            if (act != "observe") {
                allObserve = false;
                break;
            }
        }
        if (allObserve) {
            newContextSet = macussp::apply_joint_landmark_observation(
                augState.contextSet, nextJoint, landmarkGoalState, env);
        }
        return AugmentedState(nextJoint, newContextSet);
    }
};

// print FormationState that includes local states of all agents in the formation, the formation name and the context
inline void printFormationState(const FormationState &fstate) {
    std::cout << "(";
    for (int i = 0; i < fstate.states.size(); ++i) {
        std::cout << "(" << fstate.states[i].x << ", " << fstate.states[i].y << ", " << fstate.states[i].type << ")";
        if (i != fstate.states.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ", " << fstate.formationName;
    std::cout << ", contexts: {";
    for (const auto &context : fstate.contextSet) {
        std::cout << context << " ";
    }
    std::cout << "})" << std::endl;
}

// get the action given current LocalState and successor LocalState
inline std::string getActionBetweenLocalStates(const LocalState &current, const LocalState &successor) {
    if (successor.x == current.x && successor.y == current.y) {
        return "observe";
    } else if (successor.x == current.x + 1 && successor.y == current.y) {
        return "right";
    } else if (successor.x == current.x - 1 && successor.y == current.y) {
        return "left";
    } else if (successor.x == current.x && successor.y == current.y + 1) {
        return "down";
    } else if (successor.x == current.x && successor.y == current.y - 1) {
        return "up";
    }
    return "invalid"; // If the action is not recognized.
}


inline std::unordered_map<int, LocalState> computeStartingFormationLocationStates(const std::vector<Agent> &agents, const Formation &formation, Environment &env){
    const std::string& domain = env.getMapName();  // or env.mapName if you don't have a getter

    std::unordered_map<int, LocalState> result;
    const int n = static_cast<int>(agents.size());
    if (n == 0) return result;

    if (domain == "forestfire") {
        // Targets = landmark cells on the map
        std::vector<LocalState> targets;
        LocalState agentLandmarkState = env.getAllLandmarkLocalStates().at(formation.groupID);
        // the targets are the 4 cells surrounding the landmark cell (up, down, left, right)
        targets.push_back(env.getLocalState(agentLandmarkState.x, agentLandmarkState.y - 1)); // up
        targets.push_back(env.getLocalState(agentLandmarkState.x, agentLandmarkState.y + 1)); // down
        targets.push_back(env.getLocalState(agentLandmarkState.x - 1, agentLandmarkState.y)); // left
        targets.push_back(env.getLocalState(agentLandmarkState.x + 1, agentLandmarkState.y)); // right
        
        // in this domain each agent gets a unique landmark state
        if (targets.empty()) {
            std::cerr << "[computeStartingFormationLocationStates:forestfire] No landmark targets found in the environment.\n";
        }


        const int m = static_cast<int>(targets.size());

        // Manhattan distance on grid
        auto L1 = [](const LocalState& a, const LocalState& b)->int {
            return std::abs(a.x - b.x) + std::abs(a.y - b.y);
        };

        // Build n×m cost matrix: row i = agent i (start), col j = landmark j
        std::vector<std::vector<int>> cost(n, std::vector<int>(m, 0));
        for (int i = 0; i < n; ++i) {
            const LocalState s = agents[i].getState();
            for (int j = 0; j < m; ++j) {
                cost[i][j] = L1(s, targets[j]);
            }
        }

        // Hungarian algorithm (min-cost assignment), assumes n <= m
        auto hungarian_min_cost = [](const std::vector<std::vector<int>>& a){
            const int n = (int)a.size();
            const int m = (int)a[0].size();
            const int INF = std::numeric_limits<int>::max()/4;
            std::vector<int> u(n+1,0), v(m+1,0), p(m+1,0), way(m+1,0);
            for (int i = 1; i <= n; ++i) {
                p[0] = i;
                std::vector<int> minv(m+1, INF);
                std::vector<char> used(m+1, false);
                int j0 = 0;
                do {
                    used[j0] = true;
                    int i0 = p[j0], j1 = 0;
                    int delta = INF;
                    for (int j = 1; j <= m; ++j) if (!used[j]) {
                        int cur = a[i0-1][j-1] - u[i0] - v[j];
                        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                    }
                    for (int j = 0; j <= m; ++j) {
                        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                        else { minv[j] -= delta; }
                    }
                    j0 = j1;
                } while (p[j0] != 0);
                do {
                    int j1 = way[j0];
                    p[j0] = p[j1];
                    j0 = j1;
                } while (j0);
            }
            std::vector<int> assign(n, -1);
            for (int j = 1; j <= m; ++j) if (p[j] != 0) assign[p[j]-1] = j-1;
            return assign;
        };

        std::vector<int> assign = hungarian_min_cost(cost); // assign[i] = index of target for agent i

        // Build the requested mapping: agentID -> assigned LocalState
        for (int i = 0; i < n; ++i) {
            result[ agents[i].getId() ] = targets[ assign[i] ];
        }
        return result;
    }
    else if (domain == "warehouse") {
        // now shift the target states such that the closest agent is at the landmarkGoalState
        // compute the offset
        int offsetX = 0;
        int offsetY = 0;
        // The target states of the 2-pair agent formation for warehouse domain is just one below and two below the landmark (barcode).
        for (auto &agent : agents) {
            result[agent.getId()] = env.getLocalState(formation.landmarkGoalState.x + offsetX, formation.landmarkGoalState.y + offsetY++);
        }
        return result;
    }
    else if (domain == "salp") {
        // 1) compute centroid
        double sumX=0, sumY=0;
        for (auto &a : agents) {
        auto s = a.getState();
        sumX += s.x; sumY += s.y;
        }
        int avgX = int(std::round(sumX/n)),
            avgY = int(std::round(sumY/n));

        // 2) build formation slots
        std::vector<LocalState> targets;
        if (formation.formationName == "chain") {
        int offset = (n-1)/2;
        for (int i = 0; i < n; ++i) {
            int rawX = avgX - offset + i;
            int x    = std::max(0, std::min(rawX, env.getWidth()-1));
            int rawY = avgY;
            int y    = std::max(0, std::min(rawY, env.getHeight()-1));
            targets.push_back(env.getLocalState(x,y));
        }
        }
        else if (formation.formationName == "cross" && n==5) {
        std::array<std::pair<int,int>,5> offs = {{
            {-1,-1},{+1,-1},{0,0},{-1,+1},{+1,+1}
        }};
        for (auto &o: offs) {
            int rawX = avgX + o.first;
            int x    = std::max(0, std::min(rawX, env.getWidth()-1));
            int rawY = avgY + o.second;
            int y    = std::max(0, std::min(rawY, env.getHeight()-1));
            targets.push_back(env.getLocalState(x,y));
        }
        }
        else {
        std::cerr << "[f.h]Unknown formation or wrong size\n";
        return result;
        }

        // 3) brute-force best agent→target assignment by checking all permutations:
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);

        double bestCost = std::numeric_limits<double>::infinity();
        std::vector<int> bestPerm = perm;

        do {
        double cost = 0;
        for (int i = 0; i < n; ++i) {
            auto si = agents[i].getState();
            auto tj = targets[perm[i]];
            double dx = si.x - tj.x, dy = si.y - tj.y;
            cost += dx*dx + dy*dy;  // squared distance
        }
        if (cost < bestCost) {
            bestCost = cost;
            bestPerm = perm;
        }
        } while (std::next_permutation(perm.begin(), perm.end()));

        // 4) build the final map
        for (int i = 0; i < n; ++i) {
        result[ agents[i].getId() ] = targets[ bestPerm[i] ];
        }
        // return result;
        // compute the agent closes to the formation.landmarkGoalState as per the agent2TargetStateMap
        int closestAgentIndex = -1;
        double minDistance = std::numeric_limits<double>::max();
        for (auto &agent : agents) {
            double distance = std::sqrt(std::pow(result[agent.getId()].x - formation.landmarkGoalState.x, 2) +
                                        std::pow(result[agent.getId()].y - formation.landmarkGoalState.y, 2));
            if (distance < minDistance) {
                minDistance = distance;
                closestAgentIndex = agent.getId();
            }
        }
        // now shift the target states such that the closest agent is at the landmarkGoalState
        // compute the offset
        int offsetX = formation.landmarkGoalState.x - result[closestAgentIndex].x;
        int offsetY = formation.landmarkGoalState.y - result[closestAgentIndex].y;
        // shift the target states by this offset in the agent2TargetStateMap
        for (auto &agent : agents) {
            result[agent.getId()] = env.getLocalState(result[agent.getId()].x + offsetX, result[agent.getId()].y + offsetY);
        }
        // Check if the number of target states matches the number of agents in the formation.
        if (result.size() != agents.size()) {
            std::cerr << "[f.h]Error: Mismatch between number of agents and formation target states." << std::endl;
        } 

        // 2. Print the target formation positions for debugging.
        std::cout << "\033[1;33m[f.h l439]Target formation positions:\033[0m" << std::endl;
        for (int i = 0; i < agents.size(); ++i) {
            std::cout << "Agent " << agents[i].getId() << ": ";
            printLocalState(result[agents[i].getId()]);
            std::cout << std::endl;
        }

        return result;
    }
    else {
        std::cerr << "[f.h l447]Error: computeStartingFormationLocationStates() not implemented for domain " << domain << "\n";
        return result;
    }
}


inline void printFormationTrajectory(const Formation &formation) {
    std::cout << "Formation trajectory for group " << formation.groupID << ":\n";
    for (int i = 0; i < formation.augmentedStatesTrajForThisFormation.size(); ++i) {
        std::cout << "Step " << i << ": ";
        printAugmentedState(formation.augmentedStatesTrajForThisFormation[i]);
        std::cout << "\n";
    }
}
// This function computes the trajectory for each agent to reach its target formation position.
// It uses CBS MAPF to compute a local plamn for each agent and simulates the trajectory.
// The function takes in a vector of Agent objects, the desired formation name, and the environment.
inline std::unordered_map<int, LocalState> getAgent2FormationTargetStatesMap(Formation &formation, Environment &env) {

    // 1. Compute target formation positions.
    std::vector<Agent>& agentsInThisFormation = formation.agentsInFormation;
    if (formation.formationName == "") {
        // If the formation name is not set, the agents stay where they are so the target states are the same as the current states.
        std::unordered_map<int, LocalState> agent2TargetStateMap;
        for (Agent & agent : agentsInThisFormation) {
            agent2TargetStateMap[agent.getId()] = agent.getState();
        }
        return agent2TargetStateMap;
    }
    std::unordered_map<int, LocalState> agent2TargetStateMap = computeStartingFormationLocationStates(agentsInThisFormation, formation, env);

    return agent2TargetStateMap;
}

// Function that gets all agents in their respective formations to their target formation states using MAPF.
inline void getToFormationForAllAgentsUsingMAPF(std::vector<Formation> &formations, Environment &env) {

    // 1. Get the target formation states for each agent in the formation.
    std::vector<Agent> allAgents;
    std::unordered_map<int, LocalState> Agent2TargetStatesMap, agent2targetState4ThisFormation;
    for (size_t f_idx = 0; f_idx < formations.size(); f_idx++) {
        agent2targetState4ThisFormation = getAgent2FormationTargetStatesMap(formations[f_idx], env);
        for (auto &agent : formations[f_idx].agentsInFormation) {
            allAgents.push_back(agent);
            Agent2TargetStatesMap[agent.getId()] = agent2targetState4ThisFormation[agent.getId()];
        }
    }

    // print the Agent2TargetStatesMap for debugging
    for (auto agentTargetPair : Agent2TargetStatesMap) {
        std::cout << "\033[1;33m[f.h l484]Agent ID: " << agentTargetPair.first << " Target State: ";
        printLocalState(agentTargetPair.second);
        std::cout << "\033[0m" << std::endl;
    }

    // 2. Compute MAPF paths for the agents to reach their target states to get into formation.
    std::unordered_map<int, std::vector<LocalState>> Get2FormationPaths4AllAgents = runMapfSolver("LCBS", env, allAgents, Agent2TargetStatesMap, (g_active_mapf_time_limit_sec > 0 ? g_active_mapf_time_limit_sec : TIME_LIMIT), "Phase1");
    std::unordered_map<int, std::vector<LocalState>> policies4AllAgents;
    for (size_t i = 0; i < allAgents.size(); i++) {
        policies4AllAgents[allAgents[i].getId()] = Get2FormationPaths4AllAgents[i]; // path for all agents to get to their target states wary of other formation agents
    }

    // now we separate out the policies for each formation since they need individual padding
    std::unordered_map<int,std::unordered_map<int, std::vector<LocalState>>> policiesByFormation;
    for (size_t f_idx = 0; f_idx < formations.size(); f_idx++) {
        for (auto agent : formations[f_idx].agentsInFormation) {
            policiesByFormation[f_idx][agent.getId()] = policies4AllAgents[agent.getId()];
        }
    }

    size_t maxLen = 0;
    std::set<int> lastContextSet;
    int initialTrajSize;
    std::unordered_map<int, std::vector<LocalState>> policy4ThisFormation;
    for ( size_t f_idx = 0; f_idx < formations.size(); f_idx++) {
        policy4ThisFormation = policiesByFormation[f_idx];
        // 3. Determine the maximum trajectory length and pad shorter trajectories (only for agents within the formation).
        maxLen = 0;
        for (const auto &p : policy4ThisFormation) {
            if (p.second.size() > maxLen)
                maxLen = p.second.size();
        }

        for (auto &p : policy4ThisFormation) {
            while (p.second.size() < maxLen) {
                p.second.push_back(p.second.back());
            }
        }

        // 4. Set the trajectory for each agent.
        for (size_t i = 0; i < formations[f_idx].agentsInFormation.size(); i++) {
            int agentId = formations[f_idx].agentsInFormation[i].getId();
            for (size_t j = 0; j < maxLen; j++) {
                formations[f_idx].agentsInFormation[i].trajectory.push_back(env.getLocalState(policy4ThisFormation[agentId][j].x, policy4ThisFormation[agentId][j].y));  // i-th agent and j-th step
            }
        }

        initialTrajSize = formations[f_idx].augmentedStatesTrajForThisFormation.size();
        if (initialTrajSize == 0) {
            lastContextSet = env.getBeliefContextSet();
        }
        else {
            // if the trajectory is not empty, then we can just push the last context
            lastContextSet = formations[f_idx].augmentedStatesTrajForThisFormation.back().contextSet;
        }

        // 5. keep track of the context as set of all possible contexts, as its has not been resolved yet
        for (size_t i = initialTrajSize; i < formations[f_idx].agentsInFormation[0].trajectory.size(); i++) {
            formations[f_idx].contextSetTracking.push_back(lastContextSet);
        }


        // 6. Keep track of augmented states of the formation trajeoctory
        for (size_t i =  initialTrajSize; i < formations[f_idx].agentsInFormation[0].trajectory.size(); i++) {
            std::vector<LocalState> localStates;
            for (int j = 0; j < formations[f_idx].agentsInFormation.size(); j++) {
                localStates.push_back(formations[f_idx].agentsInFormation[j].trajectory[i]);
            }
            AugmentedState augState(JointState(localStates), lastContextSet);
            formations[f_idx].augmentedStatesTrajForThisFormation.push_back(augState);
        }

        // 7. Update agent states to the target formation state
        for (Agent &agent : formations[f_idx].agentsInFormation) {
            agent.setState(agent.trajectory.back());
            int agentId = agent.getId();
            std::cout << "\033[1;30m[f.h l557] Agent ID: " << agentId << ", Current State: ";
            printLocalState(agent.getState());
            std::cout << ", Goal State: ";
            printLocalState(agent.getGoal());
            std::cout << "\033[0m" << std::endl;
        }
    
        // // 8. After the formation is formed, we can set the formation state to the target formation state
        // formations[f_idx].navigationAgent = formations[f_idx].getNavigationAgent();
    
        // 9. Also keep track of the current formation Aug state (not used but could come in handy)
        formations[f_idx].currentAugState = formations[f_idx].augmentedStatesTrajForThisFormation.back();
        env.setBeliefContextSet(formations[f_idx].currentAugState.contextSet);
    }

}

// Helper: Compare two pairs (used for sorting offsets)
inline bool cmpPair(const std::pair<int, int> &a, const std::pair<int, int> &b) {
    if (a.first == b.first)
        return a.second < b.second;
    return a.first < b.first;
}

// convery single action to formation action
inline std::vector<std::string> convertToFormationAction(const std::string &action, int numAgents) {
    std::vector<std::string> formationAction(numAgents, action);
    return formationAction;
}


// print the formation details
inline void printFormationDetails(const Formation &formation) {
    std::cout << "Formation Group ID: " << formation.groupID << std::endl;
    std::cout << "\tAgent IDs: {";
    for (int j = 0; j < formation.agentsInFormation.size(); ++j) {
        if (j == formation.agentsInFormation.size() - 1){
            std::cout << formation.agentsInFormation[j].getId() << "}" << std::endl;
        }
        else{
            std::cout << formation.agentsInFormation[j].getId() << ", ";
        }
    }
    std::cout << "\tstate: {";
    for (int j = 0; j < formation.agentsInFormation.size(); ++j) {
        std::cout << formation.agentsInFormation[j].getId() << ": ";
        printLocalState(formation.agentsInFormation[j].getState());
        if (j == formation.agentsInFormation.size() - 1){
            std::cout << "}" << std::endl;
        }
        else{
            std::cout << ", ";
        }
    }
    std::cout << "\tformation: " << formation.formationName << std::endl;
    std::cout << "\tLandmark state: ";
    printLocalState(formation.landmarkGoalState);
    std::cout << "\n";
    std::cout << "\tNavigation agent: " << formation.navigationAgent.getId() << std::endl;
}

// performFormationValueIteration: uses a navigation agent among the formation already chosen and stpred on by formation.navigationAgent variable
// then uses performValueIteration to run value iteration over the augmented state space for a single agent and then basically mimics actions
// for all agents in the formation so that they all do the same action and remain in formation
inline void performFormation2BeliefCollapseMAPF(std::vector<Formation> &formations, Environment &env) {
    getToFormationForAllAgentsUsingMAPF(formations, env);

    // We need to use this plan to get the path for all agents in the formation.
    // We can use the do_action function to get the new LocalState for each agent in the formation
    std::vector<std::vector<std::string>> formationBeliefCollapsePlan;
    std::vector<std::string> formationAction;
    for (auto &formation : formations) {    
        formationAction = convertToFormationAction("observe", formation.agentsInFormation.size()); // NEED TO SEE HOW OBSERVE ACTION WORKS ON FORESTFIRE DOMAIN... Then move on to aimation changes and debugging stuff if the case
        formationBeliefCollapsePlan.push_back(formationAction);

        AugmentedState AState = formation.currentAugState;
        AState = formation.do_augmented_state_action(AState, formationAction);
        for (size_t j = 0; j < formation.agentsInFormation.size(); ++j) {
            formation.agentsInFormation[j].trajectory.push_back(AState.joint.states[j]);
        }
        formation.augmentedStatesTrajForThisFormation.push_back(AugmentedState(JointState(AState.joint.states), AState.contextSet));
        formation.contextSetTracking.push_back(AState.contextSet);

        for (size_t i = 0; i < formation.agentsInFormation.size(); ++i) {
            formation.agentsInFormation[i].setState(formation.agentsInFormation[i].trajectory.back());
        }
        formation.currentAugState = formation.augmentedStatesTrajForThisFormation.back();
    }
}

inline void performBeliefCollapse(std::vector<Formation> &formations, Environment &env, std::string SOLVER) {
    if (SOLVER == "OURS") {
        performFormation2BeliefCollapseMAPF(formations, env);
    }
    else if (SOLVER == "ARVI") {
        performFormation2BeliefCollapseARVI_VI(formations, env);
    }
    else if (SOLVER == "SAIA") {
        performFormation2BeliefCollapseARVI_VI(formations, env);
    }
    else {
        std::cerr << "[formation.h] Error: Unknown SOLVER for performFormation2BeliefCollapse: " << SOLVER << std::endl;
    }
}

#endif // FORMATION_H