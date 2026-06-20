#ifndef HELPER_UTILS_H
#define HELPER_UTILS_H

#include <iostream>
#include <fstream>
#include <json.hpp>
#include <vector>
#include "environment.h"
#include "action.h"
#include "state.h"
#include "agent.h"
#include "mapf_solver.h"
#include "formation.h"
#include "InfoGathering/ARVI.h"
#include "params.h"
#include <limits>
#include <tuple>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <queue> 
#include <unordered_map>
#include <set>
#include <string>
#include <sstream>
#include <filesystem>
using json = nlohmann::json;

// Returns the number of groups to be formed by the agents.
// Currently, this is hard-coded to return 1 (just testing out formation with 1 group first).
inline int getNumberOfGroups() {
    return 2;
}

inline void updateAgentsWithTrajectory(std::vector<AugmentedState> &globalTrajectory, std::vector<Agent> &agents) {
    // set each agents' current state to its local state in the last step of the global trajectory
    AugmentedState &state = globalTrajectory[globalTrajectory.size() - 1];
    for (size_t j = 0; j < state.joint.states.size(); ++j) {
        LocalState &ls = state.joint.states[j];
        // Assuming agents are stored in the same order as their IDs.
        agents[j].setState(ls);
    }
}

// Prints a joint action (vector of strings)
inline void printJointAction(const JointAction &ja) {
    std::cout << "(";
    for (size_t i = 0; i < ja.size(); ++i) {
        std::cout << ja[i];
        if(i < ja.size()-1)
            std::cout << ", ";
    }
    std::cout << ")";
}



// ---------- Minimal Hungarian (assignment) for non-negative integer costs ----------
static std::vector<int> hungarian_min_cost(const std::vector<std::vector<int>>& a) {
    // cp-algorithms-style implementation; handles rectangular by padding via potentials.
    int n = (int)a.size();
    int m = (int)a[0].size();
    const int INF = std::numeric_limits<int>::max()/4;

    // 1-indexed internals
    std::vector<int> u(n+1, 0), v(m+1, 0), p(m+1, 0), way(m+1, 0);
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
    std::vector<int> assignment(n, -1); // assignment[i] = column j chosen for row i
    for (int j = 1; j <= m; ++j) if (p[j] != 0) assignment[p[j]-1] = j-1;
    return assignment;
}
// -------------------------------------------------------------------------------

inline std::vector<Agent> init_agents(Environment &env, const int numAgents, int scen_id) 
{
    std::vector<Agent> agents;
    agents.reserve(numAgents);

    const std::string mapName  = env.getMapName();  // or env.mapName if public
    const std::string scenFile = "../maps/" + mapName + "/scenario_files/" + mapName + std::to_string(scen_id) + ".scen";
    std::ifstream in(scenFile);
    if (!in.is_open()) {
        std::cerr << "[init_agents] Error: cannot open scenario file: " << scenFile << "\n";
        std::exit(1);
    }

    // Access to raw map chars; assume env.MAP[y][x] with y=row in [0,H), x=col in [0,W)
    const int W = env.getWidth();
    const int H = env.getHeight();
    auto cell = [&](int x, int y)->char {
        if (x < 0 || x >= W || y < 0 || y >= H) return '@';
        return env.MAP[y][x];
    };
    auto manhattan = [&](int x1, int y1, int x2, int y2)->int {
        return std::abs(x1 - x2) + std::abs(y1 - y2);
    };

    std::string a; double b; int c;
    in >> a >> c; // skip "version 1"

    if (mapName == "salp") {
        for (int i = 0; i < numAgents; ++i) {
            size_t x0,y0,xg,yg;
            // id, map, width, height, startX, startY, goalX, goalY, dist
            in >> c >> a >> a >> a >> x0 >> y0 >> xg >> yg >> b;
            const char type0 = cell((int)x0,(int)y0);
            const char typeg = cell((int)xg,(int)yg);
            agents.emplace_back(Agent(i, LocalState((int)x0,(int)y0,type0),
                                         LocalState((int)xg,(int)yg,typeg)));
        }
        in.close();
        return agents;
    }
    else if (mapName == "forestfire") {
        // 1) Read all starts from scenario; ignore scenario goals
        std::vector<std::pair<int,int>> starts; starts.reserve(numAgents);
        for (int i = 0; i < numAgents; ++i) {
            size_t x0,y0, ignore_gx, ignore_gy;
            in >> c >> a >> a >> a >> x0 >> y0 >> ignore_gx >> ignore_gy >> b;
            starts.emplace_back((int)x0, (int)y0);
        }
        in.close();

        // 2) Detect rectangular fire blocks (4-neighbor connected components of 'F')
        struct Rect { int minx, miny, maxx, maxy; };
        std::vector<Rect> fireRects;

        std::vector<std::vector<char>> visited(H, std::vector<char>(W, 0));
        auto inb = [&](int x, int y){ return 0 <= x && x < W && 0 <= y && y < H; };
        const int dx4[4] = {+1,-1,0,0};
        const int dy4[4] = {0,0,+1,-1};

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (cell(x,y) != 'F' || visited[y][x]) continue;

                // BFS flood-fill
                int minx = x, maxx = x, miny = y, maxy = y;
                std::queue<std::pair<int,int>> q;
                q.push({x,y}); visited[y][x] = 1;

                while (!q.empty()) {
                    auto [cx, cy] = q.front(); q.pop();
                    minx = std::min(minx, cx); maxx = std::max(maxx, cx);
                    miny = std::min(miny, cy); maxy = std::max(maxy, cy);
                    for (int k = 0; k < 4; ++k) {
                        int nx = cx + dx4[k], ny = cy + dy4[k];
                        if (!inb(nx,ny) || visited[ny][nx]) continue;
                        if (cell(nx,ny) == 'F') { visited[ny][nx] = 1; q.push({nx,ny}); }
                    }
                }

                // (Optional) sanity: ensure rectangle is solid fire
                bool solid = true;
                for (int yy = miny; yy <= maxy && solid; ++yy)
                    for (int xx = minx; xx <= maxx; ++xx)
                        if (cell(xx,yy) != 'F') { solid = false; break; }
                if (!solid) {
                    std::cerr << "[init_agents:forestfire] Warning: fire component not a perfect rectangle; "
                            << "using its bounding box anyway.\n";
                }
                fireRects.push_back({minx,miny,maxx,maxy});
            }
        }

        // 3) From each rectangle, derive the four diagonal-one-away corner goals
        //    (outside the fire): (minx-1,miny-1), (maxx+1,miny-1), (minx-1,maxy+1), (maxx+1,maxy+1)
        auto passable = [&](int x,int y){
            if (!inb(x,y)) return false;
            char t = cell(x,y);
            return (t != '@' && t != 'T' && t != 'F'); // not obstacle and not fire
        };

        std::vector<std::pair<int,int>> goals;
        goals.reserve(4 * fireRects.size());
        std::vector<std::vector<char>> used(H, std::vector<char>(W, 0));
        for (const auto& r : fireRects) {
            const int corners[4][2] = {
                {r.minx, r.miny}, {r.maxx, r.miny},
                {r.minx, r.maxy}, {r.maxx, r.maxy}
            };
            const int diag[4][2] = {
                {-1,-1}, {+1,-1}, {-1,+1}, {+1,+1}
            };
            for (int k = 0; k < 4; ++k) {
                int cx = corners[k][0], cy = corners[k][1];
                int gx = cx + diag[k][0], gy = cy + diag[k][1]; // exactly one diagonal step outward
                if (passable(gx,gy) && !used[gy][gx]) {
                    goals.emplace_back(gx,gy);
                    used[gy][gx] = 1;
                } else {
                    // Optional: keep strict “one step” rule—skip if invalid; do NOT search further.
                    // If you prefer a gentle fallback, uncomment the following probe up to 2 steps:
                    /*
                    bool placed = false;
                    for (int step = 2; step <= 2 && !placed; ++step) {
                        int gx2 = cx + step*diag[k][0], gy2 = cy + step*diag[k][1];
                        if (passable(gx2,gy2) && !used[gy2][gx2]) {
                            goals.emplace_back(gx2,gy2);
                            used[gy2][gx2] = 1;
                            placed = true;
                        }
                    }
                    if (!placed) {
                        std::cerr << "[init_agents:forestfire] Corner ("<<cx<<","<<cy
                                <<") produced no valid diagonal goal; skipped.\n";
                    }
                    */
                }
            }
        }

        // 4) Sanity: enough goals?
        if ((int)goals.size() < numAgents) {
            std::cerr << "[init_agents:forestfire] Not enough corner-diagonal goals: have "
                    << goals.size() << ", need " << numAgents << ".\n";
            std::exit(1); // or implement a secondary policy
        }

        // 5) Assign agents to these goals via Hungarian (min total Manhattan distance)
        const int N = numAgents;
        const int M = static_cast<int>(goals.size());
        std::vector<std::vector<int>> cost(N, std::vector<int>(M, 0));
        for (int i = 0; i < N; ++i) {
            const auto [x0,y0] = starts[i];
            for (int j = 0; j < M; ++j) {
                const auto [gx,gy] = goals[j];
                cost[i][j] = manhattan(x0,y0,gx,gy);
            }
        }
        // Requires your existing hungarian_min_cost to support rectangular (n ≤ m).
        std::vector<int> match = hungarian_min_cost(cost); // match[i] = goal index for agent i
        assert((int)match.size() == N);

        // 6) Instantiate agents with assigned corner-diagonal goals
        for (int i = 0; i < N; ++i) {
            const auto [x0,y0] = starts[i];
            const auto [xg,yg] = goals[ match[i] ];
            const char type0 = cell(x0,y0);
            const char typeg = cell(xg,yg);
            agents.emplace_back(Agent(i, LocalState(x0,y0,type0),
                                        LocalState(xg,yg,typeg)));
        }
        return agents;
    }

    else if (mapName == "warehouse"){
        for (int i = 0; i < numAgents; ++i) {
            size_t x0,y0,xg,yg;
            // id, map, width, height, startX, startY, goalX, goalY, dist
            in >> c >> a >> a >> a >> x0 >> y0 >> xg >> yg >> b;
            const char type0 = cell((int)x0,(int)y0);
            const char typeg = cell((int)xg,(int)yg);
            agents.emplace_back(Agent(i, LocalState((int)x0,(int)y0,type0),
                                         LocalState((int)xg,(int)yg,typeg)));
        }
        in.close();
        return agents;
    }

    else {
        std::cerr << "[init_agents] Error: unrecognized map name: " << mapName << "\n";
        std::exit(1);
    }
}


// function print for every group, its members and the landmark state it is assigned to and the formation amd the observed context at that landmark:
inline void printGroupAssignmentInfo(std::unordered_map<int, std::vector<Agent>> &groupsIdToAgentsVectorMap, 
                                    std::unordered_map<int, LocalState> &groupIDToLandmarkState,
                                    std::unordered_map<LocalState, std::string> &landmarkState2FormationName,
                                    Environment &env) {
    std::unordered_map<LocalState, std::set<int>> landmarkToContextSubsetMap = env.getLandmarkToContextSubsetMap();
    for (int i = 0; i < groupsIdToAgentsVectorMap.size(); ++i) {
        std::cout << "Group " << i << ": ";
        for (const auto &agent : groupsIdToAgentsVectorMap[i]) {
            std::cout << agent.getId() << " ";
        }
        std::cout << " -> Landmark state: ";
        printLocalState(groupIDToLandmarkState[i]);
        std::cout <<  " -> Formation: " 
                  << landmarkState2FormationName[groupIDToLandmarkState[i]] << " -> Observed Contexts at this landmark: {";
        for (const auto &context : landmarkToContextSubsetMap[groupIDToLandmarkState[i]]) {
            std::cout << context;
            if (context != *landmarkToContextSubsetMap[groupIDToLandmarkState[i]].rbegin())
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
    }
}


inline void printAugStateTrajectoryByGroup(const std::unordered_map<int, std::vector<AugmentedState>> augmentedStatesTrajForByGroupId){
    std::cout << "\n--- Augmented State Trajectory by Group ---\n";
    for (int i = 0; i < augmentedStatesTrajForByGroupId.size(); i++) {
        std::cout << "\033[32mGroup " << i << ":\033[0m \n";
        for (const auto &augState : augmentedStatesTrajForByGroupId.at(i)) {
            printAugmentedState(augState);
            std::cout << "\n";
        }
        std::cout << "\n\n";
    }
}

// inline bool isBeliefCollapsed(const std::unordered_map<int, std::vector<AugmentedState>> augmentedStatesTrajForByGroupId){
//     // Check if all groups have the true context in their last augmented state.
//     for (int i = 0; i < augmentedStatesTrajForByGroupId.size(); i++) {
//         const auto &augState = augmentedStatesTrajForByGroupId.at(i).back();
//         if (augState.context == -1) {
//             return false; // Not all groups have belief collapsed.
//         }
//     }
//     return true; // All groups have belief collapsed.
// }

inline bool checkBeliefCollapse(const Environment &env) {
    std::set<int> envLatestContextSet = env.getBeliefContextSet();
    std::set<int> trueContextSet = {env.getTrueContext()}; // set with only the true context (so singleton set) but can be extended to multiple true contexts in future
    bool collapsed = false;
    // Check if all groups have the true context in their last augmented state.
    if (envLatestContextSet == trueContextSet) {
        // print "Belief has collapsed." in green
        std::cout << "\033[1;32mBelief has collapsed.\t True Conctext = " << env.getTrueContext() << "\033[0m" << std::endl;
        collapsed = true;
    }
    else {
        // print "Belief has not collapsed yet." in red
        std::cout << "\033[1;31mBelief has not collapsed yet.\033[0m" << std::endl;
    }
    return collapsed;
}

// combine individual steps in agent.trajectory and the env.contextTracking to make a list of augmented states
inline std::unordered_map<int, std::vector<AugmentedState>> getAugmentedStateTrajectoryByGroups(std::vector<Formation> formations) {
    std::unordered_map<int, std::vector<AugmentedState>> augmentedStatesTrajForByGroupId;
    // iterate over each formation and use formations[i].augmentedStatesTrajForThisFormation stored stuff so far that already has it
    for (int i = 0; i < formations.size(); ++i) {
        // get the groupID
        int groupID = formations[i].groupID;
        // get the augmented states trajectory for this formation
        std::vector<AugmentedState> augmentedStatesTrajForThisFormation = formations[i].augmentedStatesTrajForThisFormation;
        // append it to the groupID
        augmentedStatesTrajForByGroupId[groupID] = augmentedStatesTrajForThisFormation;
    }
    // make sure that the augmented states trajectory for each group is of same length by padding with the last state
    int maxLength = 0;
    for (const auto &pair : augmentedStatesTrajForByGroupId) {
        maxLength = std::max(maxLength, static_cast<int>(pair.second.size()));
    }
    // based on the maxLength, pad the augmented states trajectory for each group
    for ( int i = 0; i < augmentedStatesTrajForByGroupId.size(); ++i) {
        int currentLength = augmentedStatesTrajForByGroupId[i].size();
        if (currentLength < maxLength) {
            AugmentedState lastState = augmentedStatesTrajForByGroupId[i].back();
            for (int j = currentLength; j < maxLength; ++j) {
                augmentedStatesTrajForByGroupId[i].push_back(lastState);
            }
        }
    }

    // debug print the augmented state trajectory by group
    for ( int k = 0; k < augmentedStatesTrajForByGroupId.size(); ++k) {
        std::cout << "[h_u.h]Group " << k << " Augmented State traj: \n";
        for ( int j = 0; j < augmentedStatesTrajForByGroupId[k].size(); ++j) {
            printAugmentedState(augmentedStatesTrajForByGroupId[k][j]);
            std::cout << "\n";
        }
        std::cout << "\n===========\n" <<std::endl;
    }
    return augmentedStatesTrajForByGroupId;
}

// Reads the coordination graph from a JSON file. The JSON file is expected to have keys 
// like "2", "4", "5", each mapping to a 2D array (matrix) of integers.
// Returns the matrix for the given number of agents.
inline std::vector<std::vector<int>> readCoordinationGraph(int numAgents, const std::string &filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Could not open coordination graph file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    json j;
    infile >> j;
    infile.close();
    std::string key = std::to_string(numAgents);
    if (j.find(key) == j.end()) {
        std::cerr << "Error: Coordination graph for " << numAgents << " agents not found in " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<std::vector<int>> cg;
    for (auto &row : j[key]) {
        std::vector<int> r;
        for (auto &val : row) {
            r.push_back(val.get<int>());
        }
        cg.push_back(r);
    }
    return cg;
}




// Saves the trajectory of overall aurmentedStates from all agents to a JSON file much like the above commented function.
// This function takes in vector of all agents, and environment object as input,
// then uses the each agent's trajectory and the environment's contextTracking to create a vector of augmented states.
// then save it to a JSON file same as the above function.
inline void saveTrajectoryToFile(const std::vector<Agent> &agents, const Environment &env , const std::string &filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    std::set<int> contextSet = env.getBeliefContextSet();
    ofs << "{\n  \"trajectory\": [\n";
    for (size_t step_i = 0; step_i < agents[0].trajectory.size(); step_i++) {
        std::vector<LocalState> localStates;
        for (int j = 0; j < agents.size(); j++) {
            localStates.push_back(agents[j].trajectory[step_i]);
        }
        contextSet = env.collectiveContextSetTracking[step_i];

        AugmentedState state(JointState(localStates), contextSet);
        ofs << "    { \"joint\": [";
        for (size_t j = 0; j < state.joint.states.size(); j++) {
            const LocalState &ls = state.joint.states[j];
            ofs << "[" << ls.x << ", " << ls.y << ", \"" << ls.type << "\"]";
            if (j < state.joint.states.size() - 1)
                ofs << ", ";
        }
        ofs << "], \"contexts\": [";
        for (const auto &context : contextSet) {
            ofs << context;
            if (context != *contextSet.rbegin())
                ofs << ", ";
        }
        ofs << "]}";
        if (step_i < agents[0].trajectory.size() - 1)
            ofs << ",";
        ofs << "\n";
    }
    ofs << " ],\n\n";
    ofs << "  \"goals\": [";
    for (int i = 0; i < agents.size(); i++) {
        const LocalState &ls = agents[i].getGoal();
        ofs << "[" << ls.x << ", " << ls.y << ", \"" << ls.type << "\"]";
        if (i < agents.size() - 1)
            ofs << ", ";
    }
    ofs << "]\n}\n";
    ofs.close();
    
    // Dump fire trajectory per animation frame; freeze at T*
    if (env.domain() == DomainType::ForestFire && env.fireEnabled()) {
        const int n_frames = static_cast<int>(agents.at(0).trajectory.size()); 
        const std::string fire_path = "../results/MAPF_OUTPUT/fire_trajectory.json";
        const bool okDump = ff_helper::writeFireTrajectory(env, n_frames, fire_path);
    }
}

// Combines the MCUSSP trajectory (globalTrajectoriesTillCollapse) and the MAPF trajectories 
// (provided as an unordered_map<int, vector<LocalState>> indexed by agent ID)
// into a complete trajectory (vector<AugmentedState> completeTrajectories).
// The MAPF portion is constructed by, for each time step, combining the local states 
// from each agent (in order) into a JointState, and setting the context to trueContext.
// If any MAPF trajectory is shorter than the others, its last state is repeated until all are equal.
inline std::vector<AugmentedState> combineCompleteTrajectories(
    const std::vector<AugmentedState>& sspTrajectory,
    const std::unordered_map<int, std::vector<LocalState>> &mapfPaths,
    std::set<int> trueContextSet)
{
    std::vector<AugmentedState> completeTrajectories;
    // Insert the SSP trajectory first.
    completeTrajectories.insert(completeTrajectories.end(), sspTrajectory.begin(), sspTrajectory.end());
    
    // Assume MAPF trajectories are already padded. Get the number of steps from one agent's path.
    size_t numSteps = 0;
    if (!mapfPaths.empty()) {
        numSteps = mapfPaths.begin()->second.size();
    }
    
    // For each time step, combine the states from all agents (assuming agent IDs 0 to n-1)
    for (size_t t = 0; t < numSteps; ++t) {
        std::vector<LocalState> jointStates;
        for (int agentId = 0; agentId < static_cast<int>(mapfPaths.size()); agentId++) {
            const LocalState &ls = mapfPaths.at(agentId)[t];
            jointStates.push_back(ls);
        }
        JointState js(jointStates);
        AugmentedState aug(js, trueContextSet);
        completeTrajectories.push_back(aug);
    }
    
    return completeTrajectories;
}


// simulateAgentTransition: Given an agent's current LocalState and intended next LocalState,
// simulate a stochastic outcome. For simplicity, we assume the following probabilities:
//   - 0.8: move as intended
//   - 0.1: slide left relative to the intended move
//   - 0.1: slide right relative to the intended move
// (The actual implementation may depend on your environment; here is a simple version.)
inline LocalState simulateAgentTransition(const LocalState &current, const LocalState &intended, const Environment &env) {
    int dx = intended.x - current.x;
    int dy = intended.y - current.y;
    double r = static_cast<double>(rand()) / RAND_MAX;
    int out_dx = dx, out_dy = dy;
    
    // For simplicity, assume non-diagonal moves only.
    if (r < 0.8) {
        // intended move.
    } else if (r < 0.9) {
        // slide left: assume left turn means (-dy, dx)
        out_dx = -dy;
        out_dy = dx;
    } else {
        // slide right: assume right turn means (dy, -dx)
        out_dx = dy;
        out_dy = -dx;
    }
    
    int newX = current.x + out_dx;
    int newY = current.y + out_dy;
    if (!env.isWithinBounds(newX, newY)) {
        // If the new state is out-of-bounds, return the current state.
        return current;
    }
    return env.getLocalState(newX, newY);
}


// padAgentTrajectories: Given a vector of Agent objects, this function pads each agent's trajectory
inline void padAgentTrajectories(std::vector<Agent> &agents){
    // Get the maximum trajectory length among all agents.
    size_t maxLength = 0;
    for (const auto &agent : agents) {
        maxLength = std::max(maxLength, agent.trajectory.size());
    }
    
    // Pad each agent's trajectory to the maximum length.
    for (auto &agent : agents) {
        auto &trajectory = agent.trajectory;
        while (trajectory.size() < maxLength) {
            trajectory.push_back(trajectory.back()); // Repeat the last state.
        }
    }

    // update the agent's current state to the last state in their trajectory
    for (int i = 0; i < static_cast<int>(agents.size()); i++) {
        agents[i].setState(agents[i].trajectory.back());
    }
}
// appendMapfTrajectories: Given a vector of Agent objects, and mapfPaths (a map of agent ID to their planned paths),
// this function appends the planned paths to each agent's trajectory.
inline void appendMapfTrajectories(std::vector<Agent> &agents, std::unordered_map<int, std::vector<LocalState>> finalPaths,
                                   Environment &env, const bool beliefCollapsed) {
    int agentTrajectoryLengthAtMapfStart = agents[0].trajectory.size();  // all are same at this point (till belief collapse of all synced-up), so we just use agent[0] to get that length
    for (int i = 0; i < agents.size(); ++i) {
        int agentId = agents[i].getId();
        for (int j = 0; j < finalPaths[agentId].size(); ++j) {
            agents[i].trajectory.push_back(env.getLocalState(finalPaths[agentId][j].x, finalPaths[agentId][j].y));
        }
    }
    padAgentTrajectories(agents);
    // append true context to env.contextTracking corresponding to the final padded length of theMAPF trajectories
    for (int i = 0; i< agents[0].trajectory.size() - agentTrajectoryLengthAtMapfStart; ++i) {
        if (beliefCollapsed) env.collectiveContextSetTracking.push_back({env.getTrueContext()});
        else env.collectiveContextSetTracking.push_back(env.getBeliefContextSet());
    }
}

// appendTillCollapseTrajectories:  Given a vector of Agent objects, and augmented State trajectory
inline void updateAgentTragectoriesTillCollapse(std::vector<Agent> &agents,  
                                                std::unordered_map<int, std::vector<Agent>> groupsIdToAgentsVectorMap,
                                                Environment &env) {
    for (int i = 0; i < groupsIdToAgentsVectorMap.size(); ++i) {
        for (Agent &agent : groupsIdToAgentsVectorMap[i]) {
            agents[agent.getId()].trajectory.insert(agents[agent.getId()].trajectory.end(), agent.trajectory.begin(), agent.trajectory.end());
        }
    }
}
                                                
// update all agent current states to the last state in their trajectory till belief collapse
inline void updateAllAgentCurrentStates(std::vector<Agent> &agents, std::unordered_map<int, std::vector<Agent>> groupsIdToAgentsVectorMap,
                                        std::unordered_map<int, std::vector<AugmentedState>> augmentedStatesTrajForByGroupId) {
    for (int i = 0; i < groupsIdToAgentsVectorMap.size(); ++i) {
        for (Agent &agent : groupsIdToAgentsVectorMap[i]) {
            // printLocalState(augmentedStatesTrajForByGroupId[i].back().joint.states[agent.getId()]);
            agents[agent.getId()].setState(augmentedStatesTrajForByGroupId[i].back().joint.states[agent.getId()]);
            // printLocalState(agents[agent.getId()].getState());
        }
    }
    // print the current state for each agent
    for (int i = 0; i < static_cast<int>(agents.size()); i++) {
        std::cout << "[in updateAllAgentCurrentStates()]Agent " << agents[i].getId() << ": ";
        printLocalState(agents[i].getState());
        std::cout << "\n";
    }

}

// Helper function to compute the Euclidean distance between two LocalState objects.
inline double euclideanDistance(const LocalState &a, const LocalState &b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// assignGroups: Given a vector of Agent objects, assign a group number to each agent
// based on their spatial vicinity. Two agents will be assigned to the same group
// if the Euclidean distance between their current states is within the given threshold.
// No group will have more than 5 agents. This function returns an unordered_map
// mapping agent ID to group number.
inline std::unordered_map<int, int> assignGroups(const Environment&env, std::vector<Agent> &agents, int numOfGroups, double threshold = 8.0) {
    std::unordered_map<int, int> groupMap;
    int groupId = 0;
    
    // just hard-coding for know assigning  in groups of 5, folling sequientially 
    for (int i = 0; i < agents.size(); ++i) {
        groupMap[agents[i].getId()] = i / 5;
    }
    
    return groupMap;
}

// given a vector of agents, a mapping of agent ID to group number (groupAssignment),
// return an unordered_map mapping group IDs to vectors of agents in that group
inline std::unordered_map<int, std::vector<Agent>> getGroupToAgentsVectorMap(std::vector<Agent> &agents,
                                                              const std::unordered_map<int, int> &groupAssignment) {
    std::unordered_map<int, std::vector<Agent>> groupsIdToAgentsVectorMap;
    
    for (auto &agent : agents) {
        int agentId = agent.getId();
        auto it = groupAssignment.find(agentId);
        if (it != groupAssignment.end()) {
            int groupId = it->second;
            groupsIdToAgentsVectorMap[groupId].push_back(agent);
        }
    }
    
    return groupsIdToAgentsVectorMap;
}

/// mapGroupToLandmark:
/// Given a vector of agents, a mapping of agent ID to group number (groupAssignment),
/// and the environment (which can return a landmark map), this function computes the centroid of
/// each group (based on the agents’ current states) and then, for each group, finds the landmark
/// whose position is closest to the centroid. It returns an unordered_map mapping group IDs to landmark IDs.
///
inline std::unordered_map<int, LocalState> getMapGroupIDToLandmarkState(const std::vector<Agent>& agents,
                                                         const std::unordered_map<int, int>& groupAssignment,
                                                         const Environment& env) {
    // Obtain the landmark map from the environment.
    // Assumes env.getLandmarkMap() returns an unordered_map<int, LocalState>.
    std::unordered_map<int, LocalState> landmarks = env.getLandmarkMap();

    // First, for each group, compute the centroid (average x and y).
    std::unordered_map<int, std::pair<double, double>> groupCentroids; // group ID -> (sum_x, sum_y)
    std::unordered_map<int, int> groupCounts; // group ID -> number of agents

    for (const auto &agent : agents) {
        int agentId = agent.getId();
        // Get the group number for this agent.
        auto it = groupAssignment.find(agentId);
        if (it == groupAssignment.end()) {
            // If agent is not assigned, skip it.
            continue;
        }
        int groupId = it->second;
        LocalState s = agent.getState();
        groupCentroids[groupId].first += s.x;
        groupCentroids[groupId].second += s.y;
        groupCounts[groupId]++;
    }

    // // Compute the average (centroid) for each group.
    std::unordered_map<int, std::pair<double, double>> groupAverage;
    for (const auto &entry : groupCentroids) {
        int groupId = entry.first;
        double sumX = entry.second.first;
        double sumY = entry.second.second;
        int count = groupCounts[groupId];
        groupAverage[groupId] = { sumX / count, sumY / count };
    }

    // // For each group, select the landmark that is closest to the group's centroid. but dont assign the same landmark to multiple groups
    // // maintain a vector of landmarks that have been assigned
    std::vector<LocalState> assignedLandmarks; // landmarks that have been assigned
    std::vector<LocalState> landmarkStates = env.getAllLandmarkLocalStates(); // get all landmark states in the environment
    std::unordered_map<int, LocalState> groupIdToLandmarkState; // group ID -> landmark LocalState
    for (const auto &entry : groupAverage) {
        int groupId = entry.first;
        double centroidX = entry.second.first;
        double centroidY = entry.second.second;

        // Find the closest landmark to this centroid.
        double minDistance = std::numeric_limits<double>::max();
        LocalState closestLandmark;
        for (const auto &landmark : landmarkStates) {
            // Check if this landmark has already been assigned to another group.
            if (std::find(assignedLandmarks.begin(), assignedLandmarks.end(), landmark) != assignedLandmarks.end()) {
                continue; // Skip this landmark, it's already assigned.
            }
            double distance = euclideanDistance(LocalState(centroidX, centroidY, 'L'), landmark);
            if (distance < minDistance) {
                minDistance = distance;
                closestLandmark = landmark;
            }
        }
        // printLocalState(closestLandmark);
        // Assign this closest landmark to the group.
        groupIdToLandmarkState[groupId] = closestLandmark;
        assignedLandmarks.push_back(closestLandmark); // Mark this landmark as assigned.
    }
    
    return groupIdToLandmarkState;
}

inline std::vector<int> diagonalRingGoalsAtT(const Environment& env, int fire_id, int t) {
    auto [xf, yf] = env.firePosAt(fire_id, t);
    static const int dx[4] = {-1,+1,-1,+1};
    static const int dy[4] = {-1,-1,+1,+1};
    std::vector<int> vids; vids.reserve(4);
    for (int k=0;k<4;k++){
        int x = xf + dx[k], y = yf + dy[k];
        if (!env.isWithinBounds(x,y)) continue;
        int v = env.nodeIdAt(x,y);
        if (v >= 0) vids.push_back(v);
    }
    return vids;
}

// Earliest arrival time to vertex 'goal_v' by or before deadline T,
// forbidding the fire’s occupied vertex at each time.
inline int earliestArrivalByT(const Environment& env, int start_v, int goal_v, int T) {
    // BFS over (vertex, time) with waits allowed
    const int V = (int)env.vertices().size();
    std::vector<char> seen((size_t)V*(T+1), 0);
    auto idx = [&](int v,int t){ return v*(T+1)+t; };

    std::deque<std::pair<int,int>> dq;
    // t=0 start is valid only if free at t=0
    if (!env.isVertexFreeAtTime(start_v, 0)) return INT_MAX;
    dq.emplace_back(start_v, 0);
    seen[idx(start_v,0)] = 1;

    while (!dq.empty()) {
        auto [v,t] = dq.front(); dq.pop_front();
        if (v == goal_v) return t;             // reached; may be < T
        if (t == T) continue;                  // cannot go beyond T

        // 1) wait in place
        if (env.isVertexFreeAtTime(v, t+1) && !seen[idx(v,t+1)]) {
            seen[idx(v,t+1)] = 1;
            dq.emplace_back(v, t+1);
        }
        // 2) move to neighbors
        for (int u : env.neighbors(v)) {
            if (!env.isVertexFreeAtTime(u, t+1)) continue;
            if (!seen[idx(u,t+1)]) {
                seen[idx(u,t+1)] = 1;
                dq.emplace_back(u, t+1);
            }
        }
    }
    return INT_MAX; // not reachable by T
}

inline bool hasPerfectMatchingSize4(const std::vector<std::vector<int>>& adjL, int nAgents) {
    // adjL: size L<=4; adjL[l] contains agent indices reachable for that diagonal
    int L = (int)adjL.size();
    std::vector<int> matchR(nAgents, -1);
    std::function<bool(int,std::vector<char>&)> dfs = [&](int l, std::vector<char>& vis){
        for (int r : adjL[l]) if (!vis[r]) {
            vis[r] = 1;
            if (matchR[r] == -1 || dfs(matchR[r], vis)) { matchR[r] = l; return true; }
        }
        return false;
    };
    int m = 0;
    for (int l=0; l<L; ++l) {
        std::vector<char> vis(nAgents, 0);
        if (dfs(l, vis)) ++m; else return false;
    }
    return (m == 4);
}



/// Plot |context set| vs time for each solver as a PNG using a tiny Python+Matplotlib script.
/// - ContextSetTrackingPerSolver[i] holds the timeline for SOLVERS[i].
/// - out_png is where the figure is saved.
/// Requires: python3 with matplotlib installed.
inline void plotContextSetTrackingFigure(
    const std::unordered_map<int, std::vector<std::set<int>>>& ContextSetTrackingPerSolver,
    const std::vector<std::string>& SOLVERS,
    const std::string& out_png = "../results/plots/belief_distance.png")
{
    namespace fs = std::filesystem;

    // 1) Build JSON-like payload for Python
    std::ostringstream data_json;
    data_json << "{";
    bool first = true;
    for (size_t i = 0; i < SOLVERS.size(); ++i) {
        auto it = ContextSetTrackingPerSolver.find((int)i);
        if (it == ContextSetTrackingPerSolver.end()) continue;

        if (!first) data_json << ",";
        first = false;

        data_json << "\"" << SOLVERS[i] << "\":[";
        const auto& track = it->second;
        for (size_t t = 0; t < track.size(); ++t) {
            if (t) data_json << ",";
            data_json << track[t].size();
        }
        data_json << "]";
    }
    data_json << "}";

    // // 2) Ensure output directory exists
    // try {
    //     fs::path outp(out_png);
    //     fs::create_directories(outp.parent_path());
    // } catch (...) {
    //     // ignore; system may create it anyway
    // }

    // 3) Write the small Python script
    const std::string py_path = "../results/plots/_plot_context_tmp.py";
    std::ofstream py(py_path);
    if (!py.is_open()) {
        std::cerr << "[plot] ERROR: cannot open " << py_path << " for writing.\n";
        return;
    }

    py <<
        R"(import json, sys
import matplotlib.pyplot as plt

# --- data inlined from C++ ---
data = )" << data_json.str() << R"(

out_png = r")" << out_png << R"("
solvers = ["OURS", "ARVI"]
if not data:
    print("[plot] No data to plot."); sys.exit(0)

plt.figure(figsize=(8, 4.5), dpi=150)
for solver_idx, ys in data.items():
    xs = list(range(len(ys)))
    if not xs: 
        continue
    plt.plot([int(i) for i in xs], [int(j)-1 for j in ys], linewidth=2, label=solver_idx)

plt.title("Belief Deviation from Oracle")
plt.xlabel("Time steps")
plt.ylabel("Belief Distance")
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc="best")
plt.tight_layout()
plt.savefig(out_png)
print(f"[plot] Saved figure to {out_png}")
)";
    py.close();

    // 4) Execute the script
    std::ostringstream cmd;
    cmd << "python3 " << py_path;
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::cerr << "[plot] WARNING: python returned non-zero (" << rc
                  << "). Ensure python3 + matplotlib are available.\n";
    }
}

#endif // HELPER_UTILS_H
