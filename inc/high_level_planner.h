#ifndef HIGH_LEVEL_PLANNER_H
#define HIGH_LEVEL_PLANNER_H

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <unordered_map>
#include "environment.h"
#include "agent.h"
#include "BeliefSet.h"
#include "formation.h"
#include "mapf_helper.h"
#include <random>
#include <limits>
#include <cmath>
#include <unordered_set>

// High-level planning utilities for belief-guided grouping and landmark assignment


struct CapturePlanResult {
    bool success{false};
    int  T_star{-1};
    std::array<int,4> assignedAgents{-1,-1,-1,-1};
    std::array<int,4> assignedGoals{-1,-1,-1,-1}; // goal vertex-ids
    std::unordered_map<int, std::vector<LocalState>> finalPaths; // filled if success
};

// Offline earliest-capture + single LCBS plan (no MAPF inside the loop).
// Hmax: length-1 of generated fire trajectory; per_try_limit_sec: small per-try BFS budget;
// mapf_time_limit_sec: LCBS time budget for the single final solve.
CapturePlanResult ComputeEarliestCaptureAndPlan(
    Environment& env,
    std::vector<Agent>& agents,
    const std::unordered_map<int, LocalState>& baseAgent2GoalStateMap,
    int Hmax,
    int per_try_limit_sec,
    int mapf_time_limit_sec);
    
namespace high_level_planner {

inline bool hasBeliefCollapsed(const std::vector<Formation> formations){
    // Check if all formations have collapsed
    std::set<int> intersectionOfContextSetFromEachFormation = formations[0].contextSetTracking.back();
    // compute intersection of context set of from the last step of each formation to ...
    // see if the intersection leads to a set with only one element.
    for (int i = 0; i < formations.size(); ++i) {
        std::set<int> tempSet;
        std::set_intersection(intersectionOfContextSetFromEachFormation.begin(), intersectionOfContextSetFromEachFormation.end(),
                              formations[i].contextSetTracking.back().begin(), formations[i].contextSetTracking.back().end(),
                              std::inserter(tempSet, tempSet.begin()));
        intersectionOfContextSetFromEachFormation = tempSet;
    }
    // check if the intersection is empty
    if (intersectionOfContextSetFromEachFormation.empty()) {
        std::cerr << "\033[1;31mError: Intersection context set is empty (check individual formation AugState trajectories!).\033[0m" << std::endl;
        exit(1);
    }
    // check if the intersection has only one element
    if (intersectionOfContextSetFromEachFormation.size() == 1) {
        std::cout << "\033[1;32mBelief Collapsed!!\033[0m" << std::endl;
        return true; // belief has collapsed
    }
    else {
        std::cout << "\033[1;31mBelief has not collapsed yet!\033[0m" << std::endl;
        return false; // belief has not collapsed
    }
}

// This function goes through the same formation step parallely of each formation to compute the intersection context set
// maintains a context set tracking that is a vector of sets of context sets from each formation resulting in a single set from that step
// this context set tracking is done till belief collapse (i.e., the intersection of all context sets is a single element)
// finally this function clips the trajectory of each agent to the length of this context set tracking vector
// and also updates the current state of the agent to the last state of the, now clipped, trajectory
inline std::vector<std::set<int>> synchronizeFormationsTillCollapse(std::vector<Formation> &formations, Environment &env) {
    std::vector<std::set<int>> contextSetTracking;
    std::string domainName = formations[0].env.getMapName();
    if (domainName == "salp"){
        std::set<int> intersectionOfContextSetFromEachFormation = formations[0].env.getPossibleContextSet();
        // find the largest length of the augmented state trajectory among all formations
        size_t maxLen = formations[0].augmentedStatesTrajForThisFormation.size();
        for (int i = 1; i < formations.size(); ++i) {
            if (formations[i].augmentedStatesTrajForThisFormation.size() > maxLen)
                maxLen = formations[i].augmentedStatesTrajForThisFormation.size();
        }
        // pad each agent trajectory of each formation to the same length
        for (auto &formation : formations) {
            for (size_t i = formation.augmentedStatesTrajForThisFormation.size(); i < maxLen; ++i) {
                formation.augmentedStatesTrajForThisFormation.push_back(formation.augmentedStatesTrajForThisFormation.back());
                formation.contextSetTracking.push_back(formation.contextSetTracking.back());
                for (size_t j = 0; j < formation.agentsInFormation.size(); ++j) {
                    formation.agentsInFormation[j].trajectory.push_back(formation.agentsInFormation[j].trajectory.back());
                }
            }
        }
        // the belief has collapsed for all in #minLen of steps
        // now we iterate for #max steps for all formations and compute the intesection of the context sets and storing it as well
        for (size_t step_i = 0; step_i <= maxLen; ++step_i) {
            intersectionOfContextSetFromEachFormation = formations[0].augmentedStatesTrajForThisFormation[step_i].contextSet;
            // compute intersection of context set of from the last step of each formation to ...
            // see if the intersection leads to a set with only one element.
            for (int i = 0; i < formations.size(); ++i) {
                std::set<int> tempSet;
                std::set_intersection(intersectionOfContextSetFromEachFormation.begin(), intersectionOfContextSetFromEachFormation.end(),
                                    formations[i].augmentedStatesTrajForThisFormation[step_i].contextSet.begin(), formations[i].augmentedStatesTrajForThisFormation[step_i].contextSet.end(),
                                    std::inserter(tempSet, tempSet.begin()));
                intersectionOfContextSetFromEachFormation = tempSet;
            }
            contextSetTracking.push_back(intersectionOfContextSetFromEachFormation);
            // check if the intesection that was just computed has only one element (i.e., belief has collapsed)
            if (intersectionOfContextSetFromEachFormation.size() == 1) break;
        }
        std::cout << "\033[1;30m[high_level_planner] Belief collapsed in " << contextSetTracking.size() << " steps.\033[0m" << std::endl;

        // clip all the agent trajectories to the length of the context set tracking vector...
        // and update the current state of the agent to the last state of the, now clipped, trajectory
        for (auto &formation : formations) {
            formation.augmentedStatesTrajForThisFormation.resize(contextSetTracking.size());
            for (size_t i = 0; i < formation.agentsInFormation.size(); ++i) {
                formation.agentsInFormation[i].trajectory.resize(contextSetTracking.size());
                formation.agentsInFormation[i].setState(formation.agentsInFormation[i].trajectory[contextSetTracking.size()-1]);
            }
        }
        return contextSetTracking;
    }
    // else if (domainName == "warehouse"){
    //     std::set<int> intersectionOfContextSetFromEachFormation = formations[0].env.getPossibleContextSet();
    //     // instead of padding to maximum length, the agents tragedies remain as is
    //     // the belief has collapsed for all in #minLen of steps
    //     size_t minLen = formations[0].augmentedStatesTrajForThisFormation.size();
    //     for (int i = 1; i < formations.size(); ++i) {
    //         if (formations[i].augmentedStatesTrajForThisFormation.size() < minLen)
    //             minLen = formations[i].augmentedStatesTrajForThisFormation.size();
    //     }

    //     for (size_t step_i = 0; step_i < minLen; ++step_i) {
    //         intersectionOfContextSetFromEachFormation = formations[0].augmentedStatesTrajForThisFormation[step_i].contextSet;
    //         // compute intersection of context set of from the last step of each formation to ...
    //         // see if the intersection leads to a set with only one element.
    //         for (int i = 0; i < formations.size(); ++i) {
    //             std::set<int> tempSet;
    //             std::set_intersection(intersectionOfContextSetFromEachFormation.begin(), intersectionOfContextSetFromEachFormation.end(),
    //                                 formations[i].augmentedStatesTrajForThisFormation[step_i].contextSet.begin(), formations[i].augmentedStatesTrajForThisFormation[step_i].contextSet.end(),
    //                                 std::inserter(tempSet, tempSet.begin()));
    //             intersectionOfContextSetFromEachFormation = tempSet;
    //         }
    //         contextSetTracking.push_back(intersectionOfContextSetFromEachFormation);
    //     }
    //     std::cout << "\033[1;30m[high_level_planner] Belief collapsed in " << contextSetTracking.size() << " steps.\033[0m" << std::endl;
    //     // clip all the agent trajectories to the length of the context set tracking vector...
    //     // and update the current state of the agent to the last state of the, now clipped, trajectory
    //     for (auto &formation : formations) {
    //         formation.augmentedStatesTrajForThisFormation.resize(contextSetTracking.size());
    //         for (size_t i = 0; i < formation.agentsInFormation.size(); ++i) {
    //             formation.agentsInFormation[i].trajectory.resize(contextSetTracking.size());
    //             formation.agentsInFormation[i].setState(formation.agentsInFormation[i].trajectory.back());
    //         }
    //     }
    //     return contextSetTracking;
    // }
    else if (domainName == "forestfire" or domainName == "warehouse"){
        std::set<int> intersectionOfContextSetFromEachFormation = formations[0].env.getPossibleContextSet();
        // find the largest length of the augmented state trajectory among all formations
        size_t maxLen = formations[0].augmentedStatesTrajForThisFormation.size();
        for (int i = 1; i < formations.size(); ++i) {
            if (formations[i].augmentedStatesTrajForThisFormation.size() > maxLen)
                maxLen = formations[i].augmentedStatesTrajForThisFormation.size();
        }
        // pad each agent trajectory of each formation to the same length
        for (auto &formation : formations) {
            for (size_t i = formation.augmentedStatesTrajForThisFormation.size(); i < maxLen; ++i) {
                formation.augmentedStatesTrajForThisFormation.push_back(formation.augmentedStatesTrajForThisFormation.back());
                formation.contextSetTracking.push_back(formation.contextSetTracking.back());
                for (size_t j = 0; j < formation.agentsInFormation.size(); ++j) {
                    formation.agentsInFormation[j].trajectory.push_back(formation.agentsInFormation[j].trajectory.back());
                }
            }
        }
        // the belief has collapsed for all in #minLen of steps
        // now we iterate for #max steps for all formations and compute the intesection of the context sets and storing it as well
        for (size_t step_i = 0; step_i < maxLen; ++step_i) {
            intersectionOfContextSetFromEachFormation = formations[0].augmentedStatesTrajForThisFormation[step_i].contextSet;
            // compute intersection of context set of from the last step of each formation to ...
            // see if the intersection leads to a set with only one element.
            for (int i = 0; i < formations.size(); ++i) {
                std::set<int> tempSet;
                std::set_intersection(intersectionOfContextSetFromEachFormation.begin(), intersectionOfContextSetFromEachFormation.end(),
                                    formations[i].augmentedStatesTrajForThisFormation[step_i].contextSet.begin(), formations[i].augmentedStatesTrajForThisFormation[step_i].contextSet.end(),
                                    std::inserter(tempSet, tempSet.begin()));
                intersectionOfContextSetFromEachFormation = tempSet;
            }
            contextSetTracking.push_back(formations[0].env.getPossibleContextSet());
        }
        // contextSetTracking.push_back({formations[0].env.getTrueContext()}); // at the end, we know the true context
        std::cout << "\033[1;30m[high_level_planner] Belief collapsed in " << contextSetTracking.size() << " steps.\033[0m" << std::endl;
        env.beliefContextSetsTillCollapse = contextSetTracking;

        // clip all the agent trajectories to the length of the context set tracking vector...
        // and update the current state of the agent to the last state of the, now clipped, trajectory
        for (auto &formation : formations) {
            formation.augmentedStatesTrajForThisFormation.resize(contextSetTracking.size());
            for (size_t i = 0; i < formation.agentsInFormation.size(); ++i) {
                formation.agentsInFormation[i].trajectory.resize(contextSetTracking.size());
                formation.agentsInFormation[i].setState(formation.agentsInFormation[i].trajectory.back());
            }
        }
        return contextSetTracking;
    }
}


// ---------------------------------------------------------------------------
// Returns a visit sequence of *all* landmarks such that, at each step,
// intersecting the current belief with that landmark’s observation‐subset
// shrinks the belief as much as possible.
// ---------------------------------------------------------------------------
inline std::vector<LocalState> getLandmarkVisitSequence(const Environment &env) {
    // 1) Start with your full belief‐set
    std::set<int> currentBelief = env.getBeliefContextSet();

    // 2) Grab all landmark positions and the map → {landmark → context‐subset}
    auto landmarks   = env.getAllLandmarkLocalStates();
    auto lmCtxMap    = env.getLandmarkToContextSubsetMap();

    std::vector<LocalState> sequence;
    sequence.reserve(landmarks.size());
    std::vector<LocalState> remaining = landmarks;

    // 3) While we still have unvisited landmarks…
    while (!remaining.empty()) {
        size_t bestIdx      = 0;
        std::set<int> bestInter;
        bool firstCandidate = true;

        // 4) For each remaining landmark, compute
        //      inter = currentBelief ∩ obsSubset(landmark)
        //    and pick the one with *smallest* inter.size()
        for (size_t i = 0; i < remaining.size(); ++i) {
            const auto &obs = lmCtxMap[ remaining[i] ];
            std::set<int> inter;
            std::set_intersection(
                currentBelief.begin(), currentBelief.end(),
                obs.begin(),           obs.end(),
                std::inserter(inter, inter.begin())
            );

            if (firstCandidate || inter.size() < bestInter.size()) {
                firstCandidate = false;
                bestInter      = std::move(inter);
                bestIdx        = i;
            }
        }

        // 5) “Visit” that landmark next…
        sequence.push_back(remaining[bestIdx]);
        // …and update belief to the intersection we just computed
        currentBelief = std::move(bestInter);
        // remove it from remaining
        remaining.erase(remaining.begin() + bestIdx);
    }

    return sequence;
}

// ---------------------------------------------------------------------------
// Returns a visit sequence of *all* landmarks such that, at each step,
// intersecting the current belief with that landmark’s observation‐subset
// shrinks the belief as *slowly* as possible. In other words, it greedily
// delays collapse to a singleton by:
//   (1) Prefering landmarks whose intersection keeps |belief| > 1;
//   (2) Among those, picking the one with *largest* intersection.
// ---------------------------------------------------------------------------
inline std::vector<LocalState> getMostUnhelpfulLandmarkVisitSequence(const Environment &env) {
    // 1) Start with your full belief‐set
    std::set<int> currentBelief = env.getBeliefContextSet();

    // 2) Grab all landmark positions and the map → {landmark → context‐subset}
    auto landmarks = env.getAllLandmarkLocalStates();
    auto lmCtxMap  = env.getLandmarkToContextSubsetMap();

    std::vector<LocalState> sequence;
    sequence.reserve(landmarks.size());
    std::vector<LocalState> remaining = landmarks;

    // 3) While we still have unvisited landmarks…
    while (!remaining.empty()) {
        size_t chosenIdx = 0;
        std::set<int> chosenInter;
        bool haveCandidate       = false;
        bool haveNonCollapsing   = false; // whether we found |inter| > 1

        // 4) For each remaining landmark, compute:
        //        inter = currentBelief ∩ obsSubset(landmark)
        //    and pick:
        //      - preferably one with |inter| > 1, maximizing |inter|;
        //      - if none have |inter| > 1, pick among the rest (|inter| ≤ 1),
        //        e.g. maximizing |inter| to collapse as late as possible.
        for (size_t i = 0; i < remaining.size(); ++i) {
            const auto &obs = lmCtxMap[ remaining[i] ];
            std::set<int> inter;
            std::set_intersection(
                currentBelief.begin(), currentBelief.end(),
                obs.begin(),           obs.end(),
                std::inserter(inter, inter.begin())
            );

            const std::size_t sz = inter.size();

            if (sz > 1) {
                // Prefer non-collapsing intersections (|inter| > 1),
                // and among them keep the *largest*.
                if (!haveNonCollapsing || sz > chosenInter.size()) {
                    haveNonCollapsing = true;
                    haveCandidate     = true;
                    chosenInter       = std::move(inter);
                    chosenIdx         = i;
                }
            } else if (!haveNonCollapsing) {
                // Only consider |inter| ≤ 1 if we have *no* non-collapsing
                // candidate at all; among these, maximize |inter| as well.
                if (!haveCandidate || sz > chosenInter.size()) {
                    haveCandidate = true;
                    chosenInter   = std::move(inter);
                    chosenIdx     = i;
                }
            }
        }

        // Safety: if something went wrong and we have no candidate, break.
        if (!haveCandidate) {
            break;
        }

        // 5) “Visit” that landmark next…
        sequence.push_back(remaining[chosenIdx]);
        // …and update belief to the intersection we just computed
        currentBelief = std::move(chosenInter);
        // remove it from remaining
        remaining.erase(remaining.begin() + chosenIdx);
    }

    return sequence;
}


// ---------------------------------------------------------------------------
// Returns a visit sequence of *all* landmarks such that, at each step,
// intersecting the current belief with that landmark’s observation‐subset
// shrinks the belief as much as possible.
// ---------------------------------------------------------------------------
inline std::vector<LocalState> getLandmarkVisitSequence_ARVI(const Environment &env) {
    // these agents cannot see what landmark is more informative, so we randomize the order
    // 1) Start with your full belief‐set
    std::set<int> currentBelief = env.getBeliefContextSet();
    // 2) Grab all landmark positions and the map → {landmark → context‐subset}
    auto landmarks   = env.getAllLandmarkLocalStates();
    auto lmCtxMap    = env.getLandmarkToContextSubsetMap();
    std::vector<LocalState> sequence;
    sequence.reserve(landmarks.size());
    std::vector<LocalState> remaining = landmarks;
    // random engine
    std::random_device rd;
    std::mt19937 g(rd());
    // shuffle remaining landmarks
    std::shuffle(remaining.begin(), remaining.end(), g);
    // 3) While we still have unvisited landmarks… 
    while (!remaining.empty()) {
        size_t bestIdx      = 0;
        std::set<int> bestInter;
        bool firstCandidate = true;
        // 4) For each remaining landmark, compute
        //      inter = currentBelief ∩ obsSubset(landmark)
        //    and pick the one with *smallest* inter.size()
        for (size_t i = 0; i < remaining.size(); ++i) {
            const auto &obs = lmCtxMap[ remaining[i] ];
            std::set<int> inter;
            std::set_intersection(
                currentBelief.begin(), currentBelief.end(),
                obs.begin(),           obs.end(),
                std::inserter(inter, inter.begin())
            );
            if (firstCandidate || inter.size() < bestInter.size()) {
                firstCandidate = false;
                bestInter      = std::move(inter);
                bestIdx        = i;
            }
        }
        // 5) “Visit” that landmark next…
        sequence.push_back(remaining[bestIdx]);
        // …and update belief to the intersection we just computed
        currentBelief = std::move(bestInter);
        // remove it from remaining
        remaining.erase(remaining.begin() + bestIdx);
    }
    
    return sequence;
}

// SAIA: sampling-based non-myopic landmark ordering.
inline std::vector<LocalState> getLandmarkVisitSequence_SAIA(const Environment& env, unsigned seed = 0) {
    const auto& lm_map = env.getLandmarkToContextSubsetMap();
    std::vector<LocalState> remaining = env.getAllLandmarkLocalStates();
    std::set<int> belief = PossibleContexts;
    std::vector<LocalState> sequence;
    std::mt19937 rng(seed == 0 ? 1u : seed);
    constexpr int kSamples = 32;

    while (belief.size() > 1 && !remaining.empty()) {
        double best_gain = -1.0;
        int best_idx = 0;
        std::uniform_int_distribution<int> pick(0, static_cast<int>(remaining.size()) - 1);
        for (int s = 0; s < kSamples; ++s) {
            const int idx = pick(rng);
            const LocalState& lm = remaining[static_cast<size_t>(idx)];
            const auto it = lm_map.find(lm);
            if (it == lm_map.end()) continue;
            const auto next = macussp::belief_intersect(belief, it->second);
            const double gain = static_cast<double>(macussp::belief_entropy_cardinality(belief))
                              - static_cast<double>(macussp::belief_entropy_cardinality(next));
            if (gain > best_gain) {
                best_gain = gain;
                best_idx = idx;
            }
        }
        const LocalState chosen = remaining[static_cast<size_t>(best_idx)];
        sequence.push_back(chosen);
        belief = macussp::belief_intersect(belief, lm_map.at(chosen));
        remaining.erase(remaining.begin() + best_idx);
    }
    return sequence;
}

inline std::vector<LocalState> getLandmarkInformativeSequence(const Environment &env, std::string SOLVER) {
    if (SOLVER == "OURS") {
        return getLandmarkVisitSequence(env);
    }
    else if (SOLVER == "ARVI"){
        return getMostUnhelpfulLandmarkVisitSequence(env);
    }
    else if (SOLVER == "SAIA") {
        return getLandmarkVisitSequence_SAIA(env, 0);
    }
    else {
        std::cerr << "[high_level_planner] Error: Unknown SOLVER for landmark visit sequence: " << SOLVER << std::endl;
        exit(1);
    }
}
// // Assign agents into numOfGroups of size 5 each, based on proximity to landmarks in visit sequence.
// // Returns a map from agent ID to group ID (0..numOfGroups-1).
inline std::unordered_map<int,int> assignGroups_proximity(const Environment &env, const std::vector<Agent> &agents, int numOfGroups) {
    auto landmarks = getLandmarkVisitSequence(env);
    // // print the landmark visit sequence
    // std::cout << "\033[1;33m[Before]Landmark visit sequence:\033[0m" << std::endl;
    // for (const auto &lm : landmarks) {
    //     std::cout << "(" << lm.x << ", " << lm.y << ", " << lm.type << ")\n";
    // }

    // if more landmarks than groups, truncate
    if ((int)landmarks.size() > numOfGroups)
        landmarks.resize(numOfGroups);

    // copy agents into working list
    std::vector<Agent> unassigned = agents;
    std::unordered_map<int,int> agent2group;
    agent2group.reserve(agents.size());

    for (int gid = 0; gid < numOfGroups; ++gid) {
        const LocalState &lm = landmarks[gid];
        // sort unassigned by actual distance to this landmark
        std::sort(unassigned.begin(), unassigned.end(), [&](const Agent &a, const Agent &b) {
            auto sa = a.getState(); auto sb = b.getState();
            float da = std::sqrt((sa.x - lm.x) * (sa.x - lm.x) + (sa.y - lm.y) * (sa.y - lm.y));
            float db = std::sqrt((sb.x - lm.x) * (sb.x - lm.x) + (sb.y - lm.y) * (sb.y - lm.y));
            return da < db;
        });
        // take up to 5 closest
        int take = std::min(5, (int)unassigned.size());
        for (int i = 0; i < take; ++i) {
            agent2group[ unassigned[i].getId() ] = gid;
        }
        // erase those assigned
        unassigned.erase(unassigned.begin(), unassigned.begin() + take);
    }
    return agent2group;
}
// –– Balanced K‑Means assignGroups ––
//   Clusters agents into `numOfGroups` groups of exactly 5 each.
//   1) Run a few iterations of K‑means to get centroids.
//   2) For each centroid in turn, pick its 5 nearest *unassigned* agents.
// Returns map<agentID, groupID>.
inline std::unordered_map<int,int> assignGroups_kmeans(const Environment &env,
                                                const std::vector<Agent> &agents,
                                                int numOfGroups) {
    const int N = agents.size();
    const int clusterSize = N / numOfGroups;  // assume divisible, i.e. 5

    // 1) Extract points
    struct P { double x,y; };
    std::vector<P> pts(N);
    for (int i = 0; i < N; ++i) {
        auto s = agents[i].getState();
        // std::cout << "\n - Agent " << agents[i].getId() << ": ";
        // printLocalState(s);
         
        pts[i] = { double(s.x), double(s.y) };
    }

    // 2) Initialize K centroids by choosing first K points (could use kmeans++)
    std::vector<P> centroids(numOfGroups);
    for (int k = 0; k < numOfGroups; ++k)
        centroids[k] = pts[k];

    // 3) Run a few Lloyd iterations *without* enforcing balance
    for (int iter = 0; iter < 5; ++iter) {
        std::vector<std::vector<int>> clusters(numOfGroups);
        // assign each point to nearest centroid
        for (int i = 0; i < N; ++i) {
            double bestD = 1e18;
            int   bestK = 0;
            for (int k = 0; k < numOfGroups; ++k) {
                double dx = pts[i].x - centroids[k].x;
                double dy = pts[i].y - centroids[k].y;
                double d2 = dx*dx + dy*dy;
                if (d2 < bestD) { bestD = d2; bestK = k; }
            }
            clusters[bestK].push_back(i);
        }
        // recompute centroids
        for (int k = 0; k < numOfGroups; ++k) {
            if (clusters[k].empty()) continue;
            double sx = 0, sy = 0;
            for (int i : clusters[k]) {
                sx += pts[i].x;
                sy += pts[i].y;
            }
            centroids[k] = { sx / clusters[k].size(), sy / clusters[k].size() };
        }
    }

    // 4) Balanced assignment: for each centroid, pick its clusterSize nearest unassigned
    std::unordered_map<int,int> agent2group;
    agent2group.reserve(N);
    std::vector<bool> assigned(N,false);

    for (int k = 0; k < numOfGroups; ++k) {
        // compute distances to centroid k
        std::vector<std::pair<double,int>> distIdx;
        distIdx.reserve(N);
        for (int i = 0; i < N; ++i) if (!assigned[i]) {
            double dx = pts[i].x - centroids[k].x;
            double dy = pts[i].y - centroids[k].y;
            distIdx.emplace_back(dx*dx + dy*dy, i);
        }
        // pick the clusterSize nearest
        std::nth_element(distIdx.begin(),
                         distIdx.begin() + clusterSize,
                         distIdx.end(),
                         [](const std::pair<double,int> &a, const std::pair<double,int> &b){ return a.first < b.first; });

        for (int t = 0; t < clusterSize; ++t) {
            int idx = distIdx[t].second;
            assigned[idx] = true;
            agent2group[ agents[idx].getId() ] = k;
        }
    }

    return agent2group;
}
// Build a map from group ID to the landmark LocalState assigned to that group.
// This should match the grouping in assignGroups (i.e., group 0 gets sequence[0], etc.).
inline std::unordered_map<int, LocalState> getMapGroupIDToLandmarkState(const std::vector<Agent> &agents, const std::unordered_map<int,int> &agentID2groupID, const Environment &env) {
    // derive number of groups from max groupID + 1
    int maxG = -1;
    for (auto &p : agentID2groupID) maxG = std::max(maxG, p.second);
    int numGroups = maxG + 1;

    auto landmarks = getLandmarkVisitSequence(env);
    if ((int)landmarks.size() > numGroups)
        landmarks.resize(numGroups);

    std::unordered_map<int, LocalState> group2lm;
    for (int gid = 0; gid < numGroups; ++gid) {
        group2lm[gid] = landmarks[gid];
    }
    return group2lm;
}


// Build a map from group ID to the landmark LocalState assigned to that group.
// This should match the grouping in assignGroups (i.e., group 0 gets sequence[0], etc.).
inline std::unordered_map<int, LocalState> getMapGroupIDToLandmarkState2(std::unordered_map<int, LocalState> groupIDToLandmarkState, const std::unordered_map<int, std::vector<Agent>> groupsIdToAgentsVectorMap, std::vector<LocalState> &landmark_sequence) {
    // go throtuh the priprity landmark states in order of landmark_sequence and find the groupID whose agents' mean state location that is closest to that landmark and assign it to that group and then pop that landmark from the sequence and do this till all groups are assigned
    std::unordered_map<int, LocalState> group2lm;
    std::vector<int> already_assigned_groups = {};
    for (int landmark_idx = 0; landmark_idx < landmark_sequence.size(); ++landmark_idx) {
        if (already_assigned_groups.size() == groupsIdToAgentsVectorMap.size()) {
            break; // all groups have been assigned a landmark
        }
        LocalState lm = landmark_sequence[landmark_idx];
        int best_groupID = -1;
        double min_distance = std::numeric_limits<double>::max();
        for (auto &group : groupsIdToAgentsVectorMap) {
            int groupID = group.first;
            // check if this group has already been assigned a landmark
            if (std::find(already_assigned_groups.begin(), already_assigned_groups.end(), groupID) != already_assigned_groups.end()) {
                continue; // skip this group
            }
            std::vector<Agent> agents = group.second;
            // compute the mean state of the agents in this group
            double sum_x = 0, sum_y = 0;
            for (auto &agent : agents) {
                sum_x += agent.getState().x;
                sum_y += agent.getState().y;
            }
            double mean_x = sum_x / agents.size();
            double mean_y = sum_y / agents.size();
            // compute the distance to the landmark
            double distance = std::sqrt((mean_x - lm.x) * (mean_x - lm.x) + (mean_y - lm.y) * (mean_y - lm.y));
            if (distance < min_distance) {
                min_distance = distance;
                best_groupID = groupID;
            }
        }                   
        // assign the landmark to the best group
        group2lm[best_groupID] = lm;
        already_assigned_groups.push_back(best_groupID);
    }
    // for groups that are not assigned (not in already_assigned_groups) just assign the same landmark the were assigned in groupIDToLandmarkState
    for (auto &group : groupsIdToAgentsVectorMap) {
        int groupID = group.first;
        if (std::find(already_assigned_groups.begin(), already_assigned_groups.end(), groupID) == already_assigned_groups.end()) {
            group2lm[groupID] = groupIDToLandmarkState[groupID];
        }
    }
    // remove the assigned landmarks from the landmark_sequence
    for (auto &group : group2lm) {
        auto it = std::find(landmark_sequence.begin(), landmark_sequence.end(), group.second);
        if (it != landmark_sequence.end()) {
            landmark_sequence.erase(it);
        }
    }
    return group2lm;  
}

inline std::pair<bool, std::unordered_map<int, LocalState>> getTaskGoals4AllAgents(Environment &env, const std::vector<Agent> &agents){
    
    std::unordered_map<int, LocalState> agent2goalStateMap;
    bool all_tasks_assigned = false;
    std::string mapName = env.getMapName();
    if (mapName == "salp" or mapName == "warehouse"){
        for (const auto &agent : agents) {
            int agentId = agent.getId();
            agent2goalStateMap[agentId] = agent.getGoal();
            // print agent IDs ab current state of the agents
            std::cout << "\033[1;30m[h_l_p.h l612] Agent ID: " << agentId << ", Current State: ";
            printLocalState(agent.getState());
            std::cout << ", Goal State: ";
            printLocalState(agent.getGoal());
            std::cout << "\033[0m" << std::endl;

        } 
        all_tasks_assigned = true;  // since salp domain is only go-to single-goal, we can assume all tasks are assigned
        return std::make_pair(all_tasks_assigned, agent2goalStateMap);
    }
    else if (mapName == "forestfire"){
        // lets check if number of agents/4 is equal to number of fires in the environment, if so we can assign each group of 4 agents to a fire landmark
        // if num_agents < num_fires * 4, then the agents cath the closest fire first and then move on to other fires till all are done in the next task for them
        std::cout << "@@@@@@@@@@@@@@@@@@@[h_l_p.h l501]#Agents: " << agents.size() << ", #Fires: " << env.fires_.size() << std::endl;
        if (agents.size() == env.fires_.size() * 4) { 
            int time_step_of_decision = agents[0].trajectory.size() - 1; // assuming all agents have the same length of trajectory after synchronization till belief collapse
            std::unordered_map<int, std::vector<Agent>> fireIdToAgentsVectorMap;
            for (const auto &fire : env.fires_) {
                fireIdToAgentsVectorMap[fire.id] = {};
            }
            // compute of distance of all agents for each fire and assign the closest 4 agents to that fire without assigning the same agent to multiple fires (cannot exceed 4 agents per fire)
            for (const auto &agent : agents) {
                int agentId = agent.getId();
                LocalState agentState = agent.getState();
                double minDistance = std::numeric_limits<double>::max();
                int closestFireId = -1;
                for (const auto &fire : env.fires_) {
                    auto [xf, yf] = env.firePosAt(fire.id, time_step_of_decision);
                    double distance = std::sqrt((agentState.x - xf) * (agentState.x - xf) + (agentState.y - yf) * (agentState.y - yf));
                    if (distance < minDistance && fireIdToAgentsVectorMap[fire.id].size() < 4) {
                        minDistance = distance;
                        closestFireId = fire.id;
                    }
                }
                if (closestFireId != -1) {
                    fireIdToAgentsVectorMap[closestFireId].push_back(agent);
                }
            }
            for (const auto &fire : env.fires_) {
                std::unordered_map<int, LocalState> agent2goalStateMap4thisFire = ff_helper::computeInterceptGoalsOnly(env, fireIdToAgentsVectorMap[fire.id], env.H_MAX, fire.id);
                for (const auto &pair : agent2goalStateMap4thisFire) {
                    agent2goalStateMap[pair.first] = pair.second;
                }
            }
            return std::make_pair(true, agent2goalStateMap);
        }
        else if (agents.size() < env.fires_.size() * 4) {
            // lets assign each agent to the closest fire first and then move on to other fires till all are done and check the entinguish flag on them
            int maxLenT = 0;
            for (const auto &agent : agents) {
                if (agent.trajectory.size() > maxLenT) {
                    maxLenT = agent.trajectory.size();
                }
            }
            std::cout << "\033[1;30m[h_l_p.h l425] Debug1: maxLenT = " << maxLenT << "\033[0m" << std::endl; 
            int time_step_of_decision = std::max(0,maxLenT-1); // assuming all agents have the same length of trajectory after synchronization till belief collapse
            std::set<int> assignedAgents;
            // since num of firesa re more that the num of groups, agents will be assigned with outer vloop variable as fire and agent in the inner loop 
            // since a fire should have 4 agents and no less or more
            std::unordered_map<int, std::vector<Agent>> fireIdToAgentsVectorMap;
            for (const auto &fire : env.fires_) {
                fireIdToAgentsVectorMap[fire.id] = {};
            }
            for (const auto &fire : env.fires_) {
                int agents_assigned_to_this_fire = 0;
                if (env.fireActiveAt(fire.id, time_step_of_decision)) {
                    continue; // skip this fire if it is already extinguished
                }
                while (agents_assigned_to_this_fire < 4 && assignedAgents.size() < agents.size()) {
                    double minDistance = std::numeric_limits<double>::max();
                    int closestAgentId = -1;
                    for (const auto &agent : agents) {
                        int agentId = agent.getId();
                        // check if this agent is already assigned to a fire
                        if (assignedAgents.find(agentId) != assignedAgents.end()) {
                            continue; // skip this agent
                        }
                        LocalState agentState = agent.getState();
                        auto [xf, yf] = env.firePosAt(fire.id, time_step_of_decision);
                        double distance = std::sqrt((agentState.x - xf) * (agentState.x - xf) + (agentState.y - yf) * (agentState.y - yf));
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestAgentId = agentId;
                        }
                    }
                    if (closestAgentId != -1) {
                        // assign this agent to this fire
                        auto it = std::find_if(agents.begin(), agents.end(), [&](const Agent &a){ return a.getId() == closestAgentId; });
                        if (it != agents.end()) {
                            fireIdToAgentsVectorMap[fire.id].push_back(*it);
                            assignedAgents.insert(closestAgentId);
                            agents_assigned_to_this_fire++;
                        }
                    }
                    else {
                        break; // no more unassigned agents
                    }
                }
            }
            for (const auto &fire : env.fires_) {
                std::unordered_map<int, LocalState> agent2goalStateMap4thisFire = ff_helper::computeInterceptGoalsOnly(env, fireIdToAgentsVectorMap[fire.id], env.H_MAX, fire.id);
                for (const auto &pair : agent2goalStateMap4thisFire) {
                    agent2goalStateMap[pair.first] = pair.second;
                }
            }
            for (const auto &agent : agents) {
                int agentId = agent.getId();
                // check if this agent is already assigned to a fire
                if (assignedAgents.find(agentId) == assignedAgents.end()) {
                    // this agent is not assigned to any fire, so assign it to its current position as goal
                    agent2goalStateMap[agentId] = agent.getState();
                }
            }
            return std::make_pair(true, agent2goalStateMap);
        }
        else {
            std::cerr << "\033[1;31mError: Number of agents is not equal to number of fires * 4 in forestfire domain. Cannot assign tasks.\033[0m" << std::endl;
            return std::make_pair(false, agent2goalStateMap);
        }       
    }
}


}; // namespace high_level_planner

#endif // HIGH_LEVEL_PLANNER_H
