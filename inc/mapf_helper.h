#ifndef MAPF_HELPER_H
#define MAPF_HELPER_H

#include "mapf_solver.h"
#include "agent.h"
#include "state.h"
#include "environment.h"
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <array>
#include <deque>
#include <climits>
#include <functional>
#include <iomanip>
#include <algorithm>

// Forward declarations so we can call these in this header regardless of include order.
std::vector<int> diagonalRingGoalsAtT(const Environment& env, int t);                 // legacy: single-fire wrapper
std::vector<int> diagonalRingGoalsAtT(const Environment& env, int fire_id, int t);    // multi-fire

int ff_earliestArrivalByT(const Environment& env, int start_v, int goal_v, int T);


// forbid occupying the fire vertex at time t
inline bool ff_isVertexFreeAtTime(const Environment& env, int v, int t) {
    if (!(env.domain() == DomainType::ForestFire && env.fireEnabled())) return true;
    if (t < 0) return false;
    for (int f = 0; f < env.numFires(); ++f) {
        if (env.fireVidAt(f, t) == v) return false;
    }
    return true;
}
// time-expanded BFS: earliest arrival time to goal_v within horizon T (wait allowed)
inline int ff_earliestArrivalByT(const Environment& env, int start_v, int goal_v, int T) {
    if (T < 0 || start_v < 0 || goal_v < 0) return INT_MAX;
    const int Vn = (int)env.vertices().size();
    const int stride = T + 1;
    std::vector<char> seen((size_t)Vn * (size_t)stride, 0);
    auto IDX = [&](int v,int t){ return v*stride + t; };

    if (!ff_isVertexFreeAtTime(env, start_v, 0)) return INT_MAX;

    std::deque<std::pair<int,int>> dq;
    dq.emplace_back(start_v, 0);
    seen[IDX(start_v,0)] = 1;

    while (!dq.empty()) {
        auto [v,t] = dq.front(); dq.pop_front();
        if (v == goal_v) return t;
        if (t == T) continue;

        // wait
        if (ff_isVertexFreeAtTime(env, v, t+1)) {
            int k = IDX(v, t+1);
            if (!seen[k]) { seen[k] = 1; dq.emplace_back(v, t+1); }
        }
        // move
        for (int u : env.neighbors(v)) {
            if (!ff_isVertexFreeAtTime(env, u, t+1)) continue;
            int k = IDX(u, t+1);
            if (!seen[k]) { seen[k] = 1; dq.emplace_back(u, t+1); }
        }
    }
    return INT_MAX;
}


// -------- ForestFire: dump per-frame fire trajectory up to capture T* --------
namespace ff_helper {

    // Local helpers (no symbol collisions)
    inline bool isFreeAtTime(const Environment& env, int v, int t) {
        if (!(env.domain() == DomainType::ForestFire && env.fireEnabled())) return true;
        if (t < 0) return false;
        const auto& n = env.node(v);
        auto [fx, fy] = env.firePosAt(std::max(0, t));
        return !(n.x == fx && n.y == fy);
    }

    inline bool maxMatch4(const std::vector<std::vector<int>>& adjL, int nAgents,
                        std::array<int,4>& diag2agent) {
        const int L = (int)adjL.size();
        std::vector<int> matchR(nAgents, -1);
        std::function<bool(int,std::vector<char>&)> dfs = [&](int l, std::vector<char>& vis)->bool{
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
        diag2agent.fill(-1);
        for (int r=0; r<nAgents; ++r) if (matchR[r] != -1) {
            int l = matchR[r];
            if (l>=0 && l<4) diag2agent[l] = r;
        }
        return (m == 4);
    }


    // - env = Environment object (with fire trajectory already generated).
    // - n_frames = total animation frames (length of agent trajectories).
    // - move_period = frames per one fire step (2 for half-speed).
    inline bool writeFireTrajectory(const Environment& env, int n_frames, const std::string& path){
        /*writes the final fire joint trajectory to path:= ..../fire_trajectory.json*/

        std::ofstream out(path);
        if (!out) { std::cerr << "[ff_helper] Cannot open " << path << " for writing.\n"; return false; }

        const int K = env.numFires();
        out << "[\n";
        for (int frame = 0; frame < n_frames; ++frame) {
            out << "  [";
            for (int f = 0; f < K; ++f) {
                // clamp to last valid t for this fire
                // (if your generator guarantees length >= n_frames, this is just extra safety)
                auto &F = env.fires_[f]; 
                const int last_t = (int)F.xs.size() - 1;
                const int tf = std::max(0, std::min(frame, last_t));

                auto [fx, fy] = (tf < F.time_of_extinguish) ? env.firePosAt(f, tf) : env.firePosAt(f, F.time_of_extinguish);
                out << "[" << fx << ", " << fy << ", " << (tf >= F.time_of_extinguish) << "]";
                if (f < K - 1) out << ", ";
            }
            out << "]";
            if (frame < n_frames - 1) out << ",";
            out << "\n";
        }
        out << "]\n";
        return true;
    }

    inline std::unordered_map<int, LocalState> computeInterceptGoalsOnly(Environment& env, 
                                                                         const std::vector<Agent>& agents, 
                                                                         int Hmax,
                                                                         int fire_id){

        std::unordered_map<int, LocalState> agent2GoalStateMap;

        // Current time t0 = agents’ executed trajectory length (assumed same for all).
        const int t0 = (int)agents[0].trajectory.size();

        // Helper: passability test via env API (treat only hard walls as blocked; adjust if needed).
        auto passable = [&](int x, int y) -> bool {
            int vid = env.nodeIdAt(x, y);
            if (vid < 0) return false;
            char ty = env.node(vid).type;
            return (ty != '@'); // adjust to your legend if necessary
        };

        // Bounded 4-neighbor BFS distance in grid steps; returns INF (large) if > bound or unreachable.
        auto shortestDistBounded = [&](int sx, int sy, int gx, int gy, int bound_steps) -> int {
            if (sx == gx && sy == gy) return 0;
            const int INF = 1e9;
            std::queue<std::pair<int,int>> q;
            std::unordered_map<long long,int> dist;
            auto key = [](int x, int y)->long long {
                return ( (static_cast<long long>(x) << 32) ^ static_cast<unsigned>(y) );
            };
            static const int dx[4] = {+1,-1,0,0};
            static const int dy[4] = {0,0,+1,-1};
            if (!passable(sx, sy) || !passable(gx, gy)) return INF;

            dist[key(sx,sy)] = 0;
            q.emplace(sx,sy);
            while(!q.empty()){
                auto [x,y] = q.front(); q.pop();
                int d = dist[key(x,y)];
                if (d >= bound_steps) continue; // no need to expand beyond time budget
                for (int k=0;k<4;k++){
                    int nx = x + dx[k], ny = y + dy[k];
                    if (!passable(nx,ny)) continue;
                    long long kk = key(nx,ny);
                    if (dist.find(kk) != dist.end()) continue;
                    dist[kk] = d+1;
                    if (nx == gx && ny == gy) return d+1;
                    q.emplace(nx,ny);
                }
            }
            return INF;
        };

        // Iterate earliest feasible t
        for (int t = t0; t <= Hmax; ++t) {
            // 1) diagonal ring at time t
            auto diag_vids = diagonalRingGoalsAtT(env, fire_id, t);
            if ((int)diag_vids.size() < 4) continue;

            // Get coordinates of the 4 diagonal cells
            std::array<std::pair<int,int>,4> diag_xy;
            for (int d = 0; d < 4; ++d) {
                const auto& nd = env.node(diag_vids[d]);
                diag_xy[d] = {nd.x, nd.y};
            }

            // Infer fire center (average of 4 diagonal corners).
            int fx = 0, fy = 0;
            for (int d = 0; d < 4; ++d) { fx += diag_xy[d].first; fy += diag_xy[d].second; }
            fx /= 4; fy /= 4;

            // 2) Build 4x4 orientation cost matrix: cost = 1 - cos(angle)
            // Agent direction = from fire center to agent; Diagonal direction = from fire center to that diagonal cell
            double cost[4][4];
            for (int i = 0; i < 4; ++i) {
                int ax = agents[i].getState().x;
                int ay = agents[i].getState().y;
                double vx = static_cast<double>(ax - fx);
                double vy = static_cast<double>(ay - fy);
                double anorm = std::sqrt(vx*vx + vy*vy);
                for (int d = 0; d < 4; ++d) {
                    int dx = diag_xy[d].first  - fx;
                    int dy = diag_xy[d].second - fy;
                    double dnorm = std::sqrt(static_cast<double>(dx*dx + dy*dy));
                    double c = 2.0; // worst default if degenerate
                    if (anorm > 0.0 && dnorm > 0.0) {
                        double dot = vx*dx + vy*dy;
                        double cosv = dot / (anorm * dnorm);
                        // numerical clamp
                        if (cosv > 1.0) cosv = 1.0;
                        if (cosv < -1.0) cosv = -1.0;
                        c = 1.0 - cosv;
                    }
                    cost[i][d] = c;
                }
            }

            // 3) Find orientation-closest one-to-one assignment (4! enumeration)
            std::array<int,4> perm = {0,1,2,3};
            std::array<int,4> best_perm = perm;
            double best_sum = 1e18;
            do {
                double s = 0.0;
                for (int i=0;i<4;++i) s += cost[i][perm[i]];
                if (s < best_sum) { best_sum = s; best_perm = perm; }
            } while (std::next_permutation(perm.begin(), perm.end()));

            // 4) Feasibility check: shortest path length <= (t - t0) for each agent->assigned diag
            bool feasible = true;
            std::array<int,4> chosen_vids;
            for (int i=0;i<4;++i) {
                int d   = best_perm[i];
                int gv  = diag_vids[d];
                auto& g = env.node(gv);
                int sx  = agents[i].getState().x;
                int sy  = agents[i].getState().y;
                int budget = t - t0;
                int dist = shortestDistBounded(sx, sy, g.x, g.y, budget);
                if (dist > budget) { feasible = false; break; }
                chosen_vids[i] = gv;
            }
            if (!feasible) continue;
            // so same the time_of_extinguish for the fire_id at time t, which is the earliest feasible time for the agents to reach their assigned diagonal goals
            env.setFireExtinguishTime(fire_id, t);

            // 5) Commit goals (earliest feasible t)
            for (int i=0;i<4;++i) {
                int gv = chosen_vids[i];
                const auto& n = env.node(gv);
                agent2GoalStateMap[agents[i].getId()] = LocalState(n.x, n.y, n.type);
            }
            return agent2GoalStateMap;
        }

        // No feasible assignment up to Hmax: unchanged
        return agent2GoalStateMap;
    }
}; // namespace ff_helper


void createScenarioFile(const std::string &scenarioFileName,
                        const std::vector<Agent> &agents,
                        const std::unordered_map<int, LocalState> &agent2TargetStatesMap,
                        const Environment &env);
std::unordered_map<int, std::vector<LocalState>> runMapfSolver( const std::string &algorithm,
                                                                Environment &env,
                                                                const std::vector<Agent> &agents,
                                                                const std::unordered_map<int, LocalState> &agent2TargetStatesMap,
                                                                int timeLimitSec,
                                                                std::string scenarioFileName);
std::vector<int> diagonalRingGoalsAtT(const Environment& env, int fire_id, int t);
inline std::vector<int> diagonalRingGoalsAtT(const Environment& env, int t) {
    return diagonalRingGoalsAtT(env, /*fire_id=*/0, t);
}

#endif // MAPF_HELPER_H