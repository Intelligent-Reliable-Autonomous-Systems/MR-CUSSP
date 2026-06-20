#include "InfoGathering/ARVI.h"
#include "formation.h"
#include "Definitions.h"
#include <queue>
#include <limits>
#include <algorithm>
#include <cassert>
#include <array>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

// ============================
// Small internal helper utils
// ============================
static inline long long CKEY(int x,int y){ return ((long long)x<<32) ^ (unsigned long long)y; }

// Pack (t,x,y) into a 64-bit key. We assume non-negative grid indices and t.
static inline long long TKEY(int t,int x,int y) {
  // 21 bits per coordinate (>= 2M), 22 bits for t (>= 4M steps)
  // [ t:22 | x:21 | y:21 ] => 64 bits
  return ( ( (long long)t & ((1LL<<22)-1) ) << 42 )
       ^ ( ( (long long)x & ((1LL<<21)-1) ) << 21 )
       ^ ( ( (long long)y & ((1LL<<21)-1) ) );
}

static inline void UNPACK_TXY(long long k, int& t,int& x,int& y) {
  y =  (int)( k & ((1LL<<21)-1) );
  x =  (int)( (k >> 21) & ((1LL<<21)-1) );
  t =  (int)( (k >> 42) & ((1LL<<22)-1) );
}

// Build stencil slots (domain-specific). Kept here to keep module self-contained.
static std::vector<LocalState>
getStencilSlotsForFormationImpl(const Formation& F, Environment& env)
{
  std::vector<LocalState> S;
  const int n = (int)F.agentsInFormation.size();
  const auto& domain = env.getMapName();

  if (domain == "forestfire") {
    // One unique landmark per agent (by agent id if available)
    auto Ls = env.getAllLandmarkLocalStates();
    for (auto& a : F.agentsInFormation) {
      int idx = a.getId();
      if (idx >= 0 && idx < (int)Ls.size()) S.push_back(Ls[idx]);
      else S.push_back(env.getLocalState(F.landmarkGoalState.x, F.landmarkGoalState.y));
    }
  }
  else if (domain == "warehouse") {
    // k cells below barcode; adjust pattern if needed
    int x = F.landmarkGoalState.x, y = F.landmarkGoalState.y;
    for (int k=1; k<=n; ++k) S.push_back(env.getLocalState(x, y+k));
  }
  else if (domain == "salp") {
    int x0 = F.landmarkGoalState.x, y0 = F.landmarkGoalState.y;
    if (F.formationName == "chain") {
      int off=(n-1)/2;
      for(int i=0;i<n;++i) S.push_back(env.getLocalState(x0-off+i, y0));
    } else if (F.formationName == "cross" && n==5) {
      std::array<std::pair<int,int>,5> offs = {{{-1,-1},{+1,-1},{0,0},{-1,+1},{+1,+1}}};
      for (auto &o: offs) S.push_back(env.getLocalState(x0+o.first, y0+o.second));
    } else {
      // Fallback: everyone at landmark cell (reservations enforce uniqueness)
      for (int i=0;i<n;++i) S.push_back(env.getLocalState(x0,y0));
    }
  }
  else {
    // Unknown domain: default to the landmark cell repeated n times
    for (int i=0;i<n;++i) S.push_back(env.getLocalState(F.landmarkGoalState.x, F.landmarkGoalState.y));
  }
  return S;
}

// =======================
// Core single-agent ARVI
// =======================
std::vector<LocalState>
runARVIToAnySlot(const Agent& a,
                 const std::vector<LocalState>& S,
                 const std::unordered_set<long long>& claimed,
                 Environment& env,
                 const ReservationTable& R,
                 const ARVIParams& p)
{
  // Frontier: best-first by cumulative cost; reduced-VI pruning via eps
  struct Node { int x,y,t; double g; long long parentKey; };
  auto cmp = [](const Node& A, const Node& B){ return A.g > B.g; };
  std::priority_queue<Node, std::vector<Node>, decltype(cmp)> pq(cmp);

  const LocalState s0 = a.getState();
  pq.push(Node{ s0.x, s0.y, 0, 0.0, -1 });

  std::unordered_map<long long, double> best;   best.reserve(200000);
  std::unordered_map<long long, long long> parent; parent.reserve(200000);

  std::vector<std::pair<int,int>> moves = {{0,0},{1,0},{-1,0},{0,1},{0,-1}};
  if (p.allow_diagonal) {
    moves.push_back({+1,+1}); moves.push_back({+1,-1});
    moves.push_back({-1,+1}); moves.push_back({-1,-1});
  }

  auto isSlot = [&](int x,int y)->bool{
    for (const auto &q: S) if (q.x==x && q.y==y) return true;
    return false;
  };

  int iters = 0;
  while (!pq.empty() && iters++ < p.max_iter) {
    Node cur = pq.top(); pq.pop();
    if (cur.t > p.max_time_cap) break;

    long long kcur = TKEY(cur.t, cur.x, cur.y);
    auto itb = best.find(kcur);
    if (itb != best.end() && cur.g >= itb->second - p.eps) {
      continue; // dominated
    }
    best[kcur] = cur.g;

    // Goal test: any unclaimed slot
    if (isSlot(cur.x,cur.y)) {
      long long ks = CKEY(cur.x,cur.y);
      if (!claimed.count(ks)) {
        std::vector<LocalState> path(cur.t + 1);
        long long k = kcur;
        for (int tt = cur.t; tt >= 0; --tt) {
          int t,x,y; UNPACK_TXY(k, t, x, y);
          path[tt] = env.getLocalState(x,y);
          if (tt>0) {
            auto itp = parent.find(k);
            if (itp == parent.end()) {
              for (int r=tt-1; r>=0; --r) path[r] = path[tt];
              break;
            }
            k = itp->second;
          }
        }
        return path;
      }
    }

    // Expand successors
    for (auto d: moves) {
      int nx = cur.x + d.first, ny = cur.y + d.second, nt = cur.t + 1;
      if (nt > p.max_time_cap) continue;

      // Domain hazards (bounds/obstacles/fire vertices + any planner blocks)
      if (!env.isCellFreeForPlanner(nx, ny, nt)) continue;

      // Reservation constraints (inter-agent): no same-cell, no head-on swap
      if (!R.freeCell(nt, nx, ny))  continue;
      if (!R.freeEdge(cur.t, cur.x, cur.y, nx, ny)) continue;

      // Domain edge hazards (fire edges, generic domain blocks, planner blocks)
      if (!env.isEdgeFreeForPlanner(cur.x, cur.y, nx, ny, cur.t)) continue;

      double ng = cur.g + p.step_cost;
      long long knext = TKEY(nt, nx, ny);

      auto itn = best.find(knext);
      if (itn != best.end() && ng >= itn->second - p.eps) continue;

      parent[knext] = kcur;
      pq.push(Node{ nx, ny, nt, ng, kcur });
    }
  }

  // Fallback: wait-in-place
  std::vector<LocalState> trivial(1, s0);
  return trivial;
}

// ---------------------------------------------
// Plan one formation (sequential, with padding)
// ---------------------------------------------
static void
_planFormationWithARVI(Formation& F,
                       Environment& env,
                       const ARVIParams& p,
                       ReservationTable& R_global,
                       std::unordered_set<long long>& claimed_global)
{
  // 1) Stencil slots
  auto S = getStencilSlotsForFormationImpl(F, env);

  // 2) Order agents (closest to landmark first)
  auto ags = F.agentsInFormation;
  std::sort(ags.begin(), ags.end(), [&](const Agent& a, const Agent& b){
    auto sa=a.getState(), sb=b.getState(), g=F.landmarkGoalState;
    int d1 = std::abs(sa.x-g.x) + std::abs(sa.y-g.y);
    int d2 = std::abs(sb.x-g.x) + std::abs(sb.y-g.y);
    return d1 < d2;
  });

  // 3) Sequential ARVI with GLOBAL reservations/claims
  for (auto& ag : ags) {
    auto path = runARVIToAnySlot(ag, S, claimed_global, env, R_global, p);
    if (path.empty()) path.push_back(ag.getState());

    auto goal = path.back();
    claimed_global.insert(CKEY(goal.x, goal.y));
    R_global.reservePath(path);

    for (auto &a : F.agentsInFormation) if (a.getId()==ag.getId()){
      for (auto &s : path) a.trajectory.push_back(s);
      break;
    }
  }

  // 4) Pad early finishers to align time for simultaneous observe()
  size_t maxLen = 0;
  for (auto &a : F.agentsInFormation) maxLen = std::max(maxLen, a.trajectory.size());
  for (auto &a : F.agentsInFormation) while (a.trajectory.size()<maxLen) a.trajectory.push_back(a.trajectory.back());

  // 5) Bookkeeping mirrors your MAPF code
  std::set<int> lastCtx = F.augmentedStatesTrajForThisFormation.empty()
    ? env.getBeliefContextSet()
    : F.augmentedStatesTrajForThisFormation.back().contextSet;

  size_t begin_i = F.augmentedStatesTrajForThisFormation.size();
  for (size_t i=begin_i; i<maxLen; ++i) {
    std::vector<LocalState> locals; locals.reserve(F.agentsInFormation.size());
    for (auto &a : F.agentsInFormation) locals.push_back(a.trajectory[i]);
    F.augmentedStatesTrajForThisFormation.emplace_back(JointState(locals), lastCtx);
    F.contextSetTracking.push_back(lastCtx);
  }

  for (auto &a : F.agentsInFormation) a.setState(a.trajectory.back());
  F.currentAugState = F.augmentedStatesTrajForThisFormation.back();
  env.setBeliefContextSet(F.currentAugState.contextSet);
}

// Public wrapper (intra-formation only)
void planFormationWithARVI(Formation& F, Environment& env, const ARVIParams& p)
{
  ReservationTable R_local;
  std::unordered_set<long long> claimed_local;
  _planFormationWithARVI(F, env, p, R_local, claimed_local);
}

// Public wrapper (global safety across formations)
void planAllFormationsWithARVI(std::vector<Formation>& formations, Environment& env, const ARVIParams& p)
{
  ReservationTable Rg;
  std::unordered_set<long long> claimedG;
  for (auto &F : formations) {
    _planFormationWithARVI(F, env, p, Rg, claimedG);
  }
}



// === ARVI baseline: move each formation to its stencil with reduced-VI (no MAPF) ===
void getToFormationForAllAgentsUsingARVI(std::vector<Formation>& formations,
                                                Environment& env,
                                                const ARVIParams& ap)
{
    // Uses ARVI.cpp's global planner (handles inter-formation reservations)
    planAllFormationsWithARVI(formations, env, ap);
}

// Stage 1 with ARVI: go-to-stencil via ARVI, then do one synchronized "observe"
void performFormation2BeliefCollapseARVI(std::vector<Formation>& formations, Environment& env){
    ARVIParams ap;
    // 1) Move formations to their stencil slots via ARVI (no fixed horizon)
    planAllFormationsWithARVI(formations, env, ap);

    // 2) Synchronized observe step (mirrors your MAPF variant)
    for (auto &formation : formations) {
        std::vector<std::string> formationAction =
            convertToFormationAction("observe", (int)formation.agentsInFormation.size());

        AugmentedState AState = formation.currentAugState;              // set by ARVI planner
        AState = formation.do_augmented_state_action(AState, formationAction);

        for (size_t j = 0; j < formation.agentsInFormation.size(); ++j) {
            // usually position unchanged by "observe", but we record a step for parity
            formation.agentsInFormation[j].trajectory.push_back(AState.joint.states[j]);
            formation.agentsInFormation[j].setState(formation.agentsInFormation[j].trajectory.back());
        }
        formation.augmentedStatesTrajForThisFormation.emplace_back(JointState(AState.joint.states), AState.contextSet);
        formation.contextSetTracking.push_back(AState.contextSet);
        formation.currentAugState = formation.augmentedStatesTrajForThisFormation.back();
    }

    // // 3) Keep your global bookkeeping consistent (same helper you already use)
    // std::vector<Agent> allAgents;
    // for (auto &F : formations) for (auto &a : F.agentsInFormation) allAgents.push_back(a);
    // padFormationAugTrajectoriesToSameLength(formations, env, allAgents);
}


// ---------- Helpers used by VI ----------
// static inline long long CKEY(int x,int y){ return ((long long)x<<32) ^ (unsigned long long)y; }

static inline bool isAvailableSlotXY(int x, int y,
                                     const std::vector<LocalState>& S,
                                     const std::unordered_set<long long>& claimed) {
  for (const auto& q : S) {
    if (q.x == x && q.y == y) {
      return !claimed.count(CKEY(x,y));
    }
  }
  return false;
}

// ---------- Bellman VI on time-expanded state space ----------
std::vector<LocalState>
runARVIToAnySlot_VI(const Agent& a,
                    const std::vector<LocalState>& S,
                    const std::unordered_set<long long>& claimed,
                    Environment& env,
                    const ReservationTable& R,
                    const ARVIParams& p)
{
  const int W = env.getWidth();
  const int H = env.getHeight();
  const int T = std::max(1, p.max_time_cap);

  auto idx = [&](int t, int x, int y)->size_t {
    return (size_t)t * (size_t)(W*H) + (size_t)y * (size_t)W + (size_t)x;
  };

  const double INF = 1e50;
  const LocalState s0 = a.getState();

  // 5-connected moves: stay, up, down, left, right
  const std::array<std::pair<int,int>,9> moves = {{
    {0,0}, {1,0}, {-1,0}, {0,1}, {0,-1}, {-1,1}, {1,-1}, {-1,-1}, {1,1},
  }};

  // Allocate V for t in [0..T]; we’ll set terminal layer values at all t
  std::vector<double> V( (size_t)(T+1) * (size_t)W * (size_t)H, INF );

  // Mark terminal (absorbing) cost 0 at ALL times for available slots
  for (int t = 0; t <= T; ++t) {
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
      if (!env.isCellFreeForPlanner(x, y, t)) continue; // skip blocked
      if (isAvailableSlotXY(x, y, S, claimed)) {
        V[idx(t,x,y)] = 0.0;
      }
    }
  }

  // Backward dynamic programming in time:
  // V_t(x,y) = min_a [ step_cost + gamma * V_{t+1}(x',y') ], blocked moves skipped.
  for (int t = T-1; t >= 0; --t) {
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
      if (!env.isCellFreeForPlanner(x, y, t)) continue; // current blocked
      // If already terminal at (x,y,t), keep 0
      if (isAvailableSlotXY(x, y, S, claimed)) continue;

      double best = INF;
      for (auto d : moves) {
        const int nx = x + d.first;
        const int ny = y + d.second;
        const int nt = t + 1;

        if (nt > T) continue;
        if (!env.isCellFreeForPlanner(nx, ny, nt)) continue;

        // Reservation constraints (previous agents):
        if (!R.freeCell(nt, nx, ny))  continue;
        if (!R.freeEdge(t, x, y, nx, ny)) continue;

        // Domain edge hazards (e.g., fire edges)
        if (!env.isEdgeFreeForPlanner(x, y, nx, ny, t)) continue;

        const double q = p.step_cost + p.gamma * V[idx(nt, nx, ny)];
        if (q < best) best = q;
      }
      V[idx(t,x,y)] = std::min(V[idx(t,x,y)], best);
    }
  }

  // --------- Recover a greedy (cost-minimizing) path from (s0.x, s0.y, 0) ---------
  std::vector<LocalState> path;
  int x = s0.x, y = s0.y, t = 0;
  if (!env.isCellFreeForPlanner(x, y, 0)) {
    // start blocked -> trivial wait
    path.push_back(s0);
    return path;
  }

  path.push_back(env.getLocalState(x,y));

  while (t < T && !isAvailableSlotXY(x, y, S, claimed)) {
    double best = INF;
    int bx = x, by = y;

    for (auto d : moves) {
      const int nx = x + d.first;
      const int ny = y + d.second;
      const int nt = t + 1;

      if (nt > T) continue;
      if (!env.isCellFreeForPlanner(nx, ny, nt)) continue;
      if (!R.freeCell(nt, nx, ny))  continue;
      if (!R.freeEdge(t, x, y, nx, ny)) continue;
      if (!env.isEdgeFreeForPlanner(x, y, nx, ny, t)) continue;

      const double q = p.step_cost + p.gamma * V[idx(nt, nx, ny)];
      if (q < best) { best = q; bx = nx; by = ny; }
    }

    if (best >= INF/2) break; // dead-end under constraints; stop growing
    x = bx; y = by; ++t;
    path.push_back(env.getLocalState(x,y));
  }

  if (path.empty()) path.push_back(s0);
  return path;
}

// ---------- Formation planning (sequential VI + global reservations) ----------
static void
_planFormationWithARVI_VI(Formation& F,
                          Environment& env,
                          const ARVIParams& p,
                          ReservationTable& R_global,
                          std::unordered_set<long long>& claimed_global)
{
  // 1) Build domain-specific stencil slots for this formation
  auto S = getStencilSlotsForFormationImpl(F, env);

  // 2) Order agents (closest to landmark first)
  auto ags = F.agentsInFormation;
  std::sort(ags.begin(), ags.end(), [&](const Agent& a, const Agent& b){
    auto sa=a.getState(), sb=b.getState(), g=F.landmarkGoalState;
    int d1 = std::abs(sa.x-g.x) + std::abs(sa.y-g.y);
    int d2 = std::abs(sb.x-g.x) + std::abs(sb.y-g.y);
    return d1 < d2;
  });

  // 3) VI per agent, reserving as we go
  for (auto& ag : ags) {
    auto path = runARVIToAnySlot_VI(ag, S, claimed_global, env, R_global, p);
    if (path.empty()) path.push_back(ag.getState());

    auto goal = path.back();
    claimed_global.insert(CKEY(goal.x, goal.y));  // slot is now claimed
    R_global.reservePath(path);                   // and temporally reserved

    for (auto &a : F.agentsInFormation) if (a.getId()==ag.getId()){
      for (auto &s : path) a.trajectory.push_back(s);
      break;
    }
  }

  // 4) Pad to same length for synchronized observe
  size_t maxLen = 0;
  for (auto &a : F.agentsInFormation) maxLen = std::max(maxLen, a.trajectory.size());
  for (auto &a : F.agentsInFormation) while (a.trajectory.size()<maxLen) a.trajectory.push_back(a.trajectory.back());

  // 5) Bookkeeping (same as your MAPF code)
  std::set<int> lastCtx = F.augmentedStatesTrajForThisFormation.empty()
    ? env.getBeliefContextSet()
    : F.augmentedStatesTrajForThisFormation.back().contextSet;

  size_t begin_i = F.augmentedStatesTrajForThisFormation.size();
  for (size_t i=begin_i; i<maxLen; ++i) {
    std::vector<LocalState> locals; locals.reserve(F.agentsInFormation.size());
    for (auto &a : F.agentsInFormation) locals.push_back(a.trajectory[i]);
    F.augmentedStatesTrajForThisFormation.emplace_back(JointState(locals), lastCtx);
    F.contextSetTracking.push_back(lastCtx);
  }

  for (auto &a : F.agentsInFormation) a.setState(a.trajectory.back());
  F.currentAugState = F.augmentedStatesTrajForThisFormation.back();
  env.setBeliefContextSet(F.currentAugState.contextSet);
}

void planFormationWithARVI_VI(Formation& F, Environment& env, const ARVIParams& p)
{
  ReservationTable R_local;
  std::unordered_set<long long> claimed_local;
  _planFormationWithARVI_VI(F, env, p, R_local, claimed_local);
}

void planAllFormationsWithARVI_VI(std::vector<Formation>& formations, Environment& env, const ARVIParams& p)
{
  ReservationTable Rg;
  std::unordered_set<long long> claimedG;
  for (auto &F : formations) {
    _planFormationWithARVI_VI(F, env, p, Rg, claimedG);
  }
}

// Stage 1 baseline: VI + synchronized observe
void performFormation2BeliefCollapseARVI_VI(std::vector<Formation>& formations,
                                            Environment& env)
{
  ARVIParams p;
  // Move to stencil slots via VI (no MAPF)
  planAllFormationsWithARVI_VI(formations, env, p);

  // Synchronized "observe" at the end (same as your MAPF variant)
  for (auto &formation : formations) {
    std::vector<std::string> formationAction =
        convertToFormationAction("observe", (int)formation.agentsInFormation.size());

    AugmentedState AState = formation.currentAugState;
    AState = formation.do_augmented_state_action(AState, formationAction);

    for (size_t j = 0; j < formation.agentsInFormation.size(); ++j) {
      formation.agentsInFormation[j].trajectory.push_back(AState.joint.states[j]);
      formation.agentsInFormation[j].setState(formation.agentsInFormation[j].trajectory.back());
    }
    formation.augmentedStatesTrajForThisFormation.emplace_back(JointState(AState.joint.states), AState.contextSet);
    formation.contextSetTracking.push_back(AState.contextSet);
    formation.currentAugState = formation.augmentedStatesTrajForThisFormation.back();
  }

  // // Align formation trajectories and update collective context
  // std::vector<Agent> allAgents;
  // for (auto &F : formations) for (auto &a : F.agentsInFormation) allAgents.push_back(a);
  // padFormationAugTrajectoriesToSameLength(formations, env, allAgents);
  ARVI_AuditCostsOnFormations(formations, env);
}

// === Compute & store ARVI per-step and total cost vectors ====================
static void computeAndStoreARVICosts(const std::vector<Formation>& formations,
                                     Environment& env)
{
  ARVIParams p;
  env.resetARVICostTracking();

  if (formations.empty()) return;

  // All formations' augmented trajectories have been aligned by padFormationAugTrajectoriesToSameLength
  const size_t L = formations[0].augmentedStatesTrajForThisFormation.size();
  if (L == 0) return;

  env.arviCostTracking.assign(L, std::vector<double>(env.numOfObjectives, 0.0));
  env.arviTotalCost.assign(env.numOfObjectives, 0.0);

  // We treat objectives as:
  //   0: time (per agent per tick => +step_cost)
  //   1: movement (1 if agent moved on this tick, else 0)
  //   2: energy/fuel at arrival cell (uses env.fuelAt)
  // If numOfObjectives < 3, we only fill the first N.
  // If > 3, the extra dims stay 0 unless you extend this function.

  for (size_t t = 1; t < L; ++t) { // t=0 has no transition
    std::vector<double> step(env.numOfObjectives, 0.0);

    for (const auto& F : formations) {
      const auto& prev = F.augmentedStatesTrajForThisFormation[t-1].joint.states;
      const auto& curr = F.augmentedStatesTrajForThisFormation[t  ].joint.states;

      const size_t n = curr.size();
      for (size_t i = 0; i < n; ++i) {
        const bool moved = (curr[i].x != prev[i].x) || (curr[i].y != prev[i].y);

        if (env.numOfObjectives >= 1) {
          // Time: 1 per agent per tick (scaled by p.step_cost to stay consistent with VI)
          step[0] += p.step_cost;
        }
        if (env.numOfObjectives >= 2) {
          // Movement count proxy
          step[1] += moved ? 1.0 : 0.0;
        }
        if (env.numOfObjectives >= 3) {
          // Energy/fuel surrogate at arrival cell
          step[2] += env.fuelAt(curr[i].x, curr[i].y);
        }
      }
    }

    env.arviCostTracking[t] = step;
    for (int d = 0; d < env.numOfObjectives; ++d)
      env.arviTotalCost[d] += step[d];
  }
  // arviCostTracking[0] left as zeros by design.
}

// ======= Global storage for ARVI cost outputs (per-step & totals) =======

static std::vector<ARVICostVec> g_arvi_per_step;  // [t] -> {c1,c2,c3}
static ARVICostVec              g_arvi_totals{0.0,0.0,0.0};


// ==============================
// Safe .cost grid loading/audit
// ==============================


namespace {
  static std::vector<std::vector<double>> C1, C2, C3; // [y][x]
  static int CW = 0, CH = 0;                          // width/height for the loaded grids
  static bool COSTS_READY = false;

  // Robust loader: read all tokens, expect H*W, row-major
  static std::vector<std::vector<double>>
  loadCostGridTokens(const std::string& path, int W, int H) {
    std::ifstream fin(path);
    if (!fin) {
      throw std::runtime_error(std::string("[ARVI cost] cannot open file: ") + path);
    }
    std::vector<double> vals;
    vals.reserve((size_t)W * (size_t)H);

    // read all numbers (whitespace-insensitive)
    double v;
    while (fin >> v) vals.push_back(v);

    if ((int)vals.size() != W * H) {
      std::ostringstream oss;
      oss << "[ARVI cost] token count mismatch in " << path
          << " : got " << vals.size() << " but expected " << (W*H)
          << " (W=" << W << ", H=" << H << ")";
      throw std::runtime_error(oss.str());
    }

    // build [H][W] row-major: y is row, x is col
    std::vector<std::vector<double>> G(H, std::vector<double>(W, 0.0));
    size_t idx = 0;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        G[y][x] = vals[idx++];
      }
    }
    return G;
  }

  static inline double safeAt(const std::vector<std::vector<double>>& G, int x, int y) {
    if (y < 0 || y >= (int)G.size() || x < 0 || x >= (int)G[0].size()) {
      std::ostringstream oss;
      oss << "[ARVI cost] OOB access: (x=" << x << ", y=" << y
          << ") for grid size " << (int)G[0].size() << "x" << (int)G.size();
      throw std::out_of_range(oss.str());
    }
    return G[y][x];
  }

  static inline std::string pathFor(const std::string& mapName, int k) {
    std::ostringstream oss;
    oss << "../maps/" << mapName << "/cost-" << k << ".cost";
    return oss.str();
  }
} // anon

// function to plan for all agents to their goal (agent.getGoalState()) using ARVI as the underlying single-agent planner
std::unordered_map<int, std::vector<LocalState>> performAllAgentsToGoalsUsingARVI(std::vector<Agent>& agents, Environment& env) {
  ARVIParams ap;
  // Plan each agent to its goal using ARVI
  ReservationTable R_global;
  std::unordered_set<long long> claimed_global;

  for (auto &agent : agents) {
      std::vector<LocalState> goalSlot = {agent.getGoal()};
      auto path = runARVIToAnySlot(agent, goalSlot, claimed_global, env, R_global, ap);
      if (path.empty()) path.push_back(agent.getState());

      auto goal = path.back();
      claimed_global.insert(CKEY(goal.x, goal.y));
      R_global.reservePath(path);

      for (auto &s : path) agent.trajectory.push_back(s);
      agent.setState(agent.trajectory.back());
  }
  // Build output map
  std::unordered_map<int, std::vector<LocalState>> agentPaths;
  for (const auto &agent : agents) {
      agentPaths[agent.getId()] = agent.trajectory;
  }
  double total1=0.0, total2=0.0, total3=0.0;
  size_t agents_seen = 0, steps_seen = 0;

  for (const auto& a : agents) {
    ++agents_seen;
    if (a.trajectory.empty()) {
      throw std::runtime_error("[ARVI cost] empty trajectory found during audit");
    }
    for (const auto& s : a.trajectory) {
      ++steps_seen;
      total1 += safeAt(C1, s.x, s.y);
      total2 += safeAt(C2, s.x, s.y);
      total3 += safeAt(C3, s.x, s.y);
    }
  }


  std::vector<double> cost_vec = {total1, total2, total3};
  for(int i=0; i<env.numOfObjectives; ++i){
        env.TotalCostVector[i] = cost_vec[i];
  }
  
  return agentPaths;
}
// Recompute from trajectories and loaded cost grids
void recomputeARVI_CostsFromFiles(const std::vector<Formation>& formations, Environment& env)
{
  ARVI_LoadCostGridsOrDie(env);

  // T = max trajectory length among all agents in all formations
  size_t T = 0;
  for (const auto& F : formations)
    for (const auto& a : F.agentsInFormation)
      T = std::max(T, a.trajectory.size());

  // Reset/resize output buffers
  g_arvi_per_step.assign(T, ARVICostVec{0.0,0.0,0.0});
  g_arvi_totals = static_cast<ARVICostVec>(ARVICostVec{0.0,0.0,0.0});

  if (T == 0) {
    std::cerr << "[ARVI cost] WARNING: no trajectory steps; totals remain {0,0,0}\n";
    return;
  }

  // Sum per-step costs across all agents (shorter trajs are padded by last state)
  for (const auto& F : formations) {
    for (const auto& a : F.agentsInFormation) {
      if (a.trajectory.empty()) continue;
      for (size_t t = 0; t < T; ++t) {
        const LocalState& s = (t < a.trajectory.size()) ? a.trajectory[t]
                                                        : a.trajectory.back();
        const double c1 = safeAt(C1, s.x, s.y);
        const double c2 = safeAt(C2, s.x, s.y);
        const double c3 = safeAt(C3, s.x, s.y);
        g_arvi_per_step[t][0] += c1;
        g_arvi_per_step[t][1] += c2;
        g_arvi_per_step[t][2] += c3;
      }
    }
  }

  // Totals
  for (size_t t = 0; t < T; ++t) {
    g_arvi_totals[0] += g_arvi_per_step[t][0];
    g_arvi_totals[1] += g_arvi_per_step[t][1];
    g_arvi_totals[2] += g_arvi_per_step[t][2];
  }

  std::cerr << "[ARVI cost] recompute totals = {c1=" << g_arvi_totals[0]
            << ", c2=" << g_arvi_totals[1]
            << ", c3=" << g_arvi_totals[2] << "}, T=" << T << "\n";
}

const std::vector<ARVICostVec>& ARVI_GetPerStepCosts() { return g_arvi_per_step; }
ARVICostVec                     ARVI_GetTotalCosts()    { return g_arvi_totals; }




void ARVI_LoadCostGridsOrDie(Environment& env) {
  if (COSTS_READY) return;
  const int W = env.getWidth();
  const int H = env.getHeight();
  if (W <= 0 || H <= 0) {
    throw std::runtime_error("[ARVI cost] invalid env dims: width/height are non-positive");
  }
  const std::string mapName = env.getMapName();

  // load cost-1..3 with the same row-major semantics as your MAPF preprocessor
  C1 = loadCostGridTokens(pathFor(mapName, 1), W, H);
  C2 = loadCostGridTokens(pathFor(mapName, 2), W, H);
  C3 = loadCostGridTokens(pathFor(mapName, 3), W, H);
  CW = W; CH = H; COSTS_READY = true;

  std::cerr << "[ARVI cost] Loaded grids for '" << mapName
            << "' dims " << CW << "x" << CH << "\n";
}

void ARVI_AuditCostsOnFormations(const std::vector<Formation>& formations, Environment& env) {
  ARVI_LoadCostGridsOrDie(env);

  // Walk every agent trajectory; sum c1/c2/c3 with OOB checks
  double total1=0.0, total2=0.0, total3=0.0;
  size_t agents_seen = 0, steps_seen = 0;

  for (const auto& F : formations) {
    for (const auto& a : F.agentsInFormation) {
      ++agents_seen;
      if (a.trajectory.empty()) {
        throw std::runtime_error("[ARVI cost] empty trajectory found during audit");
      }
      for (const auto& s : a.trajectory) {
        ++steps_seen;
        total1 += safeAt(C1, s.x, s.y);
        total2 += safeAt(C2, s.x, s.y);
        total3 += safeAt(C3, s.x, s.y);
      }
    }
  }

  std::vector<double> cost_vec = {total1, total2, total3};
  for(int i=0; i<env.numOfObjectives; ++i){
        env.TotalCostVector[i] += cost_vec[i];
  }
  std::cerr << "[ARVI cost] AUDIT OK: agents=" << agents_seen
            << " steps=" << steps_seen
            << " totals = {c1=" << total1
            << ", c2=" << total2
            << ", c3=" << total3 << "}\n";
}
