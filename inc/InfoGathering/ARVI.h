#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <cstdint>
#include "agent.h"
#include "state.h"
#include "environment.h"
#include "Definitions.h"
// #include "formation.h"
class Formation;
// -----------------------------
// Public knobs for the baseline
// -----------------------------
struct ARVIParams {
  double step_cost      = 1.0;     // per-step cost (stay costs too)
  double eps            = 0.0;     // (unused in VI version; kept for parity)
  int    max_iter       = 200000;  // (unused in VI version; kept for parity)
  int    max_time_cap   = 2000;    // time expansion depth T (large cap, not a fixed horizon)
  bool   allow_diagonal = false;   // (VI uses 4+stay only; keep for parity)

  // VI-specific
  double gamma          = 0.99;    // discount for infinite-horizon equivalent
  // Note: we use single backward sweep in time; tol/iters not needed
};

// 3-objective cost vector {c1,c2,c3}
using ARVICostVec = std::array<double,3>;
// ---------------------------------------------
// Time-expanded reservation to avoid collisions
// ---------------------------------------------
struct ReservationTable {
  // t -> occupied (x,y)
  std::unordered_map<int, std::unordered_set<long long>> cell;
  // t -> occupied directed edge (x1,y1)->(x2,y2); we also mark the reverse to forbid swaps
  std::unordered_map<int, std::unordered_set<long long>> edge;

  static inline long long ckey(int x,int y) {
    return ( (long long)x << 32 ) ^ (unsigned long long)y;
  }
  static inline long long ekey(int x1,int y1,int x2,int y2) {
    return (ckey(x1,y1) << 1) ^ ckey(x2,y2);
  }

  bool freeCell(int t, int x, int y) const {
    auto it = cell.find(t); if (it == cell.end()) return true;
    return !it->second.count(ckey(x,y));
  }
  bool freeEdge(int t, int x1,int y1,int x2,int y2) const {
    auto it = edge.find(t); if (it == edge.end()) return true;
    return !it->second.count(ekey(x1,y1,x2,y2));
  }

  // Reserve the executed path. Final cell is held "forever" from arrival time onward,
  // so later agents can't steal the slot.
  void reservePath(const std::vector<LocalState>& path) {
    for (int t=0; t+1<(int)path.size(); ++t) {
      auto a=path[t], b=path[t+1];
      cell[t].insert(ckey(b.x,b.y));
      edge[t].insert(ekey(a.x,a.y,b.x,b.y));
      edge[t].insert(ekey(b.x,b.y,a.x,a.y)); // forbid head-on swaps
    }
    if (!path.empty()) {
      int T = (int)path.size()-1; auto g = path.back();
      for (int t=T; t<T+512; ++t) cell[t].insert(ckey(g.x,g.y));
    }
  }
};

// -----------------------------------------------------------------------------
// ARVI for a single agent to reach ANY free stencil slot at any time (no horizon)
// Returns a time-parameterized path (LocalState for t = 0..arrival_t)
// -----------------------------------------------------------------------------
std::vector<LocalState>
runARVIToAnySlot(const Agent& a,
                 const std::vector<LocalState>& stencil_slots,
                 const std::unordered_set<long long>& claimed_slot_keys,  // stencil slots already claimed by earlier agents
                 Environment& env,
                 const ReservationTable& R,
                 const ARVIParams& p);

// -----------------------------------------------------------------------------
// Plan a single formation with ARVI (sequential coordinate descent).
// - Builds/reserves collision-free paths to DISTINCT slots.
// - Pads early finishers to align times (for a simultaneous "observe" next step).
// - Updates the formation's trajectory bookkeeping (augmented states / context).
// -----------------------------------------------------------------------------
void planFormationWithARVI(Formation& F, Environment& env, const ARVIParams& p);

// -----------------------------------------------------------------------------
// Plan all formations with ARVI using a GLOBAL reservation table (inter-formation
// collision avoidance). Writes trajectories back into each formation.
// -----------------------------------------------------------------------------
void planAllFormationsWithARVI(std::vector<Formation>& formations, Environment& env, const ARVIParams& p);
// === Diagnostics for .cost access and totals ===
void ARVI_LoadCostGridsOrDie(Environment& env); // loads cost-1..3 for env.getMapName()
void ARVI_AuditCostsOnFormations(const std::vector<Formation>& formations,
                                 Environment& env);            // sums c1,c2,c3 along ARVI trajectories


void performFormation2BeliefCollapseARVI(std::vector<Formation>& formations,
                                                Environment& env);

// Bellman VI on time-expanded space for a single agent to ANY available slot
std::vector<LocalState>
runARVIToAnySlot_VI(const Agent& a,
                    const std::vector<LocalState>& stencil_slots,
                    const std::unordered_set<long long>& claimed_slot_keys,
                    Environment& env,
                    const ReservationTable& R,
                    const ARVIParams& p);

// Plan one formation sequentially using VI (with global reservations)
void planFormationWithARVI_VI(Formation& F, Environment& env, const ARVIParams& p);

// Plan all formations using VI + one global reservation table
void planAllFormationsWithARVI_VI(std::vector<Formation>& formations, Environment& env, const ARVIParams& p);

// Stage 1 baseline with VI: go-to-stencil via VI + synchronized "observe"
void performFormation2BeliefCollapseARVI_VI(std::vector<Formation>& formations,
                                            Environment& env);
// === Compute ARVI totals as a CostVector using cost-1/2/3 ====================
// - Reads ../maps/<map>/cost-1.cost (or cost1.cost), cost-2.cost, cost-3.cost
// - Assumes trajectories were padded/aligned (padFormationAugTrajectoriesToSameLength)
void computeARVIObjectiveCosts(const std::vector<Formation>& formations,
                               const Environment& env,
                               ARVICostVec& out_total,
                               std::vector<ARVICostVec>* per_step = nullptr);
// === ARVI cost reporting (from .cost files) ==============================
#include <array>

// using ARVICostVector = std::array<double,3>;

// Recompute costs from ../maps/<map>/cost-1/2/3.cost using executed trajectories
void recomputeARVI_CostsFromFiles(const std::vector<Formation>& formations, Environment& env);


// === ARVI cost reporting API ===
// Canonical names:
const std::vector<ARVICostVec>& ARVI_GetPerStepCosts();
ARVICostVec                     ARVI_GetTotalCosts();


std::unordered_map<int, std::vector<LocalState>> performAllAgentsToGoalsUsingARVI(std::vector<Agent>& agents, Environment& env);