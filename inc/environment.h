#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "state.h"
#include "agent.h"
#include "params.h"
#include "DomainCosts.h"
#include <vector>
#include <string>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

enum class DomainType { SALP, ForestFire, Warehouse };

class Environment {
public:
    // ----- Graph primitives -----
    struct Node {
        int id;        // unique vertex id in [0, |V|-1]
        int x, y;      // grid coordinate for convenience
        char type;     // '.', 'S', 'L', '@', ...
    };

private:
    // Dimensions of the parsed map (for bounds checks only)
    int width = 0, height = 0;

    // Graph G = (V, E)
    std::vector<Node> V_;                         // vertices by id
    std::vector<std::vector<int>> adj_;          // adjacency lists (undirected)
    std::unordered_map<long long, int> xy2id_;   // key = ((long long)y<<32) | x  -> node id

    // Obstacles (subset of V) exposed via getter
    std::vector<int> obstacles_;

    // Context machinery (unchanged)
    std::set<int> possibleContexts;
    int trueContext;
    std::string mapName;

    // Belief/context tracking (unchanged API)
    std::unordered_map<LocalState, std::set<int>> landmarkToContextSubsetMap;
    std::set<int> trueContextSet;
    std::set<int> possibleContextSet;
    std::set<int> currentBeliefContextsSet;

public:
    int numOfObjectives = 3;
    int H_MAX = 300; // horizon for fire intercept

    std::vector<std::string> MAP; // for visualization

    // define a struct called MODEL to represent the state space, action space, transition model, and observation model and cost functions for each objective specified bu numofObjectives
    struct MACUSSP {
        int numOfObjectives;
        std::vector<LocalState> StateSpace; // state space
        std::vector<std::string> ActionSpace; // action space
        // transition model: given current state and action, returns next state
        std::function<LocalState(const LocalState&, const std::string&)> TransitionModel;
        // observation model: given current state, returns observation
        std::function<std::string(const LocalState&)> ObservationModel;
        // cost function: given current state and action, returns cost for each objective
        std::function<std::vector<double>(const LocalState&, const std::string&)> CostFunction;
        };
    

    // ---- Construction ----
    Environment(std::string MapName, int numOfGroups) {
        mapName = MapName;
        initializeGraphFromMap();   // builds G=(V,E), obstacles_, xy2id_, etc.
        setLandmarkToContextSubsetMap();
        possibleContexts = PossibleContexts;
        trueContext = TrueContext;
        trueContextSet = {trueContext};
        possibleContextSet = PossibleContexts;
        currentBeliefContextsSet = possibleContexts; // initially full uncertainty
        makeMapGrid();
        std::cout << "\n[e.h line 79: ENV = " << mapName << "].\n";

        if (mapName == "forestfire") {
            std::cout << "[e.h l83" << mapName << "] " << fires_.size() << " fires initialized in forestfire domain.\n";
            setDomain(DomainType::ForestFire);
            std::cout << "[e.h l85" << mapName << "] " << fires_.size() << " fires initialized in forestfire domain.\n";
            loadFuelCostIfAvailable();
            std::cout << "[e.h l87" << mapName << "] " << fires_.size() << " fires initialized in forestfire domain.\n";
            generateKSquareFires(FIRE_HMAX, /*k=*/numOfGroups, /*side_len=*/10, /*speed=*/FIRE_SPEED_FACTOR);
            std::cout << "[e.h l89" << mapName << "] " << fires_.size() << " fires initialized in forestfire domain.\n";
        } else if (mapName == "warehouse") {
            setDomain(DomainType::Warehouse);
        } else {
            setDomain(DomainType::SALP);
        }
    }
    std::vector<std::set<int>> beliefContextSetsTillCollapse;  // to stose the overall belief context sets till collapse trajectory just to analyze later

    // ---- Basic accessors ----
    int getWidth()  const { return width;  }
    int getHeight() const { return height; }

    // Graph access
    inline const std::vector<Node>& vertices() const { return V_; }
    inline const std::vector<int>& neighbors(int id) const { return adj_.at(id); }
    inline const std::vector<int>& obstacles() const { return obstacles_; }
    inline const Node& node(int id) const { return V_.at(id); }

    // Convenience: check whether (x,y) is within parsed bounds
    bool isWithinBounds(int x, int y) const {
        return (x >= 0 && x < width && y >= 0 && y < height);
    }

    // Map (x,y) -> node id; returns -1 if absent (should not happen after load)
    int nodeIdAt(int x, int y) const {
        if (!isWithinBounds(x,y)) return -1;
        long long key = (static_cast<long long>(y) << 32) | static_cast<unsigned>(x);
        auto it = xy2id_.find(key);
        return (it == xy2id_.end()) ? -1 : it->second;
    }

    void makeMapGrid() {
        MAP = std::vector<std::string>(height, std::string(width, ' '));
        for (const auto& n : V_) {
            MAP[n.y][n.x] = n.type;
        }
    }

    // Compatibility with your prior LocalState API ---------------------------
    std::vector<double> TotalCostVector = std::vector<double>(numOfObjectives, 0.0);
    // LocalState view for a coordinate
    LocalState getLocalState(int x, int y) const {
        int id = nodeIdAt(x,y);
        if (id < 0) return LocalState(); // invalid
        const Node& n = V_[id];
        return LocalState(n.x, n.y, n.type);
    }

    // The state is valid if it corresponds to a known node and is not out-of-bounds
    bool isValidState(LocalState s) const {
        if (!isWithinBounds(s.x, s.y)) return false;
        int id = nodeIdAt(s.x, s.y);
        if (id < 0) return false;
        // Optional: require that the type matches what was loaded
        return V_[id].type == s.type;
    }

    // Enumerate all LocalStates derived from vertices (non-obstacle only)
    inline std::vector<LocalState> getLocalStateSpace() const {
        std::vector<LocalState> states;
        states.reserve(V_.size());
        for (const auto& n : V_) {
            if (n.type != '@') {
                states.emplace_back(n.x, n.y, n.type);
            }
        }
        return states;
    }

    // Landmarks (computed from vertices)
    inline std::vector<LocalState> getAllLandmarkLocalStates() const {
        std::vector<LocalState> out;
        for (const auto& n : V_) if (n.type == 'L') out.emplace_back(n.x, n.y, n.type);
        return out;
    }

    void printMap() const {
        for (const auto& row : MAP) {
            std::cout << row << std::endl;
        }
    }

    // Get possible contexts (unchanged)
    inline std::set<int> getPossibleContextSet() const { return possibleContextSet; }

    // Get map name
    inline std::string getMapName() const { return mapName; }

    // Assign IDs to landmarks in a stable order
    inline std::unordered_map<int, LocalState> getLandmarkMap() const {
        std::unordered_map<int, LocalState> m;
        auto Ls = getAllLandmarkLocalStates();
        for (size_t i = 0; i < Ls.size(); ++i) m[static_cast<int>(i)] = Ls[i];
        return m;
    }

    inline std::unordered_map<int, LocalState> getLandmarkIDtoStateMap() const {
        return getLandmarkMap();
    }

    inline std::unordered_map<LocalState, int> getLandmarkStateToIDMap() const {
        std::unordered_map<LocalState, int> m;
        auto Ls = getAllLandmarkLocalStates();
        for (size_t i = 0; i < Ls.size(); ++i) m[Ls[i]] = static_cast<int>(i);
        return m;
    }

    inline std::unordered_map<LocalState, std::string> getLandmarkStateToFormationMap() const {
        std::unordered_map<LocalState, std::string> m;
        if (mapName == "salp") {
            const std::vector<std::string> formations = {"chain","cross"};
            auto Ls = getAllLandmarkLocalStates();
            for (size_t i = 0; i < Ls.size(); ++i) m[Ls[i]] = formations[i % formations.size()];
            return m;
        }
        else if (mapName == "forestfire") {
            const std::vector<std::string> formations = {"spread"}; // your default
            auto Ls = getAllLandmarkLocalStates();
            for (size_t i = 0; i < Ls.size(); ++i) m[Ls[i]] = formations[i % formations.size()];
            return m;
        }
        else if (mapName == "warehouse") { // not implemented yet
            const std::vector<std::string> formations = {"pair-up"};
            auto Ls = getAllLandmarkLocalStates();
            for (size_t i = 0; i < Ls.size(); ++i) m[Ls[i]] = formations[i % formations.size()];
            return m;
        }
        else {
            throw std::runtime_error("[e.h l184]Error: domain "+ mapName + " not recognized.");
        }
    }

    // Context bookkeeping (unchanged signatures)
    inline std::set<int> getBeliefContextSet() const { return currentBeliefContextsSet; }
    inline void setBeliefContextSet(const std::set<int>& ctx) { currentBeliefContextsSet = ctx; }
    inline void setLandmarkToContextSubsetMap() { landmarkToContextSubsetMap = getLandmarkStates2ContextSubsetMap(); }
    inline void loadManualLandmarkToContextSubsetMap() {
        landmarkToContextSubsetMap = getLandmarkStates2ContextSubsetMap_manually();
    }
    inline std::unordered_map<LocalState, std::set<int>> getLandmarkToContextSubsetMap() const { return landmarkToContextSubsetMap; }

    // Mark pct% of landmarks as redundant: observation equals full context set so
    // visiting them does not shrink belief entropy (Fig. 3 column iii).
    void applyRedundantLandmarkPct(double pct, unsigned seed = 0) {
        if (pct <= 0.0) return;
        const auto landmarks = getAllLandmarkLocalStates();
        const size_t n = landmarks.size();
        if (n == 0) return;
        size_t k = static_cast<size_t>(std::ceil(n * pct / 100.0));
        if (k > n) k = n;

        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(seed == 0 ? 1u : seed);
        std::shuffle(order.begin(), order.end(), rng);

        for (size_t i = 0; i < k; ++i) {
            landmarkToContextSubsetMap[landmarks[order[i]]] = possibleContexts;
        }
    }

    // Hardcoded landmark-to-context-subset mapping per domain
    inline std::unordered_map<LocalState, std::set<int>>  getLandmarkStates2ContextSubsetMap_manually() const {
        std::vector<LocalState> Ls = getAllLandmarkLocalStates();
        if (mapName == "salp") {
            std::vector<std::set<int>> contextSubsets = {{1,2,4}, {0,2,3}, {1,2,4}, {1,2,3}};
            std::unordered_map<LocalState, std::set<int>> M;
            for (size_t i = 0; i < Ls.size(); ++i) M[Ls[i]] = contextSubsets[i % contextSubsets.size()];
            
            // M[(24,5,'L')] = {0,2,3};
            // M[(20,15,'L')] = {1,2,3};
            // M[(4,20,'L')] = {0,1,2,4};
            // M[(5,4,'L')] = {0,1,2,4};
            // M[(25,22,'L')] = {0,2,3};
            // M[(5,13,'L')] = {1,2,4};
            // M[(8,26,'L')] = {0,2,4};
            // M[(22,27,'L')] = {1,2,3};
        
        return M;
        }
        else if (mapName == "forestfire") {
            std::vector<std::set<int>> contextSubsets = {{1,2,4}, {0,2,3}, {1,2,4}, {1,2,3}};
            std::unordered_map<LocalState, std::set<int>> M;
            for (size_t i = 0; i < Ls.size(); ++i) M[Ls[i]] = contextSubsets[i % contextSubsets.size()];
            return M;
        }
        else if (mapName == "warehouse") {  
            std::vector<std::set<int>> contextSubsets = { {0,2,3}, {1,2,3},{1,2,4}};
            std::unordered_map<LocalState, std::set<int>> M;
            for (size_t i = 0; i < Ls.size(); ++i) M[Ls[i]] = contextSubsets[i % contextSubsets.size()];
            return M;
        }
        else {
            throw std::runtime_error("[e.h l217]Error: domain "+ mapName + " not recognized.");
        }
    }
    // Hardcoded landmark-to-context-subset mapping per domain
    inline std::unordered_map<LocalState, std::set<int>>  getLandmarkStates2ContextSubsetMap() const {
        std::vector<LocalState> Ls = getAllLandmarkLocalStates();
        if (mapName == "salp") {
        std::vector<std::set<int>> contextSubsets = {{1,2,4}, {0,2,3}, {1,2,4}, {1,2,3}};
        std::unordered_map<LocalState, std::set<int>> M;
        for (size_t i = 0; i < Ls.size(); ++i) M[Ls[i]] = contextSubsets[i % contextSubsets.size()];
        return M;
        }
        else if (mapName == "forestfire") {
            std::vector<std::set<int>> contextSubsets = {{1,2,4}, {0,2,3}, {1,2,4}, {1,2,3}};
            std::unordered_map<LocalState, std::set<int>> M;
            for (size_t i = 0; i < Ls.size(); ++i) M[Ls[i]] = contextSubsets[i % contextSubsets.size()];
            return M;
        }
        else if (mapName == "warehouse") {  // not implemented yet
            std::vector<std::set<int>> contextSubsets = {{1,2,4}, {0,2,3}, {1,2,4}, {1,2,3}};//{{0,2,3,4},{0,2,3,4},{0,2,4},{2,3}};
            std::unordered_map<LocalState, std::set<int>> M;
            for (size_t i = 0; i < Ls.size(); ++i) M[Ls[i]] = contextSubsets[i % contextSubsets.size()];
            return M;
        }
        else {
            throw std::runtime_error("[e.h l217]Error: domain "+ mapName + " not recognized.");
        }
    }

    // Joint/augmented state functions (unchanged)
    JointState getJointState(const std::vector<Agent>& agents) const {
        std::vector<LocalState> ls; ls.reserve(agents.size());
        for (const auto& a : agents) ls.push_back(a.getState());
        return JointState(ls);
    }

    AugmentedState getAugmentedState(const std::vector<Agent>& agents,
                                     const std::set<int>& currentContext) const {
        return AugmentedState(getJointState(agents), currentContext);
    }

    AugmentedState getGoalAugmentedState(const std::vector<Agent>& agents) const {
        std::vector<LocalState> states;
        states.reserve(agents.size());
        for (const auto& a : agents) states.push_back(a.getGoal());
        JointState js(states);
        std::set<int> goalContext{trueContext};
        return AugmentedState(js, goalContext);
    }

    AugmentedState getLandmarkAugmentedState(const std::vector<Agent>& agents) const {
        // If you had a special definition before, keep it here; placeholder matches prior pattern.
        return getGoalAugmentedState(agents);
    }

    int getTrueContext() const { return trueContext; }

    const macussp::DomainObjectiveGrids& ensureDomainObjectiveGrids() const;

    // Mutator retained for compatibility; updates node type in G and obstacle set.
    void setCell(int x, int y, char type);

    // ---- Graph construction ----
    void initializeGraphFromMap(); // builds V_, adj_, obstacles_, xy2id_, width/height

    // (Optional) utility: degree of a node
    int degree(int id) const { return static_cast<int>(adj_.at(id).size()); }

    // Public tracking vectors as before
    std::vector<std::set<int>> contextTracking;
    std::vector<std::set<int>> collectiveContextSetTracking;


    void setDomain(DomainType d) { domain_ = d; }
    DomainType domain() const { return domain_; }

    int fire_move_period = 8; // default value, can be changed in initDomainFeatures

    // Fire API (safe NO-OPs unless ForestFire is active and fire is enabled)
    bool loadFireTrajectory(const std::string& path);
    bool fireEnabled() const { return fire_.active; }
    std::pair<int,int> firePosAt(int t) const; // implement in .cpp
    int  fireVidAt(int t) const;               // implement in .cpp
    void reserveFireCAT(std::vector<std::unordered_set<int>>& vertex_cat, int t0, int t1) const; // implement in .cpp

    void loadFuelCostIfAvailable();     // ../maps/<mapName>/<mapName>.cost
    double fuelAt(int x, int y) const;  // returns 1.0 if no cost loaded

    struct Fire {
        std::vector<int> xs, ys;          // optional offline trajectory
        bool active{false};
        bool has_t(int t) const {
            return active && t >= 0 && t < (int)xs.size();
        }
        int id;
    } fire_;

    bool blocksAgentMoveAt(int cur_vid, int next_vid, int t_abs) const;

    // === Add below the existing Fire struct ===
    struct FireTraj {
        std::vector<int> xs, ys;
        bool active{false};
        bool has_t(int t) const { 
            return active && t >= 0 && t < (int)xs.size(); }

        int id;
        int time_of_extinguish{ -1 }; // time step when the fire is extinguished, -1 if not extinguished
    };

    // --- Snapshot of a fire at absolute time t ---
    struct FireState {
        int  x{-1};
        int  y{-1};
        bool extinguished{false};
    };

    // Queries
    FireState fireStateAt(int fire_id, int t_abs) const;
    bool      fireActiveAt(int fire_id, int t_abs) const;

    // Fast membership used by the planner
    bool isOnFireVertex(int t_abs, int vid) const;
    bool isOnFireEdge  (int t_abs, int u_vid, int v_vid) const;

    // Reset caches at the start of each (re)plan
    void clearFireCache() const;

    // Keep fire_ for backward compatibility, but also support k fires.
    std::vector<FireTraj> fires_;

    inline int numFires() const {
        // if legacy fire_ is active and fires_ is empty, treat as one fire
        if (!fires_.empty()) return (int)fires_.size();
        return fire_.active ? 1 : 0;
    }

    // Overloads that are fire-indexed (0..numFires-1).
    std::pair<int,int> firePosAt(int f, int t) const;
    int  fireVidAt(int f, int t) const;

    // Convenience: all fire vertex-ids at time t (for collision/forbids)
    std::vector<int> fireVidsAt(int t) const;

    // Replace previous single-fire semantics with “forbid any fire cell at t”
    inline bool isVertexFreeAtTime(int v, int t) const {
        if (!(domain_ == DomainType::ForestFire)) return true;
        if (t < 0) return false;
        auto vids = fireVidsAt(t);
        return std::find(vids.begin(), vids.end(), v) == vids.end();
    }

    // Generate k independent square-loop fires (reuses your stencil generator)
    void generateKSquareFires(int Hmax, int k, int side_len, float fire_speed_factor);

    void setFireExtinguishTime(int fire_id, int t){
        if (fire_id < 0 || fire_id >= (int)fires_.size()) {
            throw std::out_of_range("Invalid fire_id in setFireExtinguishTime");
        }
        fires_[fire_id].time_of_extinguish = t;
        clearFireCache();
    }

    // === Planner-facing occupancy API (static+dynamic domain hazards, NOT agent reservations) ===
    // Returns true if (x,y) is a static obstacle (e.g., wall '@') or out of bounds.
    bool isStaticObstacle(int x, int y) const;

    // Returns true iff cell (x,y) is usable for the planner at time t:
    // - in bounds
    // - not a static obstacle
    // - not occupied by domain hazards at time t (e.g., fire vertex)
    bool isCellFreeForPlanner(int x, int y, int t) const;

    // Returns true iff moving (x1,y1)->(x2,y2) at time t is allowed by domain hazards:
    // - not a time-varying blocked edge (e.g., fire edge)
    // - not blocked by generic domain rules (blocksAgentMoveAt)
    bool isEdgeFreeForPlanner(int x1, int y1, int x2, int y2, int t) const;

    // Optional: ad-hoc planner blocklist (if you want to mark cells/edges off-limits externally)
    // NOTE: ARVI does NOT use these; it uses its own ReservationTable. Keep for future use.
    void plannerBlockCell(int x, int y, int t) const;
    void plannerUnblockCell(int x, int y, int t) const;
    void plannerBlockEdge(int x1, int y1, int x2, int y2, int t) const;
    void plannerUnblockEdge(int x1, int y1, int x2, int y2, int t) const;
    void plannerClearBlocks() const;

    // ---- ARVI baseline cost tracking (sum over all agents, per time step) ----
    std::vector<std::vector<double>> arviCostTracking; // shape: [t][obj], t aligned to formation trajectories
    std::vector<double>              arviTotalCost;    // size = numOfObjectives, cumulative sum over t

    inline void resetARVICostTracking() {
        arviCostTracking.clear();
        arviTotalCost.assign(numOfObjectives, 0.0);
    }
    

private:
    DomainType domain_{DomainType::SALP}; // default

    std::vector<double> fuelGrid_; // size width*height, row-major; empty => uniform 1.0
    std::vector<std::pair<int,int>> findFireSeeds(int k) const;  // Find all seeds cell to start fire: 'F' on the map representes the number and locations of starting fires

    // Cache: t_abs -> set of burning vertices (vids)
    mutable std::unordered_map<int, std::unordered_set<int>> cache_fire_vids_;
    // Cache: t_abs -> set of burning directed edges (u->v) the fire uses from t-1 -> t
    mutable std::unordered_map<int, std::unordered_set<long long>> cache_fire_edges_;

    static inline long long packEdge(int u, int v) {
        return ( (static_cast<long long>(u) << 32) ^ static_cast<unsigned>(v) );
    }

    // Planner ad-hoc blocks (optional feature; safe if never used)
    mutable std::unordered_map<int, std::unordered_set<int>> planner_block_vids_;   // t -> set(vid)
    mutable std::unordered_map<int, std::unordered_set<long long>> planner_block_edges_; // t -> set(packed u->v)

    mutable bool domain_costs_ready_{false};
    mutable macussp::DomainObjectiveGrids domain_objective_grids_;

};

#endif // ENVIRONMENT_H
