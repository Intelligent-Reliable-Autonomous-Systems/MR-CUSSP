#ifndef AGENT_H
#define AGENT_H

#include "state.h"
#include <string>

// Forward declaration of Environment to avoid circular dependency.
class Environment;

class Agent {
private:
    int id;
    LocalState state; // The agent's local state.
    LocalState goal;  // The agent's goal state.
    // the current formation the agent is in
    std::string formation_status = "none"; // The current formation status of the agent.
public:
    Agent(int id_, const LocalState &start, const LocalState &goal_state) : id(id_), state(start), goal(goal_state) {}
    std::vector<std::string> actionSpace = {"up", "down", "left", "right", "observe"}; // The action space for the agent.
    // variable to store the trajectory of the agent
    std::vector<LocalState> trajectory; // The agent's trajectory.
    // Function to set the trajectory of the agent
    void setTrajectory(const std::vector<LocalState> &traj) { trajectory = traj; }
    // Function to get add a vector of LocalStates to the trajectory of the agent
    void addToTrajectory(const std::vector<LocalState> &traj) {
        trajectory.insert(trajectory.end(), traj.begin(), traj.end());
    }
    int getId() const { return id; }
    LocalState getState() const { return state; }
    std::string getStateString() const {
        return "(" + std::to_string(state.x) + "," + std::to_string(state.y) + "," + state.type + ")";
    }
    LocalState getGoal() const { return goal; }
    void setState(const LocalState &s) { state = s; }
    // Move the agent using one of the directions: "up", "down", "left", "right", "observe".
    // Returns true if the move is successful.
    bool move(const std::string &action, const Environment &env);
    // get formation status of the agent
    std::string getFormationStatus() const { return formation_status; }
    // set formation status of the agent
    void setFormationStatus(const std::string &status) { formation_status = status; } // formation status is one of "chain", "ring", "cross", "none"
};

#endif // AGENT_H
