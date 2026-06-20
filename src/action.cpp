#include "action.h"
#include "environment.h"

// do_action: Returns the next local state after applying an action.
// "observe" returns the same state.
// For movement actions ("up", "down", "left", "right"), if the move is valid,
// returns the new local state as read from the environment's grid; else, returns the same state.
LocalState do_action(const LocalState &s, const std::string &action, const Environment &env) {
    int newX = s.x;
    int newY = s.y;
    if (action == "up") {
        newY = s.y - 1;
    } else if (action == "down") {
        newY = s.y + 1;
    } else if (action == "left") {
        newX = s.x - 1;
    } else if (action == "right") {
        newX = s.x + 1;
    } else if (action == "observe") {
        return s; // Observe action: no change in position.
    } else {
        return s; // Unrecognized action: no change.
    }
    if (env.isWithinBounds(newX, newY))
        return env.getLocalState(newX, newY);
    return s; // If move is out-of-bounds, return the same state.
}

// do_joint_action: Applies do_action to each local state in the joint state,
// given a vector of actions (one per agent).
JointState do_joint_action(const JointState &js, const std::vector<std::string> &jointAction, const Environment &env) {
    std::vector<LocalState> nextStates;
    for (size_t i = 0; i < js.states.size(); ++i) {
        LocalState next = do_action(js.states[i], jointAction[i], env);
        nextStates.push_back(next);
    }
    return JointState(nextStates);
}

// ----------------------------------------------------------------
// Stochastic slide variants

// do_action_stochastic: Returns the next local state after applying an action.
// "observe" returns the same state with full probabilty.
// For movement actions ("up", "down", "left", "right"), but with 0.8 probability of moving, 0.1 to the left and 0.1 to the right.
// with 0.8 probability it does it correctly, 0.1 to the left and 0.1 to the right of the intended direction.
// For example, if the action is "up", it will move up with 0.8 probability, left with 0.1 and right with 0.1.
LocalState do_action_stochastic(const LocalState &s, const std::string &action, const Environment &env) {
    std::unordered_map<int, std::string> index_to_action = {{0, "up"}, {1, "right"}, {2, "down"}, {3, "left"}};
    std::unordered_map<std::string, int> action_to_index = {{"up", 0}, {"right", 1}, {"down", 2}, {"left", 3}};
    // generate probability
    std::vector<double> probabilities = {0.8, 0.1, 0.1};
    // generate random number in [0, 1) and then use the cumulative select if the action corresponding probability
    // decides is the action to take is the one with the index of the action, or will it be (index+1)%4 or (index-1)%4 based on the probability
    double random_number = static_cast<double>(rand()) / RAND_MAX;
    int index = action_to_index[action];
    if (random_number < probabilities[0] || action == "observe") {
        // move in the intended direction
        return do_action(s, action, env);
        } 
    else if (random_number < probabilities[0] + probabilities[1]) {
        // move in the left direction
        return do_action(s, index_to_action[(index -1) % 4], env);
    } else {
        // move in the right direction
        return do_action(s, index_to_action[(index + 1) % 4], env);
    }
    return s; // If move is out-of-bounds, return the same state.
}

// do_joint_action_stochastic: Applies do_action_stochastic to each local state in the joint state,
// given a vector of actions (one per agent).
JointState do_joint_action_stochastic(const JointState &js, const std::vector<std::string> &jointAction, const Environment &env) {
    std::vector<LocalState> nextStates;
    for (size_t i = 0; i < js.states.size(); ++i) {
        LocalState next = do_action_stochastic(js.states[i], jointAction[i], env);
        nextStates.push_back(next);
    }
    return JointState(nextStates);
}

// do_augmented_state_action_stochastic: Given an augmented state and a joint action,
// compute the next augmented state as follows:
//   - Compute the next joint state using do_joint_action_stochastic.
//   - If every agent's action is "observe" and all resulting local states are landmarks (type 'L'),
//     update the context to env.getTrueContext(); otherwise, retain the current context.
AugmentedState do_augmented_state_action_stochastic(const AugmentedState &augState, const std::vector<std::string> &jointAction, const Environment &env, const AugmentedState &goalAug) {
    JointState nextJoint = do_joint_action_stochastic(augState.joint, jointAction, env);
    std::set<int> newContextSet = augState.contextSet;
    bool allObserve = true;
    for (const auto &act : jointAction) {
        if (act != "observe") {
            allObserve = false;
            break;
        }
    }
    if (allObserve) {
        if (nextJoint == goalAug.joint) {
            newContextSet = goalAug.contextSet; 
        }
    }
    return AugmentedState(nextJoint, newContextSet);
}


