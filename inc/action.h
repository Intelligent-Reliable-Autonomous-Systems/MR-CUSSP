#ifndef ACTION_H
#define ACTION_H

#include "state.h"
#include "environment.h"
#include <string>
#include <vector>

// // Forward declaration for Environment.
// class Environment;

// For a generalized number of agents, a joint action is represented as a vector of strings.
typedef std::vector<std::string> JointAction;
// At the local state level, perform an action.
// If action is "observe", return the same state.
// For movement actions, return the new local state as read from the environment's grid.
// If the move is invalid (out-of-bounds), return the same state.
LocalState do_action(const LocalState &s, const std::string &action, const Environment &env);

// Compute the joint action: given a joint state and a joint action (vector of actions),
// compute the resulting joint state by applying do_action for each agent.
JointState do_joint_action(const JointState &js, const std::vector<std::string> &jointAction, const Environment &env);

LocalState do_action_stochastic(const LocalState &s, const std::string &action, const Environment &env);
// Compute the joint action: given a joint state and a joint action (vector of actions),
JointState do_joint_action_stochastic(const JointState &js, const std::vector<std::string> &jointAction, const Environment &env);
// Compute the augmented state transition.
AugmentedState do_augmented_state_action_stochastic(const AugmentedState &augState, const std::vector<std::string> &jointAction, const Environment &env, const AugmentedState &goalAug);

#endif // ACTION_H
