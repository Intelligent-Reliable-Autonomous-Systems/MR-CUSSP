#include "agent.h"
#include "action.h"
#include "environment.h"
#include <iostream>

bool Agent::move(const std::string &action, const Environment &env) {
    // Use the do_action function from action module.
    // For simplicity, we can call do_action here.
    // (Alternatively, you could call do_action externally.)
    LocalState newState = do_action(state, action, env);
    // If the new state is different (or even if same), update.
    if (newState.x != state.x || newState.y != state.y || newState.type != state.type) {
        state = newState;
        return true;
    }
    // For "observe" or invalid move, return true if action is valid.
    return true;
}
