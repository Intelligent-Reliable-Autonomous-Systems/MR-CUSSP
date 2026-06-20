#ifndef STATE_H
#define STATE_H

#include <vector>
#include <string>
#include <sstream>
#include <functional> // for std::hash
#include <cmath>
#include <iostream>
#include <set>       // include for std::set

// A local state is simply (x, y, type)
// where type = 'S' for normal, 'L' for landmark, 'G' for goal.
struct LocalState {
    int x, y;
    char type;  // 'S', 'L', or 'G'
    LocalState(int x_=0, int y_=0, char t = 'S') : x(x_), y(y_), type(t) {}
};

namespace std {
    template <>
    struct hash<LocalState> {
        std::size_t operator()(const LocalState &s) const {
            std::size_t h1 = std::hash<int>()(s.x);
            std::size_t h2 = std::hash<int>()(s.y);
            std::size_t h3 = std::hash<char>()(s.type);
            // Combine the three hash values.
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

// The joint state for d agents is stored as a vector of LocalState.
struct JointState {
    std::vector<LocalState> states;
    JointState(const std::vector<LocalState> &s) : states(s) {}
    
    // Returns a unique string representation for the joint state.
    std::string toString() const {
        std::stringstream ss;
        for (size_t i = 0; i < states.size(); ++i) {
            ss << "(" << states[i].x << ", " << states[i].y << ", " << states[i].type << ")";
            if(i < states.size()-1)
                ss << ",";
        }
        return ss.str();
    }
};

// The augmented state now appends a "context" that is represented as an ordered set of ints.
// In this revised model, the context might be any subset of the available contexts.
struct AugmentedState {
    JointState joint;
    std::set<int> contextSet;  // Now an ordered set instead of a single integer.
    
    // Constructor: supply a joint state and a context (as an ordered set)
    AugmentedState(const JointState &js, const std::set<int> &ctx)
        : joint(js), contextSet(ctx) {}
        
    // Default constructor.
    AugmentedState(): joint(JointState(std::vector<LocalState>())), contextSet(std::set<int>()) {}
    
    // Returns a unique string representation for the augmented state.
    std::string toString() const {
        std::stringstream ss;
        ss << "(" << joint.toString() << ", {";
        bool first = true;
        for (const auto &c : contextSet) {
            if (!first)
                ss << ", ";
            ss << c;
            first = false;
        }
        ss << "})";
        return ss.str();
    }
};

// Equality operator for LocalState.
inline bool operator==(const LocalState &a, const LocalState &b) {
    return (a.x == b.x && a.y == b.y && a.type == b.type);
}

// Inequality operator for LocalState.
inline bool operator!=(const LocalState &a, const LocalState &b) {
    return !(a == b);
}

// Equality operator for JointState.
inline bool operator==(const JointState &a, const JointState &b) {
    if(a.states.size() != b.states.size())
        return false;
    for (size_t i = 0; i < a.states.size(); ++i) {
        if(a.states[i] != b.states[i])
            return false;
    }
    return true;
}

// Equality operator for AugmentedState using the new ordered context.
inline bool operator==(const AugmentedState &a, const AugmentedState &b) {
    return (a.joint == b.joint && a.contextSet == b.contextSet);
}

// Prints a local state (including its x, y, and type)
inline void printLocalState(const LocalState &s) {
    std::cout << "(" << s.x << "," << s.y << ",'" << s.type << "')";
}

// Prints the joint state for a joint state.
inline void printJointState(const JointState &js) {
    std::cout << "(";
    for (size_t i = 0; i < js.states.size(); ++i) {
        printLocalState(js.states[i]);
        if (i < js.states.size() - 1)
            std::cout << ",";
    }
    std::cout << ")\n";
}

// Prints an augmented state (joint state plus context) in yellow color.
inline void printAugmentedState(const AugmentedState &aug) {
    // ANSI escape code for yellow is "\033[33m", and reset is "\033[0m"
    std::cout << "\033[33m"; 
    std::cout << "{(";
    for (size_t i = 0; i < aug.joint.states.size(); ++i) {
        printLocalState(aug.joint.states[i]);
        if(i < aug.joint.states.size()-1)
            std::cout << ",";
    }
    std::cout << "), {";
    bool first = true;
    for (const auto &c : aug.contextSet) {
        if (!first)
            std::cout << ",";
        std::cout << c;
        first = false;
    }
    std::cout << "}}";
    std::cout << "\033[0m"; // reset to default color
}

#endif // STATE_H
