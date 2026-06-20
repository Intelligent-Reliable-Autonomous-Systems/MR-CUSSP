#include "BeliefSet.h"

#include <iostream>
#include <set>

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond "\n"; return 1; } } while (0)

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    using namespace macussp;
    std::set<int> belief = {0, 1, 2, 3, 4};
    std::set<int> obs = {1, 2, 4};
    auto next = belief_intersect(belief, obs);
    CHECK(next.size() == 3);
    CHECK(next.count(1) && next.count(2) && next.count(4));

    CHECK(belief_entropy_cardinality(belief) == 5);
    CHECK(belief_entropy_cardinality(next) == 3);

    std::vector<std::set<int>> trace = {belief, next, {2}};
    CHECK(cumulative_entropy_cardinality(trace) == 5 + 3 + 1);

    std::cout << "test_belief_set: OK\n";
    return 0;
}
