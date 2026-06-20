#include "environment.h"
#include "high_level_planner.h"

#include <iostream>
#include <vector>

#ifndef MACUSSP_SOURCE_DIR
#define MACUSSP_SOURCE_DIR "."
#endif

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond "\n"; return 1; } } while (0)

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    Environment env("salp", 1);
    env.loadManualLandmarkToContextSubsetMap();

    Environment env50("salp", 1);
    env50.loadManualLandmarkToContextSubsetMap();
    env50.applyRedundantLandmarkPct(50.0, 0);
    CHECK(env.getLandmarkToContextSubsetMap() != env50.getLandmarkToContextSubsetMap());

    auto seq = high_level_planner::getLandmarkVisitSequence(env);
    CHECK(!seq.empty());
    CHECK(seq.size() == env.getAllLandmarkLocalStates().size());

    // Greedy first pick must shrink belief from 5 to 3 on salp landmarks.
    auto lmCtx = env.getLandmarkToContextSubsetMap();
    std::set<int> inter = lmCtx.at(seq.front());
    CHECK(inter.size() == 3);

    std::cout << "test_landmark_sequence: OK (len=" << seq.size() << ")\n";
    return 0;
}
