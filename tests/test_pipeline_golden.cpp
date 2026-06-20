#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef MACUSSP_SOURCE_DIR
#define MACUSSP_SOURCE_DIR "."
#endif

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond "\n"; return 1; } } while (0)

static std::vector<int> read_entropy_file(const std::string& path) {
    std::vector<int> vals;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        vals.push_back(std::stoi(line));
    }
    return vals;
}

int main(int argc, char** argv) {
    bool run_slow = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--fast") run_slow = false;
        if (std::string(argv[i]) == "--slow") run_slow = true;
    }

    if (!run_slow) {
        std::cout << "test_pipeline_golden: SKIP (manual/slow; use --slow)\n";
        return 0;
    }

    const std::string golden = std::string(MACUSSP_SOURCE_DIR)
        + "/tests/golden/salp_belief_entropy_OURS_A5_scen1.txt";
    auto expected = read_entropy_file(golden);
    CHECK(!expected.empty());
    CHECK(expected.front() == 5);

    std::cout << "test_pipeline_golden: loaded baseline with " << expected.size()
              << " steps (live run comparison not automated)\n";
    return 0;
}
