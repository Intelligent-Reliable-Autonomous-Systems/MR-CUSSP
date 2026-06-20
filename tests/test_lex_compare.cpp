#include "LexCompare.h"

#include <cassert>
#include <iostream>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond "\n"; return 1; } } while (0)

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    using namespace macussp;
    const ObjectiveOrdering ord = {0, 1, 2};

    CostVector a = {1, 5, 3};
    CostVector b = {2, 1, 9};
    CHECK(lex_compare(a, b, ord) == LexCompareResult::Less);
    CHECK(lex_compare(b, a, ord) == LexCompareResult::Greater);
    CHECK(lex_compare(a, a, ord) == LexCompareResult::Equal);

    const ObjectiveOrdering ord2 = {1, 0, 2};
    CostVector c = {9, 1, 0};
    CostVector d = {1, 2, 0};
    CHECK(lex_compare(c, d, ord2) == LexCompareResult::Less);

    std::cout << "test_lex_compare: OK\n";
    return 0;
}
