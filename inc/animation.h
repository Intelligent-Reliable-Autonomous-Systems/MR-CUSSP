#ifndef ANIMATION_H
#define ANIMATION_H

#include "environment.h"
#include "state.h"
#include <vector>
#include <string>
#include "params.h"

inline void runPythonAnimator(std::string domain, bool saveAnimationToFile) {
    // Append totalNumContexts at the end of scriptPath
    std::stringstream ss;
    int saveAnimation = saveAnimationToFile ? 1 : 0;
    std::string scriptPath = "python3 ../scripts/animate_"+domain+".py";
    ss << scriptPath << " " << PossibleContexts.size() << " " << TrueContext << " " << domain << " " << saveAnimation;
    std::string command = ss.str();
    // Execute the command
    int result = system(command.c_str());
    if (result == 0) {
        std::cout << "[a.h l19]Animation script executed successfully.\n";
    } else {
        std::cerr << "[a.h l21] Error executing animation script.\n";
    }
}

#endif // ANIMATION_H
