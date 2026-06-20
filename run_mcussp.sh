#!/bin/bash
# Script to configure, build, and run the MA-CUSSP project with LCBS.

# Exit immediately if any command fails
set -e

# Move to project root (assumed to be the script's location)
cd "$(dirname "$0")"

# Build directory
BUILD_DIR="build"

# Create build directory if not present
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Get macOS SDK path (for macOS users)
if [[ "$OSTYPE" == "darwin"* ]]; then
    SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
    echo "Using macOS SDK: $SDK_PATH"
    CMAKE_OPTIONS="-DCMAKE_OSX_SYSROOT=$SDK_PATH -DCMAKE_CXX_STANDARD=17"
else
    echo "Non-macOS system detected."
    CMAKE_OPTIONS="-DCMAKE_CXX_STANDARD=17"
fi

# # Run CMake
cmake .. $CMAKE_OPTIONS

# Compile using all available cores
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# Binary name — adjust if different
# BINARY="./bin/direct_lcbs"
BINARY="./bin/macussp_solver"

# Run binary if exists
if [ -x "$BINARY" ]; then
    echo "Running $BINARY..."
    "$BINARY"
else
    echo "Error: $BINARY executable not found."
    exit 1
fi
