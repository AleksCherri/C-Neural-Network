#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

# rm -rf build

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

cd build
./main

# read -rp "Нажмите Enter для выхода..."
