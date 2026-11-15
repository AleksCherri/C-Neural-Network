# C-Neural-Network
Simple CPU-based Neural Network on C.

## How to build

### Windows
`rmdir /s /q build` if `build` repository exists.  
`cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release` if using MinGW GCC.  
`cmake --build build`  
Or run `win-build.bat`.

### Linux
`rm -rf build` if `build` repository exists.  
`cmake -S . -B build`  
`cmake --build build`