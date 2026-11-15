@echo off

pushd .
if exist "build\" (rmdir /s /q build)
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build
main.exe
pause
popd