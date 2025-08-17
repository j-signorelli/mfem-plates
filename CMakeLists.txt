cmake_minimum_required(VERSION 3.10)

# Define project
project(ae298 VERSION 1.0 LANGUAGES CXX)

# Get MFEM
find_package(MFEM 4.7 REQUIRED CONFIG)
message(STATUS "Found MFEM config: TRUE")

# Add src
add_subdirectory(src)

message(STATUS "Install directory: ${CMAKE_INSTALL_PREFIX}")